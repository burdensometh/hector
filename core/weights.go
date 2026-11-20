package core

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

/*
Sample weights
--------------

Weights are representative weights for sampled training records: a row with
weight w stands in for w units of the population, but the row itself is never
copied or duplicated. Learners that support weights scale the data loss
gradient of that row by w; rows with weight zero are excluded from learning
and from weighted metrics.

A dataset may carry weights in two equally valid ways:

  - DataSet.Weights, a slice aligned with DataSet.Samples (the preferred form,
    validated for length), or
  - Sample.Weight on each sample, for callers that build datasets by hand.

When neither is present the dataset is unweighted and learners keep their
legacy code path. DataSet.Weights always wins over Sample.Weight when both are
set.

Regularization conventions are declared by each learner; see the lr package.
*/

const (
	// WeightSourceNone means no weights were supplied; the legacy path applies.
	WeightSourceNone = "none"
	// WeightSourceDataSet means weights came from DataSet.Weights.
	WeightSourceDataSet = "dataset"
	// WeightSourceSample means weights came from per-sample Sample.Weight.
	WeightSourceSample = "sample"
)

// ValidateWeights checks that a weight vector can be used to train or evaluate
// a dataset of sampleCount rows. A nil slice means no weights were supplied and
// is always valid. Weights must be finite, non-negative, and must not all be
// zero.
func ValidateWeights(weights []float64, sampleCount int) error {
	if weights == nil {
		return nil
	}
	if len(weights) != sampleCount {
		return fmt.Errorf("sample weights: got %d weights for %d samples", len(weights), sampleCount)
	}
	if len(weights) == 0 {
		return nil
	}
	total := 0.0
	for i, w := range weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return fmt.Errorf("sample weights: weight at index %d is not finite: %v", i, w)
		}
		if w < 0 {
			return fmt.Errorf("sample weights: weight at index %d is negative: %v", i, w)
		}
		total += w
	}
	if math.IsInf(total, 0) {
		return fmt.Errorf("sample weights: total weight is not finite")
	}
	if total == 0 {
		return fmt.Errorf("sample weights: every weight is zero, nothing would be learned")
	}
	return nil
}

// ValidateSampleWeights is an alias of ValidateWeights.
func ValidateSampleWeights(weights []float64, sampleCount int) error {
	return ValidateWeights(weights, sampleCount)
}

// HasWeights reports whether the dataset carries a DataSet.Weights slice.
func (d *DataSet) HasWeights() bool {
	return d.Weights != nil
}

// SetWeights validates and attaches per-row sample weights to the dataset. The
// weights are copied, so later changes to the caller's slice do not affect the
// dataset. Passing nil removes weights and restores the legacy path.
//
// SetWeights never touches d.Samples: weights are not materialized into extra
// rows.
func (d *DataSet) SetWeights(weights []float64) error {
	if err := ValidateWeights(weights, len(d.Samples)); err != nil {
		return err
	}
	if weights == nil {
		d.Weights = nil
		return nil
	}
	d.Weights = append([]float64(nil), weights...)
	return nil
}

// SetSampleWeights is an alias of SetWeights.
func (d *DataSet) SetSampleWeights(weights []float64) error {
	return d.SetWeights(weights)
}

// ValidateWeights validates the weights currently attached to the dataset. It
// returns nil for an unweighted dataset.
func (d *DataSet) ValidateWeights() error {
	if d.Weights != nil {
		return ValidateWeights(d.Weights, len(d.Samples))
	}
	_, _, err := d.ResolveWeights()
	return err
}

// ResolveWeights returns the effective per-row weights for the dataset together
// with the source they came from. It returns (nil, WeightSourceNone, nil) when
// the dataset is unweighted. Invalid weights (length mismatch, non-finite,
// negative, or all zero) are reported as an error so callers can fail before
// any learning starts.
func (d *DataSet) ResolveWeights() ([]float64, string, error) {
	if d.Weights != nil {
		if err := ValidateWeights(d.Weights, len(d.Samples)); err != nil {
			return nil, WeightSourceDataSet, err
		}
		return d.Weights, WeightSourceDataSet, nil
	}

	weighted := false
	for _, sample := range d.Samples {
		if sample != nil && sample.Weight != 0 {
			weighted = true
			break
		}
	}
	if !weighted {
		return nil, WeightSourceNone, nil
	}

	weights := make([]float64, len(d.Samples))
	for i, sample := range d.Samples {
		if sample != nil {
			weights[i] = sample.Weight
		}
	}
	if err := ValidateWeights(weights, len(d.Samples)); err != nil {
		return nil, WeightSourceSample, err
	}
	return weights, WeightSourceSample, nil
}

// WeightSum returns the sum of the effective sample weights, or 0 when the
// dataset is unweighted.
func (d *DataSet) WeightSum() float64 {
	weights, source, err := d.ResolveWeights()
	if err != nil || source == WeightSourceNone {
		return 0
	}
	total := 0.0
	for _, w := range weights {
		total += w
	}
	return total
}

// WeightSummary describes the effective weights of a dataset. It is used by
// runners to record where the weights came from.
type WeightSummary struct {
	Source         string  `json:"source"`
	Path           string  `json:"path,omitempty"`
	Rows           int     `json:"rows"`
	NonZeroRows    int     `json:"non_zero_rows"`
	ZeroWeightRows int     `json:"zero_weight_rows"`
	WeightSum      float64 `json:"weight_sum"`
	MinWeight      float64 `json:"min_weight"`
	MaxWeight      float64 `json:"max_weight"`
}

// SummarizeWeights builds a WeightSummary for the dataset. It returns an error
// when the dataset's weights are invalid.
func (d *DataSet) SummarizeWeights() (WeightSummary, error) {
	weights, source, err := d.ResolveWeights()
	if err != nil {
		return WeightSummary{}, err
	}
	summary := WeightSummary{Source: source, Rows: len(d.Samples)}
	if source == WeightSourceNone {
		return summary, nil
	}
	summary.MinWeight = math.Inf(1)
	for _, w := range weights {
		if w == 0 {
			summary.ZeroWeightRows++
			continue
		}
		summary.NonZeroRows++
		summary.WeightSum += w
		if w < summary.MinWeight {
			summary.MinWeight = w
		}
		if w > summary.MaxWeight {
			summary.MaxWeight = w
		}
	}
	if summary.NonZeroRows == 0 {
		summary.MinWeight = 0
	}
	return summary, nil
}

// LoadWeights reads one weight per row from a text file. Blank lines and lines
// starting with '#' are ignored. Each line may be one of:
//
//	0.25            a bare weight; rows are taken in file order
//	12	0.25         "<row-id>\t<weight>"
//	12:0.25        "<row-id>:<weight>"
//
// Row ids, when present, must start at 0 and increase by one per record, so a
// weights file cannot be silently reordered against the dataset.
func LoadWeights(path string) ([]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	weights := []float64{}
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		index := len(weights)

		var weightText string
		switch {
		case len(fields) == 1 && strings.Contains(fields[0], ":"):
			parts := strings.SplitN(fields[0], ":", 2)
			if err := checkWeightRowID(parts[0], index, path, lineNo); err != nil {
				return nil, err
			}
			weightText = parts[1]
		case len(fields) == 1:
			weightText = fields[0]
		case len(fields) == 2:
			if err := checkWeightRowID(fields[0], index, path, lineNo); err != nil {
				return nil, err
			}
			weightText = fields[1]
		default:
			return nil, fmt.Errorf("weights file %s line %d: expected '<weight>' or '<row-id> <weight>', got %d fields", path, lineNo, len(fields))
		}

		weight, err := strconv.ParseFloat(strings.TrimSpace(weightText), 64)
		if err != nil {
			return nil, fmt.Errorf("weights file %s line %d: invalid weight %q", path, lineNo, weightText)
		}
		weights = append(weights, weight)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return weights, nil
}

func checkWeightRowID(text string, expected int, path string, lineNo int) error {
	id, err := strconv.Atoi(strings.TrimSpace(text))
	if err != nil {
		return fmt.Errorf("weights file %s line %d: invalid row id %q", path, lineNo, text)
	}
	if id != expected {
		return fmt.Errorf("weights file %s line %d: row id %d out of order, expected %d", path, lineNo, id, expected)
	}
	return nil
}
