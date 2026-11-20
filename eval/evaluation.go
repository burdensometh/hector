package eval

import (
	"encoding/json"
	"math"
	"sort"
)

type LabelPrediction struct {
	Prediction float64
	Label      int
	// Weight is the representative weight of the row this prediction came
	// from. It is only consulted by the weighted metrics (WeightedAUC,
	// WeightedLogLoss); the plain metrics ignore it so the legacy numbers are
	// unchanged. A zero weight means the row is not part of the evaluated
	// population and is skipped by the weighted metrics.
	Weight float64
}

type RealPrediction struct { // Real valued
	Prediction float64
	Value      float64
}

type By func(p1, p2 *LabelPrediction) bool

type labelPredictionSorter struct {
	predictions []*LabelPrediction
	by          By
}

func (s *labelPredictionSorter) Len() int {
	return len(s.predictions)
}

func (s *labelPredictionSorter) Swap(i, j int) {
	s.predictions[i], s.predictions[j] = s.predictions[j], s.predictions[i]
}

func (s *labelPredictionSorter) Less(i, j int) bool {
	return s.by(s.predictions[i], s.predictions[j])
}

func (by By) Sort(predictions []*LabelPrediction) {
	sorter := &labelPredictionSorter{
		predictions: predictions,
		by:          by,
	}
	sort.Sort(sorter)
}

func AUC(predictions0 []*LabelPrediction) float64 {
	predictions := []*LabelPrediction{}
	for _, pred := range predictions0 {
		predictions = append(predictions, pred)
	}
	prediction := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}

	By(prediction).Sort(predictions)

	pn := 0.0
	nn := float64(len(predictions))
	ret := 0.0
	count := nn
	for i, lp := range predictions {
		if lp.Label > 0 {
			pn += 1.0
			nn -= 1.0
			ret += float64(count) - float64(i)
		}
	}
	ret2 := pn * (pn + 1) / 2.0
	if pn*nn == 0.0 {
		return 0.5
	}
	return (ret - ret2) / (pn * nn)
}

// WeightedAUC returns the area under the ROC curve over weighted
// positive/negative pairs. Each positive/negative pair contributes
// weight(pos)*weight(neg); pairs with equal predictions count as one half.
// Rows with Weight <= 0 are excluded from the population. If either the
// positive or the negative population is empty, it returns 0.5, mirroring AUC.
func WeightedAUC(predictions []*LabelPrediction) float64 {
	type scoredRow struct {
		prediction float64
		label      int
		weight     float64
	}
	rows := []scoredRow{}
	for _, lp := range predictions {
		if lp == nil || lp.Weight <= 0 {
			continue
		}
		rows = append(rows, scoredRow{lp.Prediction, lp.Label, lp.Weight})
	}
	if len(rows) == 0 {
		return 0.5
	}
	sort.SliceStable(rows, func(i, j int) bool {
		return rows[i].prediction < rows[j].prediction
	})

	positive := 0.0
	negative := 0.0
	for _, row := range rows {
		if row.label > 0 {
			positive += row.weight
		} else {
			negative += row.weight
		}
	}
	if positive == 0 || negative == 0 {
		return 0.5
	}

	// Sweep from low to high score. For every tie group, every positive in the
	// group beats every negative in lower groups and ties every negative in the
	// same group.
	ret := 0.0
	negBelow := 0.0
	for i := 0; i < len(rows); {
		j := i
		groupPositive := 0.0
		groupNegative := 0.0
		for j < len(rows) && rows[j].prediction == rows[i].prediction {
			if rows[j].label > 0 {
				groupPositive += rows[j].weight
			} else {
				groupNegative += rows[j].weight
			}
			j++
		}
		ret += groupPositive * (negBelow + 0.5*groupNegative)
		negBelow += groupNegative
		i = j
	}
	return ret / (positive * negative)
}

// LogLoss returns the mean binary log loss. Every prediction is weighted
// equally, matching the unweighted metrics above.
func LogLoss(predictions []*LabelPrediction) float64 {
	sum := 0.0
	n := 0.0
	for _, lp := range predictions {
		sum += logLossOf(lp)
		n += 1
	}
	if n == 0 {
		return math.NaN()
	}
	return sum / n
}

// WeightedLogLoss returns the mean binary log loss over the weighted
// population: sum(w * loss) / sum(w). Rows with Weight <= 0 are excluded. It
// returns NaN when no row carries positive weight.
func WeightedLogLoss(predictions []*LabelPrediction) float64 {
	sum := 0.0
	total := 0.0
	for _, lp := range predictions {
		if lp == nil || lp.Weight <= 0 {
			continue
		}
		sum += lp.Weight * logLossOf(lp)
		total += lp.Weight
	}
	if total == 0 {
		return math.NaN()
	}
	return sum / total
}

// WeightedLogloss is an alias of WeightedLogLoss.
func WeightedLogloss(predictions []*LabelPrediction) float64 {
	return WeightedLogLoss(predictions)
}

func logLossOf(lp *LabelPrediction) float64 {
	const epsilon = 1e-15
	p := lp.Prediction
	if p < epsilon {
		p = epsilon
	} else if p > 1-epsilon {
		p = 1 - epsilon
	}
	if lp.Label > 0 {
		return -math.Log(p)
	}
	return -math.Log(1 - p)
}

func RMSE(predictions []*LabelPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (float64(pred.Label) - pred.Prediction) * (float64(pred.Label) - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}

func ErrorRate(predictions []*LabelPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		if (float64(pred.Label)-0.5)*(pred.Prediction-0.5) < 0 {
			ret += 1.0
		}
		n += 1.0
	}
	return ret / n
}

func RegRMSE(predictions []*RealPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (pred.Value - pred.Prediction) * (pred.Value - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}

// ClassificationMetrics is the evaluation summary attached to a runner's
// weight report. The weighted metrics are only meaningful when HasWeights is
// true; otherwise they mirror the unweighted population.
type ClassificationMetrics struct {
	HasWeights         bool    `json:"has_weights"`
	AUC                float64 `json:"auc"`
	WeightedAUC        float64 `json:"weighted_auc"`
	LogLoss            float64 `json:"logloss"`
	WeightedLogLoss    float64 `json:"weighted_logloss"`
	Population         float64 `json:"population"`
	WeightedPopulation float64 `json:"weighted_population"`
}

// ClassificationMetricsFor computes the metric summary. When hasWeights is
// false every row is treated as having weight 1 so the weighted figures agree
// with the unweighted ones.
func ClassificationMetricsFor(predictions []*LabelPrediction, hasWeights bool) ClassificationMetrics {
	populated := predictions
	if !hasWeights {
		populated = make([]*LabelPrediction, len(predictions))
		for i, lp := range predictions {
			if lp == nil {
				continue
			}
			copied := *lp
			copied.Weight = 1.0
			populated[i] = &copied
		}
	}
	metrics := ClassificationMetrics{
		HasWeights:         hasWeights,
		AUC:                AUC(predictions),
		WeightedAUC:        WeightedAUC(populated),
		LogLoss:            LogLoss(predictions),
		WeightedLogLoss:    WeightedLogLoss(populated),
		Population:         float64(len(predictions)),
		WeightedPopulation: 0,
	}
	for _, lp := range populated {
		if lp != nil && lp.Weight > 0 {
			metrics.WeightedPopulation += lp.Weight
		}
	}
	return metrics
}

// MarshalJSON encodes the metrics so a metric that is undefined (NaN, e.g. the
// log loss of an empty population) is written as null instead of failing the
// report. encoding/json cannot marshal non-finite floats.
func (m ClassificationMetrics) MarshalJSON() ([]byte, error) {
	finite := func(v float64) interface{} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil
		}
		return v
	}
	return json.Marshal(struct {
		HasWeights         bool        `json:"has_weights"`
		AUC                interface{} `json:"auc"`
		WeightedAUC        interface{} `json:"weighted_auc"`
		LogLoss            interface{} `json:"logloss"`
		WeightedLogLoss    interface{} `json:"weighted_logloss"`
		Population         interface{} `json:"population"`
		WeightedPopulation interface{} `json:"weighted_population"`
	}{
		HasWeights:         m.HasWeights,
		AUC:                finite(m.AUC),
		WeightedAUC:        finite(m.WeightedAUC),
		LogLoss:            finite(m.LogLoss),
		WeightedLogLoss:    finite(m.WeightedLogLoss),
		Population:         finite(m.Population),
		WeightedPopulation: finite(m.WeightedPopulation),
	})
}
