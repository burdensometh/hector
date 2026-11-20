package core

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func weightedTestDataSet() *DataSet {
	d := NewDataSet()
	for i := 0; i < 4; i++ {
		s := NewSample()
		s.Label = i % 2
		s.AddFeature(Feature{Id: int64(i), Value: 1.0})
		d.AddSample(s)
	}
	return d
}

func TestValidateWeights(t *testing.T) {
	if err := ValidateWeights(nil, 3); err != nil {
		t.Errorf("nil weights should be valid, got %v", err)
	}
	if err := ValidateWeights([]float64{1, 2, 3}, 2); err == nil {
		t.Error("length mismatch should fail")
	}
	if err := ValidateWeights([]float64{1, 2}, 2); err != nil {
		t.Errorf("aligned weights should be valid, got %v", err)
	}
	if err := ValidateWeights([]float64{0, 0}, 2); err == nil {
		t.Error("all-zero weights should fail")
	}
	if err := ValidateWeights([]float64{1, -1}, 2); err == nil {
		t.Error("negative weight should fail")
	}
	if err := ValidateWeights([]float64{1, math.NaN()}, 2); err == nil {
		t.Error("NaN weight should fail")
	}
	if err := ValidateWeights([]float64{1, math.Inf(1)}, 2); err == nil {
		t.Error("Inf weight should fail")
	}
}

func TestSetWeightsDoesNotDuplicateSamples(t *testing.T) {
	d := weightedTestDataSet()
	before := len(d.Samples)
	if err := d.SetWeights([]float64{1, 2, 0, 0.5}); err != nil {
		t.Fatalf("SetWeights failed: %v", err)
	}
	if len(d.Samples) != before {
		t.Fatalf("SetWeights must not copy rows: %d -> %d", before, len(d.Samples))
	}
	if !d.HasWeights() {
		t.Error("HasWeights should be true after SetWeights")
	}
	// The dataset must own its copy.
	weights := []float64{1, 2, 0, 0.5}
	if err := d.SetWeights(weights); err != nil {
		t.Fatal(err)
	}
	weights[0] = 99
	if d.Weights[0] != 1 {
		t.Error("SetWeights must copy the caller's slice")
	}
}

func TestSetWeightsRejectsBadInput(t *testing.T) {
	d := weightedTestDataSet()
	if err := d.SetWeights([]float64{1, 1}); err == nil {
		t.Error("length mismatch should be rejected")
	}
	if err := d.SetWeights([]float64{0, 0, 0, 0}); err == nil {
		t.Error("all-zero weights should be rejected")
	}
	if d.HasWeights() {
		t.Error("rejected weights must not be attached")
	}
}

func TestResolveWeightsSources(t *testing.T) {
	d := weightedTestDataSet()
	weights, source, err := d.ResolveWeights()
	if err != nil || source != WeightSourceNone || weights != nil {
		t.Fatalf("unweighted dataset: got %v %q %v", weights, source, err)
	}

	if err := d.SetWeights([]float64{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	weights, source, err = d.ResolveWeights()
	if err != nil || source != WeightSourceDataSet || len(weights) != 4 {
		t.Fatalf("dataset weights: got %v %q %v", weights, source, err)
	}

	// Per-sample weights are the alternate form.
	e := weightedTestDataSet()
	for i, s := range e.Samples {
		s.Weight = float64(i + 1)
	}
	weights, source, err = e.ResolveWeights()
	if err != nil || source != WeightSourceSample {
		t.Fatalf("sample weights: got %q %v", source, err)
	}
	if weights[3] != 4 {
		t.Errorf("sample weight not read: %v", weights)
	}
}

func TestResolveWeightsRejectsAllZeroSampleWeights(t *testing.T) {
	e := weightedTestDataSet()
	e.Samples[0].Weight = 1
	e.Samples[1].Weight = -1
	if _, _, err := e.ResolveWeights(); err == nil {
		t.Error("negative per-sample weight should fail")
	}
}

func TestSplitCarriesWeights(t *testing.T) {
	d := weightedTestDataSet()
	if err := d.SetWeights([]float64{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	train := d.Split(func(i int) bool { return i%2 == 0 })
	test := d.Split(func(i int) bool { return i%2 == 1 })
	if len(train.Samples) != 2 || len(train.Weights) != 2 {
		t.Fatalf("train split lost weights: %d samples, %d weights", len(train.Samples), len(train.Weights))
	}
	if train.Weights[0] != 1 || train.Weights[1] != 3 {
		t.Errorf("train weights not carried: %v", train.Weights)
	}
	if test.Weights[0] != 2 || test.Weights[1] != 4 {
		t.Errorf("test weights not carried: %v", test.Weights)
	}
}

func TestAddSampleKeepsWeightsAligned(t *testing.T) {
	d := weightedTestDataSet()
	if err := d.SetWeights([]float64{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	d.AddSample(NewSample())
	if len(d.Weights) != len(d.Samples) {
		t.Fatalf("weights and samples out of sync: %d vs %d", len(d.Weights), len(d.Samples))
	}
}

func TestSummarizeWeights(t *testing.T) {
	d := weightedTestDataSet()
	if err := d.SetWeights([]float64{1, 0, 3, 0.5}); err != nil {
		t.Fatal(err)
	}
	s, err := d.SummarizeWeights()
	if err != nil {
		t.Fatal(err)
	}
	if s.Source != WeightSourceDataSet || s.Rows != 4 || s.NonZeroRows != 3 || s.ZeroWeightRows != 1 {
		t.Errorf("unexpected summary: %+v", s)
	}
	if math.Abs(s.WeightSum-4.5) > 1e-9 || s.MinWeight != 0.5 || s.MaxWeight != 3 {
		t.Errorf("unexpected weight stats: %+v", s)
	}
}

func TestLoadWeights(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.txt")
	content := "# comment\n\n0.5\n1:2.0\n2\t0.25\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	weights, err := LoadWeights(path)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	want := []float64{0.5, 2.0, 0.25}
	if len(weights) != len(want) {
		t.Fatalf("got %v want %v", weights, want)
	}
	for i := range want {
		if weights[i] != want[i] {
			t.Errorf("weight %d: got %v want %v", i, weights[i], want[i])
		}
	}
}

func TestLoadWeightsRejectsBadRows(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"out_of_order": "1:0.5\n0:0.5\n",
		"not_a_number": "abc\n",
		"too_many":     "0:0.5:1\n",
	}
	for name, content := range cases {
		path := filepath.Join(dir, name+".txt")
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadWeights(path); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
