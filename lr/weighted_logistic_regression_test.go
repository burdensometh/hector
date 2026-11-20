package lr

import (
	"math"
	"testing"

	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
)

// twoFeatureDataSet returns a dataset with four rows over feature 1. Each row
// contributes equally unless weighted.
func twoFeatureDataSet() *core.DataSet {
	d := core.NewDataSet()
	rows := []struct {
		label int
		x     float64
	}{
		{1, 1.0},
		{1, 1.0},
		{0, -1.0},
		{0, -1.0},
	}
	for _, row := range rows {
		s := core.NewSample()
		s.Label = row.label
		s.AddFeature(core.Feature{Id: 1, Value: row.x})
		d.AddSample(s)
	}
	return d
}

func lrParams() map[string]string {
	return map[string]string{
		"learning-rate":          "0.1",
		"regularization":         "0.0",
		"learning-rate-discount": "1.0",
		"steps":                  "1",
	}
}

func logisticRegressionWith(params map[string]string) *LogisticRegression {
	algo := &LogisticRegression{}
	algo.Init(params)
	return algo
}

// With zero regularization and one step, the first row's update is
// lr * w * err * x. This shows that the weight multiplies the data-loss
// gradient.
func TestLogisticRegressionWeightScalesGradient(t *testing.T) {
	algo := logisticRegressionWith(lrParams())
	dataset := core.NewDataSet()
	sample := core.NewSample()
	sample.Label = 1
	sample.AddFeature(core.Feature{Id: 7, Value: 2.0})
	dataset.AddSample(sample)
	if err := dataset.SetWeights([]float64{3.0}); err != nil {
		t.Fatal(err)
	}

	algo.Train(dataset)

	// prediction is sigmoid(0)=0.5, err=0.5, x=2, lr=0.1, w=3
	want := 0.1 * 3.0 * 0.5 * 2.0
	if got := algo.Model[7]; math.Abs(got-want) > 1e-9 {
		t.Errorf("weighted update = %v, want %v", got, want)
	}
}

func TestLogisticRegressionZeroWeightRowIsSkipped(t *testing.T) {
	algo := logisticRegressionWith(lrParams())
	dataset := core.NewDataSet()
	active := core.NewSample()
	active.Label = 1
	active.AddFeature(core.Feature{Id: 7, Value: 2.0})
	ignored := core.NewSample()
	ignored.Label = 0
	ignored.AddFeature(core.Feature{Id: 99, Value: 10.0})
	dataset.AddSample(active)
	dataset.AddSample(ignored)
	if err := dataset.SetWeights([]float64{3.0, 0.0}); err != nil {
		t.Fatal(err)
	}

	algo.Train(dataset)

	if _, ok := algo.Model[99]; ok {
		t.Error("zero-weight row produced an update")
	}
	want := 0.1 * 3.0 * 0.5 * 2.0
	if got := algo.Model[7]; math.Abs(got-want) > 1e-9 {
		t.Errorf("active update = %v, want %v", got, want)
	}
}

// A zero-weight row must not receive regularization either: if it did, a
// feature seen only by zero-weight rows would appear in the model.
func TestLogisticRegressionZeroWeightRowGetsNoShrinkage(t *testing.T) {
	params := lrParams()
	params["regularization"] = "0.5"
	algo := logisticRegressionWith(params)
	dataset := core.NewDataSet()
	active := core.NewSample()
	active.Label = 1
	active.AddFeature(core.Feature{Id: 7, Value: 1.0})
	ignored := core.NewSample()
	ignored.Label = 1
	ignored.AddFeature(core.Feature{Id: 99, Value: 1.0})
	dataset.AddSample(active)
	dataset.AddSample(ignored)
	if err := dataset.SetWeights([]float64{1.0, 0.0}); err != nil {
		t.Fatal(err)
	}

	algo.Train(dataset)

	if _, ok := algo.Model[99]; ok {
		t.Error("zero-weight row was touched by regularization")
	}
}

// Without weights the update must be exactly the legacy one.
func TestLogisticRegressionUnweightedLegacyUpdate(t *testing.T) {
	params := lrParams()
	params["regularization"] = "0.2"
	algo := logisticRegressionWith(params)
	dataset := core.NewDataSet()
	sample := core.NewSample()
	sample.Label = 1
	sample.AddFeature(core.Feature{Id: 7, Value: 2.0})
	dataset.AddSample(sample)

	algo.Train(dataset)

	want := 0.1 * (0.5*2.0 - 0.2*0.0)
	if got := algo.Model[7]; math.Abs(got-want) > 1e-9 {
		t.Errorf("legacy update = %v, want %v", got, want)
	}

	// The unweighted path must not depend on Sample.Weight or DataSet.Weights.
	weighted := logisticRegressionWith(params)
	weighted_dataset := core.NewDataSet()
	s := core.NewSample()
	s.Label = 1
	s.AddFeature(core.Feature{Id: 7, Value: 2.0})
	weighted_dataset.AddSample(s)
	if err := weighted_dataset.SetWeights([]float64{1.0}); err != nil {
		t.Fatal(err)
	}
	weighted.Train(weighted_dataset)
	if math.Abs(weighted.Model[7]-algo.Model[7]) > 1e-9 {
		t.Errorf("unit weight changed the update: %v vs %v", weighted.Model[7], algo.Model[7])
	}
}

func TestLogisticRegressionInvalidWeightsFailBeforeTraining(t *testing.T) {
	algo := logisticRegressionWith(lrParams())
	if err := algo.TrainWithError(datasetWithInvalidWeights(t)); err == nil {
		t.Error("expected an error for all-zero weights")
	}
	if len(algo.Model) != 0 {
		t.Error("model changed despite invalid weights")
	}
}

func datasetWithInvalidWeights(t *testing.T) *core.DataSet {
	t.Helper()
	d := twoFeatureDataSet()
	// Bypass SetWeights validation to model a corrupt dataset from elsewhere.
	d.Weights = []float64{0, 0, 0, 0}
	return d
}

// A representative weight acts like that many copies of the row evaluated at
// the same model state (a scaled gradient), not like sequential re-visits. This
// test pins the semantics down and, more importantly, verifies training never
// materializes the copies: the dataset keeps exactly as many rows as the caller
// supplied.
func TestLogisticRegressionWeightIsScaledGradientNotRowDuplication(t *testing.T) {
	params := lrParams()
	params["steps"] = "1"
	algo := logisticRegressionWith(params)

	dataset := core.NewDataSet()
	for _, row := range []struct {
		label int
		value float64
	}{{1, 0.3}, {0, -0.2}} {
		s := core.NewSample()
		s.Label = row.label
		s.AddFeature(core.Feature{Id: 1, Value: row.value})
		dataset.AddSample(s)
	}
	rowsBefore := len(dataset.Samples)
	if err := dataset.SetWeights([]float64{3, 1}); err != nil {
		t.Fatal(err)
	}
	algo.Train(dataset)

	if len(dataset.Samples) != rowsBefore {
		t.Fatalf("training duplicated rows: %d -> %d", rowsBefore, len(dataset.Samples))
	}

	// Hand-compute the single-step SGD updates. The rows are visited in order,
	// so the second row sees the model the first row produced. The row's
	// weight only scales its own gradient.
	m := 0.1 * (3 * (1 - util.Sigmoid(0)) * 0.3)
	err2 := 0 - util.Sigmoid(m*-0.2)
	m += 0.1 * (1 * err2 * -0.2)
	want := m
	if got := algo.Model[1]; math.Abs(got-want) > 1e-9 {
		t.Errorf("model = %v, want %v", got, want)
	}
}

func ftrlParams() map[string]string {
	return map[string]string{
		"alpha":   "0.1",
		"beta":    "1",
		"lambda1": "0.0",
		"lambda2": "0.0",
		"steps":   "1",
	}
}

func TestFTRLWeightScalesGradient(t *testing.T) {
	algo := &FTRLLogisticRegression{}
	algo.Init(ftrlParams())

	dataset := core.NewDataSet()
	sample := core.NewSample()
	sample.Label = 1
	sample.AddFeature(core.Feature{Id: 7, Value: 2.0})
	dataset.AddSample(sample)
	if err := dataset.SetWeights([]float64{3.0}); err != nil {
		t.Fatal(err)
	}

	algo.Train(dataset)

	// gi = -w * err * x = -3 * 0.5 * 2 = -3, ni = 9, zi = -3.
	got := algo.Model[7]
	if math.Abs(got.ni-9.0) > 1e-9 || math.Abs(got.zi-(-3.0)) > 1e-9 {
		t.Errorf("weighted FTRL stats = {ni:%v zi:%v}, want {9, -3}", got.ni, got.zi)
	}
}

func TestFTRLZeroWeightRowIsSkipped(t *testing.T) {
	algo := &FTRLLogisticRegression{}
	algo.Init(ftrlParams())

	dataset := core.NewDataSet()
	active := core.NewSample()
	active.Label = 1
	active.AddFeature(core.Feature{Id: 7, Value: 2.0})
	ignored := core.NewSample()
	ignored.Label = 0
	ignored.AddFeature(core.Feature{Id: 99, Value: 10.0})
	dataset.AddSample(active)
	dataset.AddSample(ignored)
	if err := dataset.SetWeights([]float64{1.0, 0.0}); err != nil {
		t.Fatal(err)
	}

	algo.Train(dataset)

	if _, ok := algo.Model[99]; ok {
		t.Error("zero-weight row produced an FTRL update")
	}
}

func TestFTRLUnweightedLegacyUpdate(t *testing.T) {
	algo := &FTRLLogisticRegression{}
	algo.Init(ftrlParams())

	dataset := core.NewDataSet()
	sample := core.NewSample()
	sample.Label = 1
	sample.AddFeature(core.Feature{Id: 7, Value: 2.0})
	dataset.AddSample(sample)

	algo.Train(dataset)

	// Legacy: gi = -err*x = -1, ni = 1, zi = -1.
	got := algo.Model[7]
	if math.Abs(got.ni-1.0) > 1e-9 || math.Abs(got.zi-(-1.0)) > 1e-9 {
		t.Errorf("legacy FTRL stats = {ni:%v zi:%v}, want {1, -1}", got.ni, got.zi)
	}
}

func TestFTRLInvalidWeightsFailBeforeTraining(t *testing.T) {
	algo := &FTRLLogisticRegression{}
	algo.Init(ftrlParams())
	if err := algo.TrainWithError(datasetWithInvalidWeights(t)); err == nil {
		t.Error("expected an error for all-zero weights")
	}
	if len(algo.Model) != 0 {
		t.Error("model changed despite invalid weights")
	}
}
