package hector

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/lr"
)

func writeDatasetFile(t *testing.T, dir, name string, lines []string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeWeightsFile(t *testing.T, dir, name string, weights []float64) string {
	t.Helper()
	lines := make([]string, len(weights))
	for i, w := range weights {
		lines[i] = formatFloat(w)
	}
	return writeDatasetFile(t, dir, name, lines)
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func classifierParams(extra map[string]string) map[string]string {
	params := map[string]string{
		"learning-rate":  "0.1",
		"regularization": "0.0",
		"steps":          "3",
		"global":         "-1",
	}
	for k, v := range extra {
		params[k] = v
	}
	return params
}

func lineTestData() (train, test []string) {
	train = []string{
		"1\t1:1.0",
		"1\t1:0.8",
		"0\t1:-1.0",
		"0\t1:-0.8",
	}
	test = []string{
		"1\t1:0.9",
		"0\t1:-0.9",
	}
	return train, test
}

func TestAttachWeightsRejectsMismatchedLength(t *testing.T) {
	dir := t.TempDir()
	dataset := core.NewDataSet()
	sample := core.NewSample()
	sample.Label = 1
	sample.AddFeature(core.Feature{Id: 1, Value: 1})
	dataset.AddSample(sample)

	path := writeWeightsFile(t, dir, "w.txt", []float64{1, 2})
	if err := AttachWeights(dataset, path); err == nil {
		t.Fatal("expected a length mismatch error")
	}
}

func TestAttachWeightsRejectsAllZero(t *testing.T) {
	dir := t.TempDir()
	dataset := core.NewDataSet()
	for i := 0; i < 2; i++ {
		sample := core.NewSample()
		sample.AddFeature(core.Feature{Id: 1, Value: 1})
		dataset.AddSample(sample)
	}
	path := writeWeightsFile(t, dir, "w.txt", []float64{0, 0})
	if err := AttachWeights(dataset, path); err == nil {
		t.Fatal("expected an all-zero weights error")
	}
	if dataset.HasWeights() {
		t.Error("rejected weights must not be attached")
	}
}

// The runner must fail before training when the training weights are all zero
// or misaligned; a partially trained model is not acceptable.
func TestAlgorithmRunWithReportFailsBeforeTraining(t *testing.T) {
	dir := t.TempDir()
	train, test := lineTestData()
	train_path := writeDatasetFile(t, dir, "train.tsv", train)
	test_path := writeDatasetFile(t, dir, "test.tsv", test)

	cases := map[string]struct {
		weights     []float64
		testWeights []float64
	}{
		"all_zero":     {weights: []float64{0, 0, 0, 0}, testWeights: []float64{1, 1}},
		"too_short":    {weights: []float64{1, 1}, testWeights: []float64{1, 1}},
		"negative":     {weights: []float64{1, -1, 1, 1}, testWeights: []float64{1, 1}},
		"test_missing": {weights: []float64{1, 1, 1, 1}, testWeights: nil},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			params := classifierParams(nil)
			params[ParamsWeightsPath] = writeWeightsFile(t, dir, name+"-train.txt", tc.weights)
			if tc.testWeights != nil {
				params[ParamsTestWeightsPath] = writeWeightsFile(t, dir, name+"-test.txt", tc.testWeights)
			}
			classifier := GetClassifier("lr")
			_, _, _, err := AlgorithmRunWithReport(classifier, train_path, test_path, "", params)
			if err == nil {
				t.Fatal("expected the run to fail before training")
			}
			model := classifier.(*lr.LogisticRegression).Model
			if len(model) != 0 {
				t.Errorf("classifier was trained despite invalid weights: %v", model)
			}
		})
	}
}

// Training and evaluation must use the same weighted population.
func TestCheckSamePopulation(t *testing.T) {
	weighted := core.NewDataSet()
	unweighted := core.NewDataSet()
	for i := 0; i < 2; i++ {
		s := core.NewSample()
		s.Label = i
		s.AddFeature(core.Feature{Id: 1, Value: 1})
		weighted.AddSample(s.Clone())
		unweighted.AddSample(s.Clone())
	}
	if err := weighted.SetWeights([]float64{1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := CheckSamePopulation(weighted, unweighted); err == nil {
		t.Error("weighted train / unweighted test should fail")
	}
	if err := CheckSamePopulation(unweighted, weighted); err == nil {
		t.Error("unweighted train / weighted test should fail")
	}
	if err := CheckSamePopulation(unweighted, unweighted); err != nil {
		t.Errorf("both unweighted should pass: %v", err)
	}
	if err := CheckSamePopulation(weighted, weighted); err != nil {
		t.Errorf("both weighted should pass: %v", err)
	}
}

// A weighted run writes a provenance report naming the weight source, the
// weight files and the metric definitions.
func TestAlgorithmRunWritesWeightReport(t *testing.T) {
	dir := t.TempDir()
	train, test := lineTestData()
	train_path := writeDatasetFile(t, dir, "train.tsv", train)
	test_path := writeDatasetFile(t, dir, "test.tsv", test)
	train_weights := writeWeightsFile(t, dir, "train-w.txt", []float64{1, 1, 4, 4})
	test_weights := writeWeightsFile(t, dir, "test-w.txt", []float64{2, 2})
	report_path := filepath.Join(dir, "report.json")

	params := classifierParams(map[string]string{
		ParamsWeightsPath:     train_weights,
		ParamsTestWeightsPath: test_weights,
		ParamsWeightReport:    report_path,
	})
	classifier := GetClassifier("lr")
	auc, report, _, err := AlgorithmRunWithReport(classifier, train_path, test_path, "", params)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if report == nil || report.Train == nil || report.Test == nil {
		t.Fatal("missing weight report population")
	}
	if report.TrainWeightsPath != train_weights || report.TestWeightsPath != test_weights {
		t.Errorf("report does not record weight sources: %+v", report)
	}
	if report.Train.Source != core.WeightSourceDataSet {
		t.Errorf("train weight source = %q", report.Train.Source)
	}
	if report.Train.Rows != 4 || report.Test.Rows != 2 {
		t.Errorf("unexpected populations: %+v %+v", report.Train, report.Test)
	}
	if math.Abs(report.Train.WeightSum-10) > 1e-9 {
		t.Errorf("train weight sum = %v, want 10", report.Train.WeightSum)
	}
	if math.Abs(report.Test.WeightSum-4) > 1e-9 {
		t.Errorf("test weight sum = %v, want 4", report.Test.WeightSum)
	}
	if !report.Metrics.HasWeights {
		t.Error("report metrics should be weighted")
	}
	if report.Definitions.SampleDuplication == "" || report.Definitions.Regularization == "" {
		t.Error("report must declare its conventions")
	}

	data, err := os.ReadFile(report_path)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var decoded WeightReport
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if decoded.TrainWeightsPath != train_weights {
		t.Error("persisted report lost the weight source")
	}

	if !(auc >= 0 && auc <= 1) {
		t.Errorf("auc out of range: %v", auc)
	}
	// Weighting the separated classes must not change a perfect ranking.
	if math.Abs(report.Metrics.WeightedAUC-1.0) > 1e-9 {
		t.Errorf("weighted AUC = %v, want 1", report.Metrics.WeightedAUC)
	}
}

// Zero-weight rows change the reported population and the weighted metrics but
// are not copied by training.
func TestRunnerWeightedMetricsExcludeZeroWeightRows(t *testing.T) {
	dir := t.TempDir()
	train, test := lineTestData()
	train_path := writeDatasetFile(t, dir, "train.tsv", train)
	test_path := writeDatasetFile(t, dir, "test.tsv", test)
	train_weights := writeWeightsFile(t, dir, "train-w.txt", []float64{1, 1, 1, 1})
	test_weights := writeWeightsFile(t, dir, "test-w.txt", []float64{1, 0})

	params := classifierParams(map[string]string{
		ParamsWeightsPath:     train_weights,
		ParamsTestWeightsPath: test_weights,
	})
	classifier := GetClassifier("lr")
	_, report, predictions, err := AlgorithmRunWithReport(classifier, train_path, test_path, "", params)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if report.Test.ZeroWeightRows != 1 {
		t.Errorf("zero-weight rows in test = %d, want 1", report.Test.ZeroWeightRows)
	}
	if report.Metrics.WeightedPopulation != 1 {
		t.Errorf("weighted population = %v, want 1", report.Metrics.WeightedPopulation)
	}
	// The excluded row is a negative; with only the positive left there are no
	// positive/negative pairs, so the weighted AUC is the degenerate 0.5.
	if math.Abs(report.Metrics.WeightedAUC-0.5) > 1e-9 {
		t.Errorf("weighted AUC = %v, want 0.5", report.Metrics.WeightedAUC)
	}
	if len(predictions) != 2 {
		t.Errorf("predictions = %d, want 2 (rows are kept, only excluded from metrics)", len(predictions))
	}
}

func TestAlgorithmTrainRejectsAllZeroWeights(t *testing.T) {
	dir := t.TempDir()
	train, _ := lineTestData()
	train_path := writeDatasetFile(t, dir, "train.tsv", train)
	params := classifierParams(map[string]string{
		ParamsWeightsPath: writeWeightsFile(t, dir, "w.txt", []float64{0, 0, 0, 0}),
	})
	if err := AlgorithmTrain(GetClassifier("lr"), train_path, params); err == nil {
		t.Fatal("expected all-zero weights to be rejected")
	}
}

// Unweighted runs must keep the legacy report-less behaviour: no metrics
// populated with weights and no duplicate rows introduced.
func TestAlgorithmRunOnDataSetUnweighted(t *testing.T) {
	train := core.NewDataSet()
	test := core.NewDataSet()
	for i := 0; i < 10; i++ {
		s := core.NewSample()
		s.Label = i % 2
		s.AddFeature(core.Feature{Id: 1, Value: float64(i%2)*2 - 1})
		train.AddSample(s)
		s2 := s.Clone()
		test.AddSample(s2)
	}
	params := classifierParams(nil)
	classifier := GetClassifier("lr")
	auc, predictions := AlgorithmRunOnDataSet(classifier, train, test, "", params)
	if len(predictions) != len(test.Samples) {
		t.Errorf("predictions %d != rows %d", len(predictions), len(test.Samples))
	}
	for _, p := range predictions {
		if p.Weight != 0 {
			t.Error("unweighted predictions must not carry weights")
		}
	}
	if !(auc >= 0 && auc <= 1) {
		t.Errorf("auc out of range: %v", auc)
	}
}
