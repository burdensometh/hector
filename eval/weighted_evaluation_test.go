package eval

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func approx(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestWeightedAUCUnitWeightsMatchesAUC(t *testing.T) {
	predictions := []*LabelPrediction{
		{Label: 1, Prediction: 0.9, Weight: 1},
		{Label: 0, Prediction: 0.8, Weight: 1},
		{Label: 1, Prediction: 0.7, Weight: 1},
		{Label: 0, Prediction: 0.4, Weight: 1},
		{Label: 1, Prediction: 0.2, Weight: 1},
		{Label: 0, Prediction: 0.1, Weight: 1},
	}
	if got, want := WeightedAUC(predictions), AUC(predictions); !approx(got, want) {
		t.Errorf("unit-weight weighted AUC %v != AUC %v", got, want)
	}
}

func TestWeightedAUCCountsPairsByWeight(t *testing.T) {
	// One positive at 0.9, two negatives: one at 0.5 (weight 3) and one at 0.1
	// (weight 1). Positives beat both, so AUC is 1 regardless of weights.
	perfect := []*LabelPrediction{
		{Label: 1, Prediction: 0.9, Weight: 1},
		{Label: 0, Prediction: 0.5, Weight: 3},
		{Label: 0, Prediction: 0.1, Weight: 1},
	}
	if got := WeightedAUC(perfect); !approx(got, 1.0) {
		t.Errorf("perfect ranking weighted AUC = %v, want 1", got)
	}

	// Now the heavy negative outscores the positive; only the light negative is
	// beaten. AUC = 1 / (3+1) = 0.25.
	mixed := []*LabelPrediction{
		{Label: 0, Prediction: 0.9, Weight: 3},
		{Label: 1, Prediction: 0.5, Weight: 1},
		{Label: 0, Prediction: 0.1, Weight: 1},
	}
	if got := WeightedAUC(mixed); !approx(got, 0.25) {
		t.Errorf("weighted AUC = %v, want 0.25", got)
	}
}

func TestWeightedAUCTiesAndZeroWeight(t *testing.T) {
	// Ties count as one half: positive ties one negative, beats the other.
	predictions := []*LabelPrediction{
		{Label: 1, Prediction: 0.5, Weight: 1},
		{Label: 0, Prediction: 0.5, Weight: 1},
		{Label: 0, Prediction: 0.1, Weight: 1},
	}
	if got := WeightedAUC(predictions); !approx(got, 0.75) {
		t.Errorf("weighted AUC with a tie = %v, want 0.75", got)
	}

	// A zero-weight row must not enter the population even if it would change
	// the ranking.
	withZero := append([]*LabelPrediction{
		{Label: 0, Prediction: 0.99, Weight: 0},
	}, predictions...)
	if got := WeightedAUC(withZero); !approx(got, 0.75) {
		t.Errorf("zero-weight row leaked into weighted AUC: %v", got)
	}
}

func TestWeightedAUCDegenerate(t *testing.T) {
	noNegatives := []*LabelPrediction{{Label: 1, Prediction: 0.9, Weight: 1}}
	if got := WeightedAUC(noNegatives); !approx(got, 0.5) {
		t.Errorf("weighted AUC without negatives = %v, want 0.5", got)
	}
	allZero := []*LabelPrediction{{Label: 1, Prediction: 0.9, Weight: 0}}
	if got := WeightedAUC(allZero); !approx(got, 0.5) {
		t.Errorf("weighted AUC with no population = %v, want 0.5", got)
	}
}

func TestLogLoss(t *testing.T) {
	predictions := []*LabelPrediction{
		{Label: 1, Prediction: 1.0},
		{Label: 0, Prediction: 0.0},
	}
	if got := LogLoss(predictions); got > 1e-9 {
		t.Errorf("confident correct predictions should have ~0 logloss, got %v", got)
	}

	predictions = []*LabelPrediction{
		{Label: 1, Prediction: 0.5},
		{Label: 0, Prediction: 0.5},
	}
	if got := LogLoss(predictions); !approx(got, math.Log(2)) {
		t.Errorf("0.5 predictions logloss = %v, want ln2", got)
	}
}

func TestWeightedLogLoss(t *testing.T) {
	predictions := []*LabelPrediction{
		{Label: 1, Prediction: 0.5, Weight: 3},
		{Label: 0, Prediction: 0.5, Weight: 1},
		{Label: 0, Prediction: 0.0, Weight: 0}, // excluded
	}
	if got := WeightedLogLoss(predictions); !approx(got, math.Log(2)) {
		t.Errorf("weighted logloss = %v, want ln2", got)
	}

	// A row with weight 0 must not affect the weighted mean regardless of its
	// (possibly terrible) prediction.
	shifted := []*LabelPrediction{
		{Label: 1, Prediction: 0.5, Weight: 3},
		{Label: 0, Prediction: 0.5, Weight: 1},
		{Label: 1, Prediction: 0.0001, Weight: 0},
	}
	if got := WeightedLogLoss(shifted); !approx(got, math.Log(2)) {
		t.Errorf("zero-weight row changed weighted logloss: %v", got)
	}
}

func TestWeightedLogLossNoPopulation(t *testing.T) {
	predictions := []*LabelPrediction{{Label: 1, Prediction: 0.5, Weight: 0}}
	if got := WeightedLogLoss(predictions); !math.IsNaN(got) {
		t.Errorf("empty weighted population should give NaN, got %v", got)
	}
}

func TestClassificationMetricsForUnweighted(t *testing.T) {
	predictions := []*LabelPrediction{
		{Label: 1, Prediction: 0.9},
		{Label: 0, Prediction: 0.2},
	}
	metrics := ClassificationMetricsFor(predictions, false)
	if metrics.HasWeights {
		t.Error("HasWeights should be false")
	}
	if !approx(metrics.AUC, metrics.WeightedAUC) {
		t.Errorf("unweighted: weighted AUC %v != AUC %v", metrics.WeightedAUC, metrics.AUC)
	}
	if !approx(metrics.LogLoss, metrics.WeightedLogLoss) {
		t.Errorf("unweighted: weighted logloss %v != logloss %v", metrics.WeightedLogLoss, metrics.LogLoss)
	}
	if !approx(metrics.WeightedPopulation, 2) {
		t.Errorf("weighted population = %v, want 2", metrics.WeightedPopulation)
	}
}

func TestClassificationMetricsMarshalHandlesUndefined(t *testing.T) {
	// A weighted population containing only zero-weight rows leaves the
	// weighted log loss undefined; the JSON encoding must not fail.
	metrics := ClassificationMetricsFor([]*LabelPrediction{
		{Label: 1, Prediction: 0.5, Weight: 0},
	}, true)
	if !math.IsNaN(metrics.WeightedLogLoss) {
		t.Fatalf("expected NaN weighted logloss, got %v", metrics.WeightedLogLoss)
	}
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	if !strings.Contains(string(data), `"weighted_logloss":null`) {
		t.Errorf("undefined metric should be null: %s", data)
	}
}
