/*
Package hector is a golang based machine learning lib. It intend to implement all famous machine learning algoirhtms by golang.
Currently, it only support algorithms which can solve binary classification problems. Supported algorithms include:
1. Decision Tree (CART, Random Forest, GBDT)
2. Logistic Regression
3. SVM
4. Neural Network
*/
package hector

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/xlvector/hector/algo"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/eval"
)

// Params used to wire representative sample weights through the runners.
const (
	// ParamsWeightsPath is the weights file for the training rows, one weight
	// per row in dataset order.
	ParamsWeightsPath = "weights"
	// ParamsTestWeightsPath is the weights file for the evaluation rows.
	ParamsTestWeightsPath = "test-weights"
	// ParamsWeightReport is the path of the JSON weight provenance record that
	// the runner writes. The record names the weight source, the per-row
	// population and the metric definitions used for training and evaluation.
	ParamsWeightReport = "weight-report"
)

// AttachWeights loads a weights file, validates it against the dataset, and
// attaches it without touching the samples. An empty path is a no-op.
func AttachWeights(dataset *core.DataSet, path string) error {
	if path == "" {
		return nil
	}
	weights, err := core.LoadWeights(path)
	if err != nil {
		return err
	}
	if err := dataset.SetWeights(weights); err != nil {
		return fmt.Errorf("weights file %s: %v", path, err)
	}
	return nil
}

// AttachWeightsFromParams attaches weights from params[key] to the dataset.
func AttachWeightsFromParams(dataset *core.DataSet, params map[string]string, key string) error {
	return AttachWeights(dataset, params[key])
}

// ValidateTrainingWeights fails fast on weights that would corrupt training:
// a length mismatch, non-finite or negative weights, or an all-zero vector.
func ValidateTrainingWeights(dataset *core.DataSet) error {
	return dataset.ValidateWeights()
}

// CheckSamePopulation enforces that training and evaluation are weighted
// consistently: both unweighted, or both weighted. Weighting only one side
// would train on one population and report metrics over another.
func CheckSamePopulation(train_dataset, test_dataset *core.DataSet) error {
	if train_dataset == nil || test_dataset == nil {
		return nil
	}
	_, train_source, train_err := train_dataset.ResolveWeights()
	if train_err != nil {
		return train_err
	}
	_, test_source, test_err := test_dataset.ResolveWeights()
	if test_err != nil {
		return test_err
	}
	train_weighted := train_source != core.WeightSourceNone
	test_weighted := test_source != core.WeightSourceNone
	switch {
	case train_weighted && !test_weighted:
		return fmt.Errorf("sample weights: training is weighted (%s) but the test set is not; supply %q so training and evaluation use the same population", train_source, ParamsTestWeightsPath)
	case !train_weighted && test_weighted:
		return fmt.Errorf("sample weights: the test set is weighted (%s) but training is not; supply %q so training and evaluation use the same population", test_source, ParamsWeightsPath)
	}
	return nil
}

func AlgorithmRun(classifier algo.Classifier,
	train_path string, test_path string, pred_path string,
	params map[string]string) (float64, []*eval.LabelPrediction, error) {
	auc, _, predictions, err := AlgorithmRunWithReport(classifier, train_path, test_path, pred_path, params)
	return auc, predictions, err
}

// AlgorithmRunWithReport is AlgorithmRun plus a WeightReport describing the
// populations used for training and evaluation.
func AlgorithmRunWithReport(classifier algo.Classifier,
	train_path string, test_path string, pred_path string,
	params map[string]string) (float64, *WeightReport, []*eval.LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return 0.5, nil, nil, err
	}
	if err := AttachWeightsFromParams(train_dataset, params, ParamsWeightsPath); err != nil {
		return 0.5, nil, nil, err
	}

	test_dataset := core.NewDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil {
		return 0.5, nil, nil, err
	}
	if err := AttachWeights(test_dataset, params[ParamsTestWeightsPath]); err != nil {
		return 0.5, nil, nil, err
	}

	// Both populations are validated before the classifier is initialized or
	// trained, so an all-zero vector or a length mismatch cannot be partially
	// learned.
	if err := train_dataset.ValidateWeights(); err != nil {
		return 0.5, nil, nil, err
	}
	if err := test_dataset.ValidateWeights(); err != nil {
		return 0.5, nil, nil, err
	}
	if err := CheckSamePopulation(train_dataset, test_dataset); err != nil {
		return 0.5, nil, nil, err
	}

	classifier.Init(params)
	auc, report, predictions := AlgorithmRunOnDataSetReport(classifier, train_dataset,
		test_dataset, pred_path, params)
	report.TrainWeightsPath = params[ParamsWeightsPath]
	report.TestWeightsPath = params[ParamsTestWeightsPath]
	if err := SaveWeightReportObject(report, params); err != nil {
		return 0.5, report, nil, err
	}
	return auc, report, predictions, nil
}

func AlgorithmTrain(classifier algo.Classifier, train_path string,
	params map[string]string) error {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return err
	}
	if err := AttachWeightsFromParams(train_dataset, params, ParamsWeightsPath); err != nil {
		return err
	}
	if err := ValidateTrainingWeights(train_dataset); err != nil {
		return err
	}

	classifier.Init(params)
	classifier.Train(train_dataset)

	model_path, _ := params["model"]

	if model_path != "" {
		classifier.SaveModel(model_path)
	}
	if err := SaveWeightReport(train_dataset, nil, nil, params); err != nil {
		return err
	}
	return nil
}

func AlgorithmTest(classifier algo.Classifier, test_path string, pred_path string, params map[string]string) (float64, []*eval.LabelPrediction, error) {
	auc, _, predictions, err := AlgorithmTestWithReport(classifier, test_path, pred_path, params)
	return auc, predictions, err
}

// AlgorithmTestWithReport is AlgorithmTest plus a WeightReport describing the
// evaluation population.
func AlgorithmTestWithReport(classifier algo.Classifier, test_path string, pred_path string, params map[string]string) (float64, *WeightReport, []*eval.LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)

	model_path, _ := params["model"]
	classifier.Init(params)
	if model_path != "" {
		classifier.LoadModel(model_path)
	} else {
		return 0.0, nil, nil, nil
	}

	test_dataset := core.NewDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil {
		return 0.0, nil, nil, err
	}

	if err := AttachWeights(test_dataset, params[ParamsTestWeightsPath]); err != nil {
		return 0.0, nil, nil, err
	}
	if err := test_dataset.ValidateWeights(); err != nil {
		return 0.0, nil, nil, err
	}

	auc, report, predictions := AlgorithmRunOnDataSetReport(classifier, nil, test_dataset, pred_path, params)
	report.TestWeightsPath = params[ParamsTestWeightsPath]
	if err := SaveWeightReportObject(report, params); err != nil {
		return 0.0, report, nil, err
	}
	return auc, report, predictions, nil
}

func AlgorithmRunOnDataSet(classifier algo.Classifier, train_dataset, test_dataset *core.DataSet, pred_path string, params map[string]string) (float64, []*eval.LabelPrediction) {
	auc, _, predictions := AlgorithmRunOnDataSetReport(classifier, train_dataset, test_dataset, pred_path, params)
	return auc, predictions
}

// AlgorithmRunOnDataSetReport trains (when train_dataset is not nil) and
// evaluates, returning the unweighted AUC plus the report used by the runners.
// Invalid weights, or a train/test pair weighted inconsistently, are rejected
// *before* training starts; in that case the report carries the error and no
// prediction is produced.
func AlgorithmRunOnDataSetReport(classifier algo.Classifier, train_dataset, test_dataset *core.DataSet, pred_path string, params map[string]string) (float64, *WeightReport, []*eval.LabelPrediction) {
	report := &WeightReport{Definitions: WeightDefinition()}
	if train_dataset != nil {
		if err := train_dataset.ValidateWeights(); err != nil {
			report.Error = err.Error()
			return 0.5, report, nil
		}
	}
	weights, weight_source, err := test_dataset.ResolveWeights()
	if err != nil {
		// Evaluation must not silently fall back to an unweighted population.
		report.Error = err.Error()
		return 0.5, report, nil
	}
	if err := CheckSamePopulation(train_dataset, test_dataset); err != nil {
		report.Error = err.Error()
		return 0.5, report, nil
	}
	if train_dataset != nil {
		classifier.Train(train_dataset)
	}

	predictions := []*eval.LabelPrediction{}
	var pred_file *os.File
	if pred_path != "" {
		pred_file, _ = os.Create(pred_path)
	}
	for i, sample := range test_dataset.Samples {
		prediction := classifier.Predict(sample)
		if pred_file != nil {
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		weight := 0.0
		if weight_source != core.WeightSourceNone {
			weight = weights[i]
		}
		predictions = append(predictions, &(eval.LabelPrediction{Label: sample.Label, Prediction: prediction, Weight: weight}))
	}
	if pred_path != "" {
		defer pred_file.Close()
	}

	auc := eval.AUC(predictions)
	report.Train = summarizeDataset(train_dataset)
	report.Test = summarizeDataset(test_dataset)
	report.Metrics = eval.ClassificationMetricsFor(predictions, weight_source != core.WeightSourceNone)
	return auc, report, predictions
}

func summarizeDataset(dataset *core.DataSet) *core.WeightSummary {
	if dataset == nil {
		return nil
	}
	summary, err := dataset.SummarizeWeights()
	if err != nil {
		return nil
	}
	return &summary
}

// WeightReport is the provenance record the runner persists. It states where
// the representative weights came from, which rows formed the training and
// evaluation populations, and how the metrics were defined.
type WeightReport struct {
	TrainWeightsPath string                     `json:"train_weights_path,omitempty"`
	TestWeightsPath  string                     `json:"test_weights_path,omitempty"`
	Train            *core.WeightSummary        `json:"train,omitempty"`
	Test             *core.WeightSummary        `json:"test,omitempty"`
	Metrics          eval.ClassificationMetrics `json:"metrics"`
	Definitions      WeightDefinitions          `json:"definitions"`
	Error            string                     `json:"error,omitempty"`
}

// WeightDefinitions documents the conventions in force for this run.
type WeightDefinitions struct {
	Population        string `json:"population"`
	DataLossGradient  string `json:"data_loss_gradient"`
	Regularization    string `json:"regularization"`
	ZeroWeightRows    string `json:"zero_weight_rows"`
	SampleDuplication string `json:"sample_duplication"`
	WeightedAUC       string `json:"weighted_auc"`
	WeightedLogLoss   string `json:"weighted_logloss"`
}

// WeightDefinition returns the fixed, declared conventions for weighted
// learning and evaluation in this library.
func WeightDefinition() WeightDefinitions {
	return WeightDefinitions{
		Population:        "a row's representative weight counts it once in the weighted population; it is never copied into extra rows",
		DataLossGradient:  "per-row data-loss gradient is multiplied by the row's weight",
		Regularization:    "the L2 (lr) and FTRL proximal (lambda1/lambda2) terms are model priors and are not scaled by sample weight; zero-weight rows receive no update at all",
		ZeroWeightRows:    "excluded from learning and from weighted metrics",
		SampleDuplication: "weights never duplicate or resample rows",
		WeightedAUC:       "sum over positive/negative pairs of w_pos*w_neg, ties counted 1/2, divided by the total weighted positive-negative pair mass",
		WeightedLogLoss:   "sum(w*logloss)/sum(w) over rows with positive weight",
	}
}

// SaveWeightReport writes the JSON provenance record to params[ParamsWeightReport]
// when one was requested. Without a report path it is a no-op.
func SaveWeightReport(train_dataset, test_dataset *core.DataSet, metrics *eval.ClassificationMetrics, params map[string]string) error {
	path := params[ParamsWeightReport]
	if path == "" {
		return nil
	}
	report := &WeightReport{
		TrainWeightsPath: params[ParamsWeightsPath],
		TestWeightsPath:  params[ParamsTestWeightsPath],
		Train:            summarizeDataset(train_dataset),
		Test:             summarizeDataset(test_dataset),
		Definitions:      WeightDefinition(),
	}
	if metrics != nil {
		report.Metrics = *metrics
	}
	return WriteWeightReport(path, report)
}

// SaveWeightReportObject writes an already-built report to
// params[ParamsWeightReport] when one was requested.
func SaveWeightReportObject(report *WeightReport, params map[string]string) error {
	path := params[ParamsWeightReport]
	if path == "" || report == nil {
		return nil
	}
	return WriteWeightReport(path, report)
}

// WriteWeightReport marshals the provenance record to path.
func WriteWeightReport(path string, report *WeightReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0644)
}

/* Regression */
func RegAlgorithmRun(regressor algo.Regressor, train_path string, test_path string, pred_path string, params map[string]string) (float64, []*eval.RealPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewRealDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return 0.5, nil, err
	}

	test_dataset := core.NewRealDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil {
		return 0.5, nil, err
	}
	regressor.Init(params)
	rmse, predictions := RegAlgorithmRunOnDataSet(regressor, train_dataset, test_dataset, pred_path, params)

	return rmse, predictions, nil
}

func RegAlgorithmTrain(regressor algo.Regressor, train_path string, params map[string]string) error {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewRealDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return err
	}

	regressor.Init(params)
	regressor.Train(train_dataset)

	model_path, _ := params["model"]

	if model_path != "" {
		regressor.SaveModel(model_path)
	}

	return nil
}

func RegAlgorithmTest(regressor algo.Regressor, test_path string, pred_path string, params map[string]string) (float64, []*eval.RealPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)

	model_path, _ := params["model"]
	regressor.Init(params)
	if model_path != "" {
		regressor.LoadModel(model_path)
	} else {
		return 0.0, nil, nil
	}

	test_dataset := core.NewRealDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil {
		return 0.0, nil, err
	}

	rmse, predictions := RegAlgorithmRunOnDataSet(regressor, nil, test_dataset, pred_path, params)

	return rmse, predictions, nil
}

func RegAlgorithmRunOnDataSet(regressor algo.Regressor, train_dataset, test_dataset *core.RealDataSet, pred_path string, params map[string]string) (float64, []*eval.RealPrediction) {

	if train_dataset != nil {
		regressor.Train(train_dataset)
	}

	predictions := []*eval.RealPrediction{}
	var pred_file *os.File
	if pred_path != "" {
		pred_file, _ = os.Create(pred_path)
	}
	for _, sample := range test_dataset.Samples {
		prediction := regressor.Predict(sample)
		if pred_file != nil {
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		predictions = append(predictions, &eval.RealPrediction{Value: sample.Value, Prediction: prediction})
	}
	if pred_path != "" {
		defer pred_file.Close()
	}

	rmse := eval.RegRMSE(predictions)
	return rmse, predictions
}
