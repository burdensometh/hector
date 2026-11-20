package hector

import (
	"errors"
	"strconv"
	"strings"

	"github.com/xlvector/hector/algo"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/eval"
	"github.com/xlvector/hector/stack"
)

// ClassifierFactory adapts GetClassifier to stack.ClassifierFactory. A method
// name that GetClassifier does not recognize must fail loudly instead of
// silently falling back to a default model, so unknown methods return nil.
func ClassifierFactory(method string) algo.Classifier {
	if _, ok := knownMethods[method]; !ok {
		return nil
	}
	return GetClassifier(method)
}

// knownMethods is the fixed set of supported base-model names. It must match
// the branches implemented by GetClassifier.
var knownMethods = map[string]bool{
	"lr": true, "ftrl": true, "ep": true, "rdt": true, "cart": true,
	"cart-regression": true, "rf": true, "fm": true, "sa": true,
	"gbdt": true, "svm": true, "linear_svm": true, "l1vm": true,
	"knn": true, "ann": true, "lr_owlqn": true,
}

// ParseStackMethods parses the comma separated base-method list and keeps the
// user-supplied order, which becomes the fixed second-layer column order.
func ParseStackMethods(spec string) ([]string, error) {
	methods := []string{}
	seen := map[string]bool{}
	for _, raw := range strings.Split(spec, ",") {
		m := strings.TrimSpace(raw)
		if m == "" {
			continue
		}
		if !knownMethods[m] {
			return nil, errors.New("unknown stacking base method: " + m)
		}
		if seen[m] {
			return nil, errors.New("duplicated stacking base method: " + m)
		}
		seen[m] = true
		methods = append(methods, m)
	}
	if len(methods) == 0 {
		return nil, errors.New("no stacking base methods given")
	}
	return methods, nil
}

// StackTrain is the callable out-of-fold stacking training entry.
//
// It loads the training set once, runs K-fold OOF training for every base
// model (each base model only predicts folds it was not trained on), joins the
// OOF predictions by stable sample id into fixed-order second-layer features,
// trains the meta model, refits every base model on the full training set and
// saves the complete inference chain into modelDir.
func StackTrain(trainPath, modelDir string, methods []string, folds int,
	params map[string]string) (*stack.StackingEnsemble, error) {

	if modelDir == "" {
		return nil, errors.New("stacking model directory is required")
	}
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	dataset := core.NewDataSet()
	if err := dataset.Load(trainPath, global); err != nil {
		return nil, err
	}
	ensemble, err := stack.TrainStacking(dataset, methods, folds, params, ClassifierFactory)
	if err != nil {
		return nil, err
	}
	if err := ensemble.Save(modelDir, ClassifierFactory); err != nil {
		return nil, err
	}
	return ensemble, nil
}

// StackTest loads a previously saved inference chain and actually runs it on
// the independent test set. It returns AUC together with every prediction.
func StackTest(testPath, modelDir string, params map[string]string) (float64, []*eval.LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	ensemble, err := stack.Load(modelDir, params, ClassifierFactory)
	if err != nil {
		return 0.5, nil, err
	}
	testSet := core.NewDataSet()
	if err := testSet.Load(testPath, global); err != nil {
		return 0.5, nil, err
	}
	predictions, err := ensemble.PredictDataSet(testSet)
	if err != nil {
		return 0.5, nil, err
	}
	return eval.AUC(predictions), predictions, nil
}
