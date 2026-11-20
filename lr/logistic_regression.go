package lr

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
)

type LogisticRegressionParams struct {
	LearningRate   float64
	Regularization float64
	Steps          int
}

type LogisticRegression struct {
	Model  map[int64]float64
	Params LogisticRegressionParams
}

func (algo *LogisticRegression) SaveModel(path string) {
	sb := util.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *LogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()
	algo.Model = make(map[int64]float64)
	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		fw, _ := strconv.ParseFloat(tks[1], 64)
		algo.Model[fid] = fw
	}
}

func (algo *LogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]float64)

	algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
	algo.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
	steps, _ := strconv.ParseInt(params["steps"], 10, 32)
	algo.Params.Steps = int(steps)
}

func (algo *LogisticRegression) Train(dataset *core.DataSet) {
	if err := algo.TrainWithError(dataset); err != nil {
		panic(err)
	}
}

// TrainWithError trains the model and reports invalid sample weights instead of
// learning from them.
//
// Weight convention: the data-loss gradient of row i is scaled by its
// representative weight w_i, so the model minimizes
//
//	sum_i w_i * logloss(x_i, y_i) + (regularization / 2) * ||w||^2
//
// over the weighted population. The L2 shrinkage stays in the per-update form
// of the legacy trainer and is *not* scaled by the sample weight. A row with
// weight zero is skipped entirely: it contributes neither data loss nor
// shrinkage and is never copied into extra rows. When no weights are supplied
// the legacy loop below runs unchanged.
func (algo *LogisticRegression) TrainWithError(dataset *core.DataSet) error {
	weights, weight_source, err := dataset.ResolveWeights()
	if err != nil {
		return err
	}

	algo.Model = make(map[int64]float64)
	for step := 0; step < algo.Params.Steps; step++ {
		for i, sample := range dataset.Samples {
			if weight_source != core.WeightSourceNone && weights[i] == 0 {
				// Zero-weight rows do not participate in learning.
				continue
			}
			prediction := algo.Predict(sample)
			err := sample.LabelDoubleValue() - prediction
			scale := 1.0
			if weight_source != core.WeightSourceNone {
				scale = weights[i]
			}
			for _, feature := range sample.Features {
				model_feature_value, ok := algo.Model[feature.Id]
				if !ok {
					model_feature_value = 0.0
				}
				model_feature_value += algo.Params.LearningRate * (scale*err*feature.Value - algo.Params.Regularization*model_feature_value)
				algo.Model[feature.Id] = model_feature_value
			}
		}
		algo.Params.LearningRate *= 0.9
	}
	return nil
}

func (algo *LogisticRegression) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value
		}
	}
	return util.Sigmoid(ret)
}
