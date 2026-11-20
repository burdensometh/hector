/*
Package stack implements out-of-fold (OOF) stacked generalization.

A stacking ensemble is trained in two layers:

 1. Every base classifier is trained K times, each time on K-1 folds of the
    training set and asked to predict only the fold it was NOT trained on.
    These out-of-fold predictions are collected per stable sample id together
    with the real label and the fold that produced the prediction.
 2. The OOF predictions, aligned by sample id (never by slice position) and
    ordered by a fixed base-model order, form the second-layer feature matrix
    on which a meta model is trained.

At application time every base model is re-fitted on the complete training set
and the same meta model combines their predictions. The whole inference chain
(base models + meta model + fixed model order) is persisted together and can
be reloaded by the command line tool and the HTTP server.
*/
package stack

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/xlvector/hector/algo"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/eval"
	"github.com/xlvector/hector/lr"
)

// schemaVersion is stored in the model manifest and checked on load.
const schemaVersion = 1

// OOFPrediction is a single out-of-fold prediction record. It carries the
// stable sample id, the id of the fold whose model produced it and the real
// label, so that records can be validated instead of being trusted by
// position.
type OOFPrediction struct {
	SampleID   int     `json:"sample_id"`
	Fold       int     `json:"fold"`
	Label      int     `json:"label"`
	Prediction float64 `json:"prediction"`
}

// StackingEnsemble is the complete, serializable inference chain.
type StackingEnsemble struct {
	// Methods is the fixed base-model order. Column j of every second-layer
	// feature vector is the prediction of Methods[j], for both training and
	// serving, forever.
	Methods []string `json:"methods"`
	// Folds is the number of OOF folds used during training.
	Folds int `json:"folds"`

	// Base models re-fitted on the FULL training set, in Methods order.
	// Empty while only the OOF layer has been built.
	BaseModels []algo.Classifier `json:"-"`
	// Meta is the second-layer model trained on the OOF feature matrix.
	Meta *lr.LinearRegression `json:"-"`

	params map[string]string
}

// ClassifierFactory builds a fresh, untrained classifier for a method name.
// It is injected by the caller (the hector root package provides one backed
// by GetClassifier) to keep this package free of import cycles.
type ClassifierFactory func(method string) algo.Classifier

/*
TrainOOF runs the out-of-fold stage.

For every base model and every fold k in [0, Folds), a fresh classifier is
trained on all samples whose id mod Folds != k and predicts only the samples
in fold k (the samples it has never seen). The returned records therefore
contain exactly one out-of-fold prediction per (method, sample id) pair.

The records are validated before they are returned: a missing, duplicated or
mis-folded prediction makes TrainOOF fail instead of silently misaligning the
second-layer training data.
*/
func TrainOOF(dataset *core.DataSet, methods []string, folds int,
	params map[string]string, factory ClassifierFactory) (map[string][]*OOFPrediction, error) {

	if folds < 2 {
		return nil, fmt.Errorf("stack: need at least 2 folds, got %d", folds)
	}
	if len(methods) == 0 {
		return nil, errors.New("stack: at least one base method is required")
	}
	if dataset == nil || len(dataset.Samples) == 0 {
		return nil, errors.New("stack: empty training dataset")
	}
	if factory == nil {
		return nil, errors.New("stack: nil classifier factory")
	}
	methods, dup := dedupMethods(methods)
	if dup != "" {
		return nil, fmt.Errorf("stack: duplicated base method %q, base-model order must be unique", dup)
	}

	n := len(dataset.Samples)
	allOOF := make(map[string][]*OOFPrediction, len(methods))

	for _, method := range methods {
		records := make([]*OOFPrediction, 0, n)
		for k := 0; k < folds; k++ {
			trainSet := core.NewDataSet()
			foldSet := core.NewDataSet()
			foldIDs := make([]int, 0, n/folds+1)
			for id, sample := range dataset.Samples {
				if id%folds == k {
					foldSet.AddSample(sample)
					foldIDs = append(foldIDs, id)
				} else {
					trainSet.AddSample(sample)
				}
			}
			if len(trainSet.Samples) == 0 {
				return nil, fmt.Errorf("stack: fold %d leaves no training data for method %q", k, method)
			}
			if len(foldSet.Samples) == 0 {
				return nil, fmt.Errorf("stack: fold %d is empty for method %q", k, method)
			}

			classifier := factory(method)
			if classifier == nil {
				return nil, fmt.Errorf("stack: factory returned nil classifier for method %q", method)
			}
			classifier.Init(params)
			classifier.Train(trainSet)

			for pos, sample := range foldSet.Samples {
				records = append(records, &OOFPrediction{
					SampleID:   foldIDs[pos],
					Fold:       k,
					Label:      sample.Label,
					Prediction: classifier.Predict(sample),
				})
			}
		}
		if err := ValidateOOF(records, n, folds, method); err != nil {
			return nil, err
		}
		allOOF[method] = records
	}
	return allOOF, nil
}

/*
ValidateOOF rejects an out-of-fold prediction set that cannot safely be turned
into second-layer training data:

  - every sample id in [0, n) must appear exactly once (missing and duplicate
    predictions are rejected),
  - the recorded fold must equal sample_id mod folds (a mis-labeled fold means
    the prediction may have been produced by a model that saw the sample),
  - the recorded label must equal the label of the original sample.

The check is deliberately strict: second-layer rows are joined by sample id,
and a wrong join would silently train the meta model on leaked or shuffled
features.
*/
func ValidateOOF(records []*OOFPrediction, n, folds int, method string) error {
	tag := "stack"
	if method != "" {
		tag = "stack (" + method + ")"
	}
	if folds < 2 {
		return fmt.Errorf("%s: invalid fold count %d", tag, folds)
	}
	if len(records) != n {
		return fmt.Errorf("%s: expected %d out-of-fold predictions, got %d (missing or duplicated folds)",
			tag, n, len(records))
	}
	seen := make(map[int]bool, n)
	for _, r := range records {
		if r == nil {
			return fmt.Errorf("%s: nil out-of-fold prediction record", tag)
		}
		if r.SampleID < 0 || r.SampleID >= n {
			return fmt.Errorf("%s: out-of-range sample id %d", tag, r.SampleID)
		}
		if seen[r.SampleID] {
			return fmt.Errorf("%s: duplicated out-of-fold prediction for sample id %d", tag, r.SampleID)
		}
		seen[r.SampleID] = true
		wantFold := r.SampleID % folds
		if r.Fold != wantFold {
			return fmt.Errorf("%s: mis-labeled fold for sample id %d: record says %d, want %d",
				tag, r.SampleID, r.Fold, wantFold)
		}
		if math.IsNaN(r.Prediction) || math.IsInf(r.Prediction, 0) {
			return fmt.Errorf("%s: non-finite prediction for sample id %d", tag, r.SampleID)
		}
	}
	for id := 0; id < n; id++ {
		if !seen[id] {
			return fmt.Errorf("%s: missing out-of-fold prediction for sample id %d", tag, id)
		}
	}
	return nil
}

/*
BuildMetaDataset joins the per-method OOF records into second-layer training
data.

Rows are aligned by stable sample id through a map - never by slice position -
and feature column j is fixed to Methods[j]. The label of every joined row is
checked against the original dataset label so a mis-labeled OOF record is
rejected rather than poisoning the meta model.
*/
func BuildMetaDataset(dataset *core.DataSet, methods []string, folds int,
	oof map[string][]*OOFPrediction) (*core.DataSet, error) {

	n := len(dataset.Samples)
	for _, method := range methods {
		records, ok := oof[method]
		if !ok {
			return nil, fmt.Errorf("stack: no out-of-fold predictions for method %q", method)
		}
		if err := ValidateOOF(records, n, folds, method); err != nil {
			return nil, err
		}
	}

	meta := core.NewDataSet()
	// index[method][sampleID] = prediction
	index := make(map[string]map[int]float64, len(methods))
	for _, method := range methods {
		col := make(map[int]float64, n)
		for _, r := range oof[method] {
			col[r.SampleID] = r.Prediction
			if r.Label != dataset.Samples[r.SampleID].Label {
				return nil, fmt.Errorf("stack (%s): mis-labeled out-of-fold record for sample id %d: record label %d, dataset label %d",
					method, r.SampleID, r.Label, dataset.Samples[r.SampleID].Label)
			}
		}
		index[method] = col
	}

	for id, sample := range dataset.Samples {
		row := core.NewSample()
		row.Label = sample.Label
		for j, method := range methods {
			row.AddFeature(core.Feature{
				Id:    int64(j),
				Value: index[method][id],
			})
		}
		meta.AddSample(row)
	}
	return meta, nil
}

/*
TrainStacking runs the complete stacking training entry point:

 1. out-of-fold predictions for every base model, strictly validated,
 2. meta model trained on sample-id-aligned, fixed-order OOF features,
 3. every base model re-fitted on the complete training set for serving.

The returned ensemble is ready to Predict and/or Save. No base-model
prediction is copied into the second layer: every meta feature is an
out-of-fold prediction aligned through its sample id.
*/
func TrainStacking(dataset *core.DataSet, methods []string, folds int,
	params map[string]string, factory ClassifierFactory) (*StackingEnsemble, error) {

	oof, err := TrainOOF(dataset, methods, folds, params, factory)
	if err != nil {
		return nil, err
	}
	metaSet, err := BuildMetaDataset(dataset, methods, folds, oof)
	if err != nil {
		return nil, err
	}

	ensemble := &StackingEnsemble{
		Methods:    append([]string(nil), methods...),
		Folds:      folds,
		BaseModels: make([]algo.Classifier, len(methods)),
		Meta:       &lr.LinearRegression{},
		params:     cloneParams(params),
	}
	ensemble.Meta.Init(params)
	ensemble.Meta.Train(metaSet)

	// Application stage: refit each base model on the FULL training set.
	for j, method := range methods {
		classifier := factory(method)
		if classifier == nil {
			return nil, fmt.Errorf("stack: factory returned nil classifier for method %q", method)
		}
		classifier.Init(params)
		classifier.Train(dataset)
		ensemble.BaseModels[j] = classifier
	}
	return ensemble, nil
}

// Predict runs the saved inference chain on one sample: every full-data base
// model predicts, the predictions are laid out in the fixed Methods order and
// the same meta model produces the final output.
func (e *StackingEnsemble) Predict(sample *core.Sample) (float64, error) {
	if err := e.ready(); err != nil {
		return 0.0, err
	}
	metaSample := core.NewSample()
	metaSample.Label = sample.Label
	for j, classifier := range e.BaseModels {
		metaSample.AddFeature(core.Feature{
			Id:    int64(j),
			Value: classifier.Predict(sample),
		})
	}
	return e.Meta.Predict(metaSample), nil
}

// PredictDataSet runs the inference chain over a dataset and returns the
// label/prediction pairs used by the evaluation package.
func (e *StackingEnsemble) PredictDataSet(dataset *core.DataSet) ([]*eval.LabelPrediction, error) {
	if err := e.ready(); err != nil {
		return nil, err
	}
	out := make([]*eval.LabelPrediction, 0, len(dataset.Samples))
	for _, sample := range dataset.Samples {
		p, err := e.Predict(sample)
		if err != nil {
			return nil, err
		}
		out = append(out, &eval.LabelPrediction{Label: sample.Label, Prediction: p})
	}
	return out, nil
}

func (e *StackingEnsemble) ready() error {
	if e == nil {
		return errors.New("stack: nil ensemble")
	}
	if len(e.Methods) == 0 || e.Meta == nil {
		return errors.New("stack: ensemble is not trained")
	}
	if len(e.BaseModels) != len(e.Methods) {
		return errors.New("stack: inference chain is incomplete, base models were not refitted")
	}
	for j, c := range e.BaseModels {
		if c == nil {
			return fmt.Errorf("stack: base model %d (%s) is missing", j, e.Methods[j])
		}
	}
	return nil
}

/*
Save persists the whole inference chain into dir:

	dir/manifest.json     methods, folds, schema version
	dir/meta.model        second-layer linear regression
	dir/base_00.model     full-data base model for Methods[0]
	dir/base_01.model     ...

The manifest is written last, so a directory missing its manifest is treated
as an incomplete/corrupt chain on load rather than partially loaded.
*/
func (e *StackingEnsemble) Save(dir string, factory ClassifierFactory) error {
	if err := e.ready(); err != nil {
		return err
	}
	if factory == nil {
		return errors.New("stack: nil classifier factory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for j, method := range e.Methods {
		if factory(method) == nil {
			return fmt.Errorf("stack: cannot persist unknown method %q", method)
		}
		path := baseModelPath(dir, j)
		e.BaseModels[j].SaveModel(path)
	}
	e.Meta.SaveModel(filepath.Join(dir, "meta.model"))

	manifest := map[string]interface{}{
		"schema_version": schemaVersion,
		"folds":          e.Folds,
		"methods":        e.Methods,
	}
	tmp := filepath.Join(dir, "manifest.json.tmp")
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	if err := enc.Encode(manifest); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, "manifest.json"))
}

// Load reconstructs the full inference chain from a directory saved by Save.
// The fixed model order and fold count are read back from the manifest and
// every base model is recreated through factory and loaded in that order.
func Load(dir string, params map[string]string, factory ClassifierFactory) (*StackingEnsemble, error) {
	if factory == nil {
		return nil, errors.New("stack: nil classifier factory")
	}
	manifestPath := filepath.Join(dir, "manifest.json")
	f, err := os.Open(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("stack: cannot open model manifest %s: %v", manifestPath, err)
	}
	defer f.Close()
	var manifest struct {
		SchemaVersion int      `json:"schema_version"`
		Folds         int      `json:"folds"`
		Methods       []string `json:"methods"`
	}
	if err := json.NewDecoder(f).Decode(&manifest); err != nil {
		return nil, fmt.Errorf("stack: corrupt model manifest: %v", err)
	}
	if manifest.SchemaVersion != schemaVersion {
		return nil, fmt.Errorf("stack: unsupported schema version %d, want %d",
			manifest.SchemaVersion, schemaVersion)
	}
	if manifest.Folds < 2 || len(manifest.Methods) == 0 {
		return nil, errors.New("stack: manifest does not describe a trained stacking ensemble")
	}
	methods, dup := dedupMethods(manifest.Methods)
	if dup != "" {
		return nil, fmt.Errorf("stack: manifest contains duplicated base method %q", dup)
	}

	base := make([]algo.Classifier, len(methods))
	for j, method := range methods {
		classifier := factory(method)
		if classifier == nil {
			return nil, fmt.Errorf("stack: unknown base method %q in manifest", method)
		}
		classifier.Init(params)
		modelPath := baseModelPath(dir, j)
		if _, err := os.Stat(modelPath); err != nil {
			return nil, fmt.Errorf("stack: missing base model file for %s: %v", method, err)
		}
		classifier.LoadModel(modelPath)
		base[j] = classifier
	}

	metaPath := filepath.Join(dir, "meta.model")
	if _, err := os.Stat(metaPath); err != nil {
		return nil, fmt.Errorf("stack: missing meta model: %v", err)
	}
	meta := &lr.LinearRegression{}
	meta.Init(params)
	meta.LoadModel(metaPath)
	if len(meta.Model) == 0 {
		return nil, errors.New("stack: loaded meta model is empty, refusing an incomplete inference chain")
	}

	return &StackingEnsemble{
		Methods:    methods,
		Folds:      manifest.Folds,
		BaseModels: base,
		Meta:       meta,
		params:     cloneParams(params),
	}, nil
}

func baseModelPath(dir string, j int) string {
	return filepath.Join(dir, "base_"+padIndex(j)+".model")
}

func padIndex(j int) string {
	s := strconv.Itoa(j)
	for len(s) < 2 {
		s = "0" + s
	}
	return s
}

func dedupMethods(methods []string) ([]string, string) {
	seen := make(map[string]bool, len(methods))
	out := make([]string, 0, len(methods))
	for _, m := range methods {
		if seen[m] {
			return nil, m
		}
		seen[m] = true
		out = append(out, m)
	}
	return out, ""
}

func cloneParams(params map[string]string) map[string]string {
	out := make(map[string]string, len(params))
	for k, v := range params {
		out[k] = v
	}
	return out
}
