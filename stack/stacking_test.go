package stack

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/xlvector/hector/algo"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/lr"
)

func testParams() map[string]string {
	return map[string]string{
		"steps":                  "10",
		"learning-rate":          "0.05",
		"regularization":         "0.0001",
		"alpha":                  "0.1",
		"beta":                   "1.0",
		"lambda1":                "0.1",
		"lambda2":                "1.0",
		"global":                 "-1",
		"learning-rate-discount": "1.0",
	}
}

func factory(method string) algo.Classifier {
	switch method {
	case "ftrl":
		return &lr.FTRLLogisticRegression{}
	case "lr":
		return &lr.LogisticRegression{}
	}
	return nil
}

func TestOOFOnePredictionPerSampleAndFold(t *testing.T) {
	dataset := core.LinearDataSet(200)
	methods := []string{"ftrl", "lr"}
	folds := 5

	oof, err := TrainOOF(dataset, methods, folds, testParams(), factory)
	if err != nil {
		t.Fatalf("TrainOOF failed: %v", err)
	}
	if len(oof) != len(methods) {
		t.Fatalf("got %d method columns, want %d", len(oof), len(methods))
	}
	for _, method := range methods {
		records := oof[method]
		if len(records) != len(dataset.Samples) {
			t.Fatalf("%s: got %d OOF records, want %d", method, len(records), len(dataset.Samples))
		}
		seen := map[int]bool{}
		for _, r := range records {
			if seen[r.SampleID] {
				t.Fatalf("%s: duplicate OOF prediction for sample %d", method, r.SampleID)
			}
			seen[r.SampleID] = true
			if r.Fold != r.SampleID%folds {
				t.Fatalf("%s: sample %d recorded under fold %d, want %d",
					method, r.SampleID, r.Fold, r.SampleID%folds)
			}
			if r.Label != dataset.Samples[r.SampleID].Label {
				t.Fatalf("%s: wrong label for sample %d", method, r.SampleID)
			}
			if math.IsNaN(r.Prediction) || math.IsInf(r.Prediction, 0) {
				t.Fatalf("%s: non-finite prediction for sample %d", method, r.SampleID)
			}
		}
		for id := range dataset.Samples {
			if !seen[id] {
				t.Fatalf("%s: missing OOF prediction for sample %d", method, id)
			}
		}
	}
}

func TestValidateOOFRejectsBadRecords(t *testing.T) {
	const n, folds = 10, 5
	good := func() []*OOFPrediction {
		records := make([]*OOFPrediction, 0, n)
		for id := 0; id < n; id++ {
			records = append(records, &OOFPrediction{
				SampleID: id, Fold: id % folds, Label: 0, Prediction: 0.5,
			})
		}
		return records
	}

	t.Run("missing", func(t *testing.T) {
		records := good()
		if err := ValidateOOF(records[:n-1], n, folds, "m"); err == nil {
			t.Fatal("expected rejection of a missing OOF prediction")
		}
	})
	t.Run("duplicate", func(t *testing.T) {
		records := good()
		records[1] = records[0]
		if err := ValidateOOF(records, n, folds, "m"); err == nil {
			t.Fatal("expected rejection of a duplicated OOF prediction")
		}
	})
	t.Run("mislabeled fold", func(t *testing.T) {
		records := good()
		records[0].Fold = (records[0].Fold + 1) % folds
		if err := ValidateOOF(records, n, folds, "m"); err == nil {
			t.Fatal("expected rejection of a mis-labeled fold")
		}
	})
	t.Run("out of range id", func(t *testing.T) {
		records := good()
		records[0].SampleID = n
		if err := ValidateOOF(records, n, folds, "m"); err == nil {
			t.Fatal("expected rejection of an out-of-range sample id")
		}
	})
	t.Run("non finite", func(t *testing.T) {
		records := good()
		records[0].Prediction = math.NaN()
		if err := ValidateOOF(records, n, folds, "m"); err == nil {
			t.Fatal("expected rejection of a NaN prediction")
		}
	})
	t.Run("valid", func(t *testing.T) {
		if err := ValidateOOF(good(), n, folds, "m"); err != nil {
			t.Fatalf("valid records rejected: %v", err)
		}
	})
}

func TestBuildMetaDatasetAlignsByIDAndRejectsBadLabel(t *testing.T) {
	dataset := core.LinearDataSet(50)
	folds := 5
	oof, err := TrainOOF(dataset, []string{"ftrl", "lr"}, folds, testParams(), factory)
	if err != nil {
		t.Fatalf("TrainOOF failed: %v", err)
	}

	meta, err := BuildMetaDataset(dataset, []string{"ftrl", "lr"}, folds, oof)
	if err != nil {
		t.Fatalf("BuildMetaDataset failed: %v", err)
	}
	if len(meta.Samples) != len(dataset.Samples) {
		t.Fatalf("meta dataset has %d rows, want %d", len(meta.Samples), len(dataset.Samples))
	}

	// Column j must hold the prediction of Methods[j], joined by sample id.
	for id, row := range meta.Samples {
		if row.Label != dataset.Samples[id].Label {
			t.Fatalf("row %d label mismatch", id)
		}
		if len(row.Features) != 2 {
			t.Fatalf("row %d has %d meta features, want 2", id, len(row.Features))
		}
		if row.Features[0].Id != 0 || row.Features[1].Id != 1 {
			t.Fatalf("row %d meta feature ids not in fixed order: %v", id, row.Features)
		}
		want0 := 0.0
		for _, r := range oof["ftrl"] {
			if r.SampleID == id {
				want0 = r.Prediction
			}
		}
		if row.Features[0].Value != want0 {
			t.Fatalf("row %d column 0 not aligned to ftrl OOF prediction by id", id)
		}
	}

	// Reversing the method order must swap the columns, proving column order
	// is bound to the fixed model order rather than slice position.
	revMeta, err := BuildMetaDataset(dataset, []string{"lr", "ftrl"}, folds, oof)
	if err != nil {
		t.Fatalf("BuildMetaDataset reversed failed: %v", err)
	}
	for id := range dataset.Samples {
		a := meta.Samples[id].Features
		b := revMeta.Samples[id].Features
		if a[0].Value != b[1].Value || a[1].Value != b[0].Value {
			t.Fatalf("row %d columns were not swapped when model order reversed", id)
		}
	}

	// A mis-labeled OOF record must be rejected.
	tampered := map[string][]*OOFPrediction{}
	for m, records := range oof {
		cp := make([]*OOFPrediction, len(records))
		for i, r := range records {
			cp[i] = &OOFPrediction{
				SampleID: r.SampleID, Fold: r.Fold, Label: r.Label, Prediction: r.Prediction,
			}
		}
		tampered[m] = cp
	}
	tampered["ftrl"][0].Label = 1 - tampered["ftrl"][0].Label
	if _, err := BuildMetaDataset(dataset, []string{"ftrl", "lr"}, folds, tampered); err == nil {
		t.Fatal("expected rejection of a mis-labeled OOF prediction")
	}
}

func TestTrainStackingPredictAndReload(t *testing.T) {
	dataset := core.LinearDataSet(300)
	methods := []string{"ftrl", "lr"}
	ensemble, err := TrainStacking(dataset, methods, 5, testParams(), factory)
	if err != nil {
		t.Fatalf("TrainStacking failed: %v", err)
	}
	if len(ensemble.BaseModels) != len(methods) || ensemble.Meta == nil {
		t.Fatal("inference chain is incomplete")
	}

	predictions, err := ensemble.PredictDataSet(dataset)
	if err != nil {
		t.Fatalf("PredictDataSet failed: %v", err)
	}
	if len(predictions) != len(dataset.Samples) {
		t.Fatalf("got %d predictions, want %d", len(predictions), len(dataset.Samples))
	}
	for _, p := range predictions {
		if math.IsNaN(p.Prediction) || math.IsInf(p.Prediction, 0) {
			t.Fatal("non-finite stacking prediction")
		}
	}

	dir := filepath.Join(t.TempDir(), "chain")
	if err := ensemble.Save(dir, factory); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 2+len(methods) { // manifest + meta + one file per base model
		t.Fatalf("saved chain has %d files, want %d", len(entries), 2+len(methods))
	}

	loaded, err := Load(dir, testParams(), factory)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(loaded.Methods) != len(methods) || loaded.Methods[0] != methods[0] || loaded.Methods[1] != methods[1] {
		t.Fatalf("fixed model order not preserved: %v", loaded.Methods)
	}
	reloaded, err := loaded.PredictDataSet(dataset)
	if err != nil {
		t.Fatalf("reloaded PredictDataSet failed: %v", err)
	}
	for i, p := range predictions {
		if p.Prediction != reloaded[i].Prediction {
			t.Fatalf("prediction %d changed after reload: %f -> %f",
				i, p.Prediction, reloaded[i].Prediction)
		}
	}
}

func TestLoadRejectsCorruptChain(t *testing.T) {
	dataset := core.LinearDataSet(100)
	ensemble, err := TrainStacking(dataset, []string{"ftrl", "lr"}, 4, testParams(), factory)
	if err != nil {
		t.Fatalf("TrainStacking failed: %v", err)
	}
	dir := filepath.Join(t.TempDir(), "chain")
	if err := ensemble.Save(dir, factory); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	t.Run("missing manifest", func(t *testing.T) {
		d := filepath.Join(t.TempDir(), "c")
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(d, testParams(), factory); err == nil {
			t.Fatal("expected rejection of a chain without manifest")
		}
	})
	t.Run("missing base model", func(t *testing.T) {
		if err := os.Remove(baseModelPath(dir, 0)); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(dir, testParams(), factory); err == nil {
			t.Fatal("expected rejection when a base model file is missing")
		}
	})
}

func TestRejectsBadArguments(t *testing.T) {
	dataset := core.LinearDataSet(50)
	if _, err := TrainOOF(dataset, []string{"ftrl"}, 1, testParams(), factory); err == nil {
		t.Fatal("expected rejection of fewer than 2 folds")
	}
	if _, err := TrainOOF(dataset, nil, 5, testParams(), factory); err == nil {
		t.Fatal("expected rejection of empty method list")
	}
	if _, err := TrainOOF(dataset, []string{"ftrl", "ftrl"}, 5, testParams(), factory); err == nil {
		t.Fatal("expected rejection of duplicated base methods")
	}
	if _, err := TrainOOF(dataset, []string{"ftrl"}, 5, testParams(), nil); err == nil {
		t.Fatal("expected rejection of nil factory")
	}
	if _, err := TrainOOF(dataset, []string{"unknown"}, 5, testParams(), factory); err == nil {
		t.Fatal("expected rejection when factory cannot build a method")
	}
}
