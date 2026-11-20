package hector

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// writeLibSVM writes a small linearly separable dataset in the libsvm-like
// format core.DataSet.Load understands.
func writeLibSVM(t *testing.T, path string, n int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	// Deterministic pseudo-pattern without relying on global rand state.
	for i := 0; i < n; i++ {
		label := i % 2
		x := 0.2 + float64(i%7)*0.1
		if label == 0 {
			x = -x
		}
		if _, err := f.WriteString(
			formatLibSVMLine(label, x)); err != nil {
			t.Fatal(err)
		}
	}
}

func formatLibSVMLine(label int, x float64) string {
	return strconv.Itoa(label) + " 1:" +
		strconv.FormatFloat(x, 'f', 4, 64) + " 2:1.0\n"
}

func stackTestParams() map[string]string {
	return map[string]string{
		"steps":                  "20",
		"learning-rate":          "0.1",
		"regularization":         "0.0001",
		"alpha":                  "0.1",
		"beta":                   "1.0",
		"lambda1":                "0.01",
		"lambda2":                "1.0",
		"global":                 "-1",
		"learning-rate-discount": "1.0",
	}
}

func TestParseStackMethods(t *testing.T) {
	methods, err := ParseStackMethods(" ftrl , fm,lr ")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"ftrl", "fm", "lr"}
	for i, m := range methods {
		if m != want[i] {
			t.Fatalf("order not preserved: got %v, want %v", methods, want)
		}
	}
	if _, err := ParseStackMethods("ftrl,nope"); err == nil {
		t.Fatal("unknown method should be rejected")
	}
	if _, err := ParseStackMethods("ftrl,ftrl"); err == nil {
		t.Fatal("duplicated method should be rejected")
	}
	if _, err := ParseStackMethods(" , "); err == nil {
		t.Fatal("empty method list should be rejected")
	}
}

func TestStackTrainAndTestEndToEnd(t *testing.T) {
	work := t.TempDir()
	t.Chdir(work) // DataSet.Load writes features.tsv next to the working dir

	trainPath := filepath.Join(work, "train.tsv")
	testPath := filepath.Join(work, "test.tsv")
	modelDir := filepath.Join(work, "model_chain")
	writeLibSVM(t, trainPath, 400)
	writeLibSVM(t, testPath, 200)

	params := stackTestParams()
	methods, err := ParseStackMethods("ftrl,lr")
	if err != nil {
		t.Fatal(err)
	}

	ensemble, err := StackTrain(trainPath, modelDir, methods, 5, params)
	if err != nil {
		t.Fatalf("StackTrain failed: %v", err)
	}
	if len(ensemble.Methods) != 2 || ensemble.Methods[0] != "ftrl" || ensemble.Methods[1] != "lr" {
		t.Fatalf("fixed model order wrong: %v", ensemble.Methods)
	}

	// The persisted chain must contain the manifest, meta model and both
	// full-data base models.
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 4 {
		t.Fatalf("chain dir has %d files, want 4: %v", len(entries), entries)
	}

	// Independent test actually loads and runs the saved chain.
	auc, predictions, err := StackTest(testPath, modelDir, params)
	if err != nil {
		t.Fatalf("StackTest failed: %v", err)
	}
	if len(predictions) != 200 {
		t.Fatalf("got %d predictions, want 200", len(predictions))
	}
	if auc < 0.8 {
		t.Fatalf("stacking AUC unexpectedly low on separable data: %f", auc)
	}
	t.Logf("stacking test AUC: %f", auc)

	// A chain directory without a manifest must be refused.
	badDir := filepath.Join(work, "bad_chain")
	if err := os.MkdirAll(badDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, _, err := StackTest(testPath, badDir, params); err == nil {
		t.Fatal("missing manifest must be rejected")
	}
}
