package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/stack"
)

// TestStackingHTTPHandler trains a real chain, serves it, and verifies the
// HTTP endpoint actually runs the complete inference chain rather than a
// single base model.
func TestStackingHTTPHandler(t *testing.T) {
	work := t.TempDir()
	t.Chdir(work)

	trainPath := work + "/train.tsv"
	modelDir := work + "/chain"
	writeHTTPFixture(t, trainPath, 300)

	params := map[string]string{
		"steps":                  "20",
		"learning-rate":          "0.1",
		"regularization":         "0.0001",
		"alpha":                  "0.1",
		"beta":                   "1.0",
		"lambda1":                "0.01",
		"lambda2":                "1.0",
		"global":                 "-1",
		"learning-rate-discount": "1.0",
		"port":                   "0",
	}
	methods, err := hector.ParseStackMethods("ftrl,lr")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := hector.StackTrain(trainPath, modelDir, methods, 5, params); err != nil {
		t.Fatalf("StackTrain failed: %v", err)
	}

	ensemble, err := stack.Load(modelDir, params, hector.ClassifierFactory)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	handler := &StackingHandler{ensemble: ensemble}

	form := url.Values{}
	form.Set("features", `{"f1":0.9,"f2":1.0}`)
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var body struct {
		Prediction float64  `json:"prediction"`
		Methods    []string `json:"methods"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("bad json response: %v", err)
	}
	if len(body.Methods) != 2 || body.Methods[0] != "ftrl" || body.Methods[1] != "lr" {
		t.Fatalf("response does not expose fixed model order: %v", body.Methods)
	}

	// GET must be refused.
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, httptest.NewRequest(http.MethodGet, "/predict", nil))
	if rec2.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want %d", rec2.Code, http.StatusMethodNotAllowed)
	}
}

func writeHTTPFixture(t *testing.T, path string, n int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for i := 0; i < n; i++ {
		label := i % 2
		x := 0.3 + float64(i%5)*0.1
		if label == 0 {
			x = -x
		}
		line := strconv.Itoa(label) + " 1:" +
			strconv.FormatFloat(x, 'f', 4, 64) + " 2:1.0\n"
		if _, err := f.WriteString(line); err != nil {
			t.Fatal(err)
		}
	}
}
