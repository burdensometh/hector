/*
hector-stack-server serves a saved out-of-fold stacking ensemble over HTTP.

It loads the complete inference chain (full-data base models + meta model with
the fixed base-model order) from the --model directory and every /predict
request actually runs that chain: each base model scores the sample, its
prediction is placed in the fixed method-order column and the meta model
produces the final result.

Start:

	hector-stack-server --model model_dir --port 8080

Request:

	curl -XPOST localhost:8080/predict -d 'features={"age":0.7,"income":0.1}'
*/
package main

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/stack"
	"github.com/xlvector/hector/util"
)

type StackingHandler struct {
	ensemble *stack.StackingEnsemble
}

func (h *StackingHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	features := req.FormValue("features")
	if len(features) == 0 {
		http.Error(w, "need input features", http.StatusInternalServerError)
		return
	}
	fs := make(map[string]float64)
	if err := json.Unmarshal([]byte(features), &fs); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	sample := core.NewSample()
	for k, v := range fs {
		sample.AddFeature(core.Feature{Id: util.Hash(k), Value: v})
	}

	// The whole saved chain runs here: base models in fixed order -> meta.
	prediction, err := h.ensemble.Predict(sample)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	output, err := json.Marshal(map[string]interface{}{
		"prediction": prediction,
		"methods":    h.ensemble.Methods,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// Write, not fmt.Fprint: Fprint renders a []byte as [123 34 ...] instead
	// of emitting the JSON bytes.
	if _, err := w.Write(output); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func main() {
	_, _, _, _, params := hector.PrepareParams()
	modelDir, ok := params["model"]
	if !ok || modelDir == "" {
		log.Fatalln("please input stacking model directory via --model")
	}
	ensemble, err := stack.Load(modelDir, params, hector.ClassifierFactory)
	if err != nil {
		log.Fatalln("load stacking inference chain failed:", err)
	}
	log.Println("stacking ensemble loaded, fixed base model order:", ensemble.Methods)
	http.Handle("/predict", &StackingHandler{ensemble: ensemble})
	if err := http.ListenAndServe(":"+params["port"], nil); err != nil {
		log.Fatal(err)
	}
}
