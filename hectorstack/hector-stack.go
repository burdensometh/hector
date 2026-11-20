/*
hector-stack trains and tests an out-of-fold stacking ensemble.

Train (OOF second layer + full-data refit, whole chain saved into --model dir):

	hector-stack --action train --train train.tsv --model model_dir \
	  --stack-methods ftrl,fm --stack-folds 5

Independent test, actually invoking the saved inference chain:

	hector-stack --action test --test test.tsv --model model_dir --pred out.txt
*/
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/eval"
)

func main() {
	// Register the stacking flags BEFORE PrepareParams, because that helper
	// performs flag.Parse itself.
	stackMethods := flag.String("stack-methods", "ftrl,fm",
		"comma separated base model names, the order is the fixed second-layer column order")
	stackFolds := flag.Int("stack-folds", 5, "number of out-of-fold folds")

	train, test, pred, _, params := hector.PrepareParams()

	action, _ := params["action"]
	modelDir, _ := params["model"]

	methods, err := hector.ParseStackMethods(*stackMethods)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *stackFolds < 2 {
		fmt.Fprintln(os.Stderr, "stack-folds must be at least 2")
		os.Exit(1)
	}

	switch action {
	case "train":
		ensemble, err := hector.StackTrain(train, modelDir, methods, *stackFolds, params)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stacking train failed:", err)
			os.Exit(1)
		}
		fmt.Println("stacking ensemble trained and saved to", modelDir)
		fmt.Println("fixed base model order:", ensemble.Methods)

	case "test":
		auc, predictions, err := hector.StackTest(test, modelDir, params)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stacking test failed:", err)
			os.Exit(1)
		}
		if pred != "" {
			if err := writePredictions(pred, predictions); err != nil {
				fmt.Fprintln(os.Stderr, "write predictions failed:", err)
				os.Exit(1)
			}
		}
		fmt.Println("stacking test AUC:", auc)

	default:
		// No --action: train and test in one run, both going through the
		// persisted inference chain.
		ensemble, err := hector.StackTrain(train, modelDir, methods, *stackFolds, params)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stacking train failed:", err)
			os.Exit(1)
		}
		fmt.Println("stacking ensemble trained and saved to", modelDir)
		fmt.Println("fixed base model order:", ensemble.Methods)
		auc, predictions, err := hector.StackTest(test, modelDir, params)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stacking test failed:", err)
			os.Exit(1)
		}
		if pred != "" {
			if err := writePredictions(pred, predictions); err != nil {
				fmt.Fprintln(os.Stderr, "write predictions failed:", err)
				os.Exit(1)
			}
		}
		fmt.Println("stacking test AUC:", auc)
	}
}

func writePredictions(path string, predictions []*eval.LabelPrediction) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, p := range predictions {
		if _, err := f.WriteString(strconv.FormatFloat(p.Prediction, 'g', 5, 64) + "\n"); err != nil {
			return err
		}
	}
	return nil
}
