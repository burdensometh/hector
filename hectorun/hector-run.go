package main

import (
	"fmt"
	"runtime"

	"github.com/xlvector/hector"
)

func main() {
	train, test, pred, method, params := hector.PrepareParams()

	action, _ := params["action"]

	classifier := hector.GetClassifier(method)
	runtime.GOMAXPROCS(runtime.NumCPU())
	if action == "" {
		auc, report, _, err := hector.AlgorithmRunWithReport(classifier, train, test, pred, params)
		if err != nil {
			panic(err)
		}
		fmt.Println("AUC:")
		fmt.Println(auc)
		printWeightedMetrics(report)
	} else if action == "train" {
		if err := hector.AlgorithmTrain(classifier, train, params); err != nil {
			panic(err)
		}

	} else if action == "test" {
		auc, report, _, err := hector.AlgorithmTestWithReport(classifier, test, pred, params)
		if err != nil {
			panic(err)
		}
		fmt.Println("AUC:")
		fmt.Println(auc)
		printWeightedMetrics(report)
	}
}

// printWeightedMetrics reports the weighted metrics next to the legacy AUC so
// the numbers in the final table are on the same representative population the
// model was trained on.
func printWeightedMetrics(report *hector.WeightReport) {
	if report == nil {
		return
	}
	fmt.Printf("LogLoss: %v\n", report.Metrics.LogLoss)
	fmt.Printf("WeightedAUC: %v\n", report.Metrics.WeightedAUC)
	fmt.Printf("WeightedLogLoss: %v\n", report.Metrics.WeightedLogLoss)
	fmt.Printf("WeightedPopulation: %v\n", report.Metrics.WeightedPopulation)
}
