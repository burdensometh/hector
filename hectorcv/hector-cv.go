package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
)

func SplitFile(dataset *core.DataSet, total, part int) (*core.DataSet, *core.DataSet) {
	// DataSet.Split carries the per-row weights along with the samples, so the
	// training and evaluation folds stay on the same weighted population.
	train := dataset.Split(func(i int) bool { return i%total != part })
	test := dataset.Split(func(i int) bool { return i%total == part })
	return train, test
}

func main() {
	train_path, _, _, method, params := hector.PrepareParams()
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	profile, _ := params["profile"]
	dataset := core.NewDataSet()
	dataset.Load(train_path, global)
	if err := hector.AttachWeightsFromParams(dataset, params, hector.ParamsWeightsPath); err != nil {
		log.Fatal(err)
	}
	if err := hector.ValidateTrainingWeights(dataset); err != nil {
		log.Fatal(err)
	}
	runtime.GOMAXPROCS(runtime.NumCPU())
	cv, _ := strconv.ParseInt(params["cv"], 10, 32)
	total := int(cv)

	if profile != "" {
		fmt.Println(profile)
		f, err := os.Create(profile)
		if err != nil {
			fmt.Println(err)
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	average_auc := 0.0
	average_weighted_auc := 0.0
	average_weighted_logloss := 0.0
	has_weights := false
	for part := 0; part < total; part++ {
		train, test := SplitFile(dataset, total, part)
		classifier := hector.GetClassifier(method)
		classifier.Init(params)
		auc, report, _ := hector.AlgorithmRunOnDataSetReport(classifier, train, test, "", params)
		if report.Error != "" {
			log.Fatal(report.Error)
		}
		fmt.Println("AUC:")
		fmt.Println(auc)
		fmt.Printf("WeightedAUC: %v\n", report.Metrics.WeightedAUC)
		fmt.Printf("WeightedLogLoss: %v\n", report.Metrics.WeightedLogLoss)
		average_auc += auc
		average_weighted_auc += report.Metrics.WeightedAUC
		average_weighted_logloss += report.Metrics.WeightedLogLoss
		has_weights = has_weights || report.Metrics.HasWeights
		classifier = nil
	}
	fmt.Println(average_auc / float64(total))
	if has_weights {
		fmt.Println("mean weighted AUC:", average_weighted_auc/float64(total))
		fmt.Println("mean weighted logloss:", average_weighted_logloss/float64(total))
	}
}
