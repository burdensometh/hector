package dt

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/xlvector/hector/core"
)

type RandomForestParams struct {
	TreeCount    int
	FeatureCount float64
	// Workers limits how many trees are built concurrently. Zero means one
	// goroutine per tree (the original fully-parallel behavior). The result is
	// identical for any value because every tree's randomness is fixed by its
	// index-derived plan; this only throttles concurrency.
	Workers int
}

type RandomForest struct {
	trees               []*Tree
	params              RandomForestParams
	cart                CART
	continuous_features bool
	seed                int64
}

func (self *RandomForest) SaveModel(path string) {
	file, _ := os.Create(path)
	defer file.Close()
	for _, tree := range self.trees {
		buf := tree.ToString()
		file.Write(buf)
		file.WriteString("\n#\n")
	}
}

func (self *RandomForest) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	self.trees = []*Tree{}
	reader := bufio.NewReader(file)
	text := []string{}
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if line == "#" {
			tree := Tree{}
			tree.fromString(text)
			self.trees = append(self.trees, &tree)
			text = []string{}
		} else {
			text = append(text, line)
		}
	}
}

func (dt *RandomForest) Init(params map[string]string) {
	dt.trees = []*Tree{}
	dt.cart.Init(params)
	tree_count, _ := strconv.ParseInt(params["tree-count"], 10, 64)
	feature_count, _ := strconv.ParseFloat(params["feature-count"], 64)
	dt.params.TreeCount = int(tree_count)
	dt.params.FeatureCount = feature_count
	dt.params.Workers, _ = strconv.Atoi(params["workers"])
	dt.seed = parseRandomSeed(params)
}

func (dt *RandomForest) Train(dataset *core.DataSet) {
	samples := []*core.MapBasedSample{}
	feature_weights := make(map[int64]float64)
	for _, sample := range dataset.Samples {
		if !dt.continuous_features {
			for _, f := range sample.Features {
				_, ok := feature_weights[f.Id]
				if !ok {
					feature_weights[f.Id] = f.Value
				}
				if feature_weights[f.Id] != f.Value {
					dt.continuous_features = true
				}
			}
		}
		msample := sample.ToMapBasedSample()
		samples = append(samples, msample)
	}
	dt.cart.continuous_features = dt.continuous_features

	// Every tree identity gets its random plan up front, in index order.
	// Bootstrap draws, row subsampling and candidate feature selection are
	// therefore fixed per tree and immune to goroutine completion order.
	plans := BuildTreeRandomPlans(dt.seed, dt.params.TreeCount)
	trees := make([]*Tree, dt.params.TreeCount)

	// Optional worker cap: a buffered semaphore. With Workers <= 0 every tree
	// gets its own goroutine, preserving the original parallelism.
	workerCount := dt.params.Workers
	if workerCount <= 0 {
		workerCount = dt.params.TreeCount
	}
	semaphore := make(chan struct{}, workerCount)

	var wait sync.WaitGroup
	wait.Add(dt.params.TreeCount)
	for i := 0; i < dt.params.TreeCount; i++ {
		semaphore <- struct{}{}
		go func(treeID int) {
			defer wait.Done()
			defer func() { <-semaphore }()
			// Each goroutine works on its own CART value and its own plan:
			// nothing mutable is shared between trees, so another forest
			// training in parallel cannot perturb this one.
			cart := dt.cart
			tree := cart.SingleTreeBuild(samples, dt.params.FeatureCount, true, plans[treeID])
			trees[treeID] = &tree
		}(i)
	}
	wait.Wait()

	// Assemble the model by stable tree identity, not by channel arrival
	// order, so the forest (and its averaged predictions) is always identical.
	dt.trees = make([]*Tree, dt.params.TreeCount)
	copy(dt.trees, trees)
}

func (dt *RandomForest) Predict(sample *core.Sample) float64 {
	msample := sample.ToMapBasedSample()
	predictions := 0.0
	total := 0.0
	for _, tree := range dt.trees {
		node, _ := PredictBySingleTree(tree, msample)
		predictions += node.prediction.GetValue(1)
		total += 1.0
	}
	return predictions / total
}

func (dt *RandomForest) PredictMultiClass(sample *core.Sample) *core.ArrayVector {
	msample := sample.ToMapBasedSample()
	predictions := core.NewArrayVector()
	total := 0.0
	for _, tree := range dt.trees {
		node, _ := PredictBySingleTree(tree, msample)
		predictions.AddVector(node.prediction, 1.0)
		total += 1.0
	}
	predictions.Scale(1.0 / total)
	return predictions
}
