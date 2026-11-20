package dt

import (
	"container/list"
	"math/rand"
	"strconv"
	"strings"
	"sync"

	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
)

type TreeNode struct {
	left, right, depth int
	prediction         *core.ArrayVector
	sample_count       int
	samples            []int
	feature_split      core.Feature
}

func (t *TreeNode) ToString() string {
	return strconv.FormatInt(t.feature_split.Id, 10) + ":" + strconv.FormatFloat(t.feature_split.Value, 'g', 3, 64)
}

func (t *TreeNode) AddSample(k int) {
	t.samples = append(t.samples, k)
}

type Tree struct {
	nodes []*TreeNode
}

func (t *Tree) AddTreeNode(n *TreeNode) {
	t.nodes = append(t.nodes, n)
}

func (t *Tree) Size() int {
	return len(t.nodes)
}

func (t *Tree) GetNode(i int) *TreeNode {
	return t.nodes[i]
}

func (t *Tree) ToString() []byte {
	sb := util.StringBuilder{}
	sb.Int(len(t.nodes))
	sb.Write("\n")
	for i, node := range t.nodes {
		sb.Int(i)
		sb.Write("\t")
		sb.Int(node.left)
		sb.Write("\t")
		sb.Int(node.right)
		sb.Write("\t")
		sb.Int(node.depth)
		sb.Write("\t")
		sb.WriteBytes(node.prediction.ToString())
		sb.Write("\t")
		sb.Int(node.sample_count)
		sb.Write("\t")
		sb.Int64(node.feature_split.Id)
		sb.Write("\t")
		sb.Float(node.feature_split.Value)
		sb.Write("\n")
	}
	return sb.Bytes()
}

func (t *Tree) fromString(lines []string) {
	size, _ := strconv.Atoi(lines[0])
	t.nodes = make([]*TreeNode, size+1, size+1)
	for _, line := range lines[1:] {
		if len(line) == 0 {
			break
		}
		tks := strings.Split(line, "\t")
		node := TreeNode{}
		i, _ := strconv.Atoi(tks[0])
		node.left, _ = strconv.Atoi(tks[1])
		node.right, _ = strconv.Atoi(tks[2])
		node.depth, _ = strconv.Atoi(tks[3])
		node.prediction = core.NewArrayVector()
		node.prediction.FromString(tks[4])
		node.sample_count, _ = strconv.Atoi(tks[5])
		node.feature_split = core.Feature{}
		node.feature_split.Id, _ = strconv.ParseInt(tks[6], 10, 64)
		node.feature_split.Value, _ = strconv.ParseFloat(tks[7], 64)
		t.nodes[i] = &node
	}
}

func (t *Tree) FromString(buf string) {
	lines := strings.Split(buf, "\n")
	t.fromString(lines)
}

type RDTParams struct {
	TreeCount   int
	MinLeafSize int
	MaxDepth    int
	// Workers limits concurrent tree building; zero keeps one goroutine per
	// tree. Results do not depend on this value.
	Workers int
}

type RandomDecisionTree struct {
	trees  []*Tree
	params RDTParams
	seed   int64
}

func (self *RandomDecisionTree) SaveModel(path string) {

}

func (self *RandomDecisionTree) LoadModel(path string) {

}

// AppendNodeToTree grows one node. All random choices are drawn from the
// node's tree plan, and feature ids are visited in ascending order so that the
// "random feature" pick cannot depend on Go's randomized map iteration.
func (rdt *RandomDecisionTree) AppendNodeToTree(samples []*core.MapBasedSample, node *TreeNode, queue *list.List, tree *Tree, plan TreeRandomPlan) {
	node.prediction = core.NewArrayVector()
	for _, k := range node.samples {
		node.prediction.AddValue(samples[k].Label, 1.0)
	}
	node.prediction.Scale(1.0 / node.prediction.Sum())

	random_sample := samples[node.samples[plan.Rng.Intn(len(node.samples))]]

	split := core.Feature{Id: -1, Value: -1.0}
	feature_ids := sortedFeatureIds(random_sample.Features)
	for _, fid := range feature_ids {
		// Reproduce the original 1/len(features) per-key draw with the
		// deterministic per-tree RNG over a fixed id ordering.
		if split.Id < 0 || plan.Rng.Intn(len(feature_ids)) == 0 {
			split.Id = fid
			split.Value = random_sample.Features[fid]
		}
	}

	if split.Id < 0 || node.depth > rdt.params.MaxDepth {
		return
	}

	node.feature_split = split
	left_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: nil, sample_count: 0, samples: []int{}}
	right_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: nil, sample_count: 0, samples: []int{}}

	for _, k := range node.samples {
		if DTGoLeft(samples[k], node.feature_split) {
			left_node.samples = append(left_node.samples, k)
		} else {
			right_node.samples = append(right_node.samples, k)
		}
	}
	node.samples = nil

	if len(left_node.samples) == 0 || len(right_node.samples) == 0 {
		return
	}

	if len(left_node.samples) > rdt.params.MinLeafSize {
		queue.PushBack(&left_node)
		node.left = len(tree.nodes)
		tree.AddTreeNode(&left_node)
	}

	if len(right_node.samples) > rdt.params.MinLeafSize {
		queue.PushBack(&right_node)
		node.right = len(tree.nodes)
		tree.AddTreeNode(&right_node)
	}
}

// SingleTreeBuild builds one tree using only the randomness in plan, so the
// same data plus the same plan always produce the same tree.
func (rdt *RandomDecisionTree) SingleTreeBuild(samples []*core.MapBasedSample, plan TreeRandomPlan) Tree {
	tree := Tree{}
	queue := list.New()
	root := TreeNode{depth: 0, left: -1, right: -1, prediction: core.NewArrayVector(), samples: []int{}}

	for i := 0; i < len(samples); i++ {
		k := plan.Rng.Intn(len(samples))
		root.AddSample(k)
		root.prediction.AddValue(samples[k].Label, 1.0)
	}
	root.sample_count = len(root.samples)
	root.prediction.Scale(1.0 / root.prediction.Sum())

	queue.PushBack(&root)
	tree.AddTreeNode(&root)
	for {
		nodes := DTGetElementFromQueue(queue, 10)
		if len(nodes) == 0 {
			break
		}

		for _, node := range nodes {
			rdt.AppendNodeToTree(samples, node, queue, &tree, plan)
		}
	}
	return tree
}

// RandomShuffle performs a Fisher-Yates shuffle with an explicit RNG, keeping
// the result reproducible for a given plan instead of sharing global state.
func (rdt *RandomDecisionTree) RandomShuffle(features []core.Feature, rng *rand.Rand) {
	for i := range features {
		j := rng.Intn(i + 1)
		features[i], features[j] = features[j], features[i]
	}
}

func (rdt *RandomDecisionTree) Train(dataset *core.DataSet) {
	samples := []*core.MapBasedSample{}
	for _, sample := range dataset.Samples {
		samples = append(samples, sample.ToMapBasedSample())
	}
	dataset.Samples = nil

	// Per-tree plans are derived by stable identity before any goroutine runs,
	// and results are written back at their index, so worker scheduling cannot
	// change either a tree's draws or its position in the model.
	plans := BuildTreeRandomPlans(rdt.seed, rdt.params.TreeCount)
	forest := make([]*Tree, rdt.params.TreeCount)

	workerCount := rdt.params.Workers
	if workerCount <= 0 {
		workerCount = rdt.params.TreeCount
	}
	semaphore := make(chan struct{}, workerCount)

	var wait sync.WaitGroup
	wait.Add(rdt.params.TreeCount)
	for k := 0; k < rdt.params.TreeCount; k++ {
		semaphore <- struct{}{}
		go func(treeID int) {
			defer wait.Done()
			defer func() { <-semaphore }()
			tree := rdt.SingleTreeBuild(samples, plans[treeID])
			forest[treeID] = &tree
		}(k)
	}
	wait.Wait()
	rdt.trees = make([]*Tree, rdt.params.TreeCount)
	copy(rdt.trees, forest)
}

func (rdt *RandomDecisionTree) Predict(sample *core.Sample) float64 {
	ret := 0.0
	total := 0.0
	msample := sample.ToMapBasedSample()
	for _, tree := range rdt.trees {
		node, _ := PredictBySingleTree(tree, msample)
		ret += node.prediction.GetValue(1)
		total += 1.0
	}
	return ret / total
}

func (rdt *RandomDecisionTree) PredictMultiClass(sample *core.Sample) *core.ArrayVector {
	msample := sample.ToMapBasedSample()
	predictions := core.NewArrayVector()
	total := 0.0
	for _, tree := range rdt.trees {
		node, _ := PredictBySingleTree(tree, msample)
		predictions.AddVector(node.prediction, 1.0)
		total += 1.0
	}
	predictions.Scale(1.0 / total)
	return predictions
}

func (rdt *RandomDecisionTree) Init(params map[string]string) {
	rdt.trees = []*Tree{}
	rdt.params.MinLeafSize, _ = strconv.Atoi(params["min-leaf-size"])
	rdt.params.TreeCount, _ = strconv.Atoi(params["tree-count"])
	rdt.params.MaxDepth, _ = strconv.Atoi(params["max-depth"])
	rdt.params.Workers, _ = strconv.Atoi(params["workers"])
	rdt.seed = parseRandomSeed(params)
}
