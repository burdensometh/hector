package dt

import (
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"testing"

	"github.com/xlvector/hector/core"
)

func rfParams(seed int64, workers int) map[string]string {
	return map[string]string{
		"tree-count":      "40",
		"feature-count":   "0.8",
		"max-depth":       "10",
		"min-leaf-size":   "5",
		"gini":            "1.0",
		"dt-sample-ratio": "0.9",
		"seed":            strconv.FormatInt(seed, 10),
		"workers":         strconv.Itoa(workers),
	}
}

func rdtParams(seed int64, workers int) map[string]string {
	return map[string]string{
		"tree-count":    "40",
		"max-depth":     "10",
		"min-leaf-size": "5",
		"seed":          strconv.FormatInt(seed, 10),
		"workers":       strconv.Itoa(workers),
	}
}

// fixedDataSet builds the same training and probe sets for every call, so any
// difference between models comes from training behavior, not input data. A
// fresh copy is returned each time because RDT.Train nils out dataset.Samples.
func fixedDataSet() (*core.DataSet, *core.DataSet) {
	rand.Seed(20261120)
	train := core.XORDataSet(800)
	probe := core.XORDataSet(200)
	return train, probe
}

func forestSignature(trees []*Tree) []string {
	sig := make([]string, len(trees))
	for i, tree := range trees {
		sig[i] = string(tree.ToString())
	}
	return sig
}

func sameSignature(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func probePredictions(t *testing.T, f *RandomForest, probe *core.DataSet) []float64 {
	out := make([]float64, 0, len(probe.Samples))
	for _, s := range probe.Samples {
		out = append(out, f.Predict(s))
	}
	return out
}

func probeRDTPredictions(t *testing.T, r *RandomDecisionTree, probe *core.DataSet) []float64 {
	out := make([]float64, 0, len(probe.Samples))
	for _, s := range probe.Samples {
		out = append(out, r.Predict(s))
	}
	return out
}

func floatsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func trainRF(t *testing.T, params map[string]string, dataset *core.DataSet) *RandomForest {
	rf := &RandomForest{}
	rf.Init(params)
	rf.Train(dataset)
	if len(rf.trees) != rf.params.TreeCount {
		t.Fatalf("expected %d trees, got %d", rf.params.TreeCount, len(rf.trees))
	}
	return rf
}

func trainRDT(t *testing.T, params map[string]string, dataset *core.DataSet) *RandomDecisionTree {
	rdt := &RandomDecisionTree{}
	rdt.Init(params)
	rdt.Train(dataset)
	if len(rdt.trees) != rdt.params.TreeCount {
		t.Fatalf("expected %d trees, got %d", rdt.params.TreeCount, len(rdt.trees))
	}
	return rdt
}

// TestRFDeterministicAcrossWorkerCounts: same data and seed must give the same
// forest (tree by tree) for any worker count and any GOMAXPROCS setting.
func TestRFDeterministicAcrossWorkerCounts(t *testing.T) {
	train, probe := fixedDataSet()

	prevProcs := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prevProcs)

	baseline := trainRF(t, rfParams(7, 0), train)
	baseSig := forestSignature(baseline.trees)
	basePred := probePredictions(t, baseline, probe)

	for _, procs := range []int{1, 2, 4, 8} {
		runtime.GOMAXPROCS(procs)
		for _, workers := range []int{1, 2, 7, 40, 0} {
			rf := trainRF(t, rfParams(7, workers), train)
			if !sameSignature(baseSig, forestSignature(rf.trees)) {
				t.Fatalf("rf forest changed with workers=%d GOMAXPROCS=%d", workers, procs)
			}
			if got := probePredictions(t, rf, probe); !floatsEqual(basePred, got) {
				t.Fatalf("rf predictions changed with workers=%d GOMAXPROCS=%d", workers, procs)
			}
		}
	}
}

// TestRFStableTreeIdentity: repeated independent runs must agree per tree
// index, not just as an unordered ensemble.
func TestRFStableTreeIdentity(t *testing.T) {
	train, _ := fixedDataSet()
	first := forestSignature(trainRF(t, rfParams(99, 3), train).trees)
	second := forestSignature(trainRF(t, rfParams(99, 3), train).trees)
	if !sameSignature(first, second) {
		t.Fatal("two rf runs with identical config produced different trees")
	}
}

// TestRFConcurrentForestsDoNotInterfere: training a second forest at the same
// time must not perturb the first one, even when both share a seed and data.
func TestRFConcurrentForestsDoNotInterfere(t *testing.T) {
	train, probe := fixedDataSet()
	baseline := trainRF(t, rfParams(13, 2), train)
	baseSig := forestSignature(baseline.trees)

	for round := 0; round < 5; round++ {
		var wg sync.WaitGroup
		results := make([][]string, 4)
		for g := 0; g < 4; g++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				// Mix of same and different seeds, all sharing one dataset.
				rf := trainRF(t, rfParams(int64(13+idx), idx+1), train)
				results[idx] = forestSignature(rf.trees)
			}(g)
		}
		wg.Wait()
		// Group 0 reuses seed 13: it must match the solo baseline exactly.
		if !sameSignature(baseSig, results[0]) {
			t.Fatalf("concurrent training perturbed forest in round %d", round)
		}
		// The other seeds must each change the sampling outcome.
		for idx := 1; idx < 4; idx++ {
			if sameSignature(baseSig, results[idx]) {
				t.Fatalf("seed %d produced the same forest as seed 13", 13+idx)
			}
		}
	}

	// And the baseline's predictions must still be intact afterwards.
	again := trainRF(t, rfParams(13, 2), train)
	if !sameSignature(baseSig, forestSignature(again.trees)) {
		t.Fatal("rf forest drifted after concurrent training")
	}
	if got := probePredictions(t, again, probe); !floatsEqual(probePredictions(t, baseline, probe), got) {
		t.Fatal("rf predictions drifted after concurrent training")
	}
}

// TestRFSeedChangesSampling: a different seed must change bootstrap draws and
// therefore the trees.
func TestRFSeedChangesSampling(t *testing.T) {
	train, _ := fixedDataSet()
	sigA := forestSignature(trainRF(t, rfParams(1, 4), train).trees)
	sigB := forestSignature(trainRF(t, rfParams(2, 4), train).trees)
	if sameSignature(sigA, sigB) {
		t.Fatal("different rf seeds produced identical forests")
	}
}

// TestRFSaveLoadKeepsPredictions: a reloaded model must predict exactly what
// the in-memory model predicted.
func TestRFSaveLoadKeepsPredictions(t *testing.T) {
	train, probe := fixedDataSet()
	rf := trainRF(t, rfParams(7, 3), train)
	before := probePredictions(t, rf, probe)

	path := filepath.Join(t.TempDir(), "rf.model")
	rf.SaveModel(path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("model not saved: %v", err)
	}

	loaded := &RandomForest{}
	loaded.LoadModel(path)
	if len(loaded.trees) != len(rf.trees) {
		t.Fatalf("reloaded tree count %d != %d", len(loaded.trees), len(rf.trees))
	}
	after := probePredictions(t, loaded, probe)
	if !floatsEqual(before, after) {
		t.Fatal("predictions changed after save/load")
	}
}

// TestRFIgnoresGlobalRand: perturbing the package-global rand source between
// runs must have no effect, because training never reads from it.
func TestRFIgnoresGlobalRand(t *testing.T) {
	train, _ := fixedDataSet()
	first := forestSignature(trainRF(t, rfParams(21, 2), train).trees)
	rand.Seed(1)
	rand.Int63()
	rand.Float64()
	second := forestSignature(trainRF(t, rfParams(21, 2), train).trees)
	if !sameSignature(first, second) {
		t.Fatal("rf forest depends on the global rand source")
	}
}

// TestRDTDeterministicAcrossWorkerCounts applies the same guarantees to the
// random decision tree ensemble.
func TestRDTDeterministicAcrossWorkerCounts(t *testing.T) {
	train, probe := fixedDataSet()

	prevProcs := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prevProcs)

	baseline := trainRDT(t, rdtParams(7, 0), train)
	baseSig := forestSignature(baseline.trees)
	basePred := probeRDTPredictions(t, baseline, probe)

	for _, procs := range []int{1, 4, 8} {
		runtime.GOMAXPROCS(procs)
		for _, workers := range []int{1, 2, 7, 40, 0} {
			ds, _ := fixedDataSet()
			rdt := trainRDT(t, rdtParams(7, workers), ds)
			if !sameSignature(baseSig, forestSignature(rdt.trees)) {
				t.Fatalf("rdt forest changed with workers=%d GOMAXPROCS=%d", workers, procs)
			}
			if got := probeRDTPredictions(t, rdt, probe); !floatsEqual(basePred, got) {
				t.Fatalf("rdt predictions changed with workers=%d GOMAXPROCS=%d", workers, procs)
			}
		}
	}
}

func TestRDTSeedChangesSampling(t *testing.T) {
	dsA, _ := fixedDataSet()
	dsB, _ := fixedDataSet()
	sigA := forestSignature(trainRDT(t, rdtParams(1, 4), dsA).trees)
	sigB := forestSignature(trainRDT(t, rdtParams(2, 4), dsB).trees)
	if sameSignature(sigA, sigB) {
		t.Fatal("different rdt seeds produced identical forests")
	}
}

func TestRDTConcurrentForestsDoNotInterfere(t *testing.T) {
	// Build every dataset up front in a single goroutine: fixedDataSet() draws
	// from the global rand source, so calling it concurrently would interleave
	// draws and make the *inputs* nondeterministic. Pre-building guarantees all
	// forests train on byte-identical, independently owned datasets.
	baseDS, _ := fixedDataSet()
	baseline := forestSignature(trainRDT(t, rdtParams(13, 2), baseDS).trees)

	datasets := make([]*core.DataSet, 4)
	for g := range datasets {
		datasets[g], _ = fixedDataSet()
	}

	var wg sync.WaitGroup
	results := make([][]string, 4)
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			rdt := trainRDT(t, rdtParams(int64(13+idx), idx+1), datasets[idx])
			results[idx] = forestSignature(rdt.trees)
		}(g)
	}
	wg.Wait()
	if !sameSignature(baseline, results[0]) {
		t.Fatal("concurrent rdt training perturbed the shared-seed forest")
	}
}

// TestCARTDeterministic: a standalone CART must also be reproducible from its
// seed and independent of the global rand source (its salt used to be random).
func TestCARTDeterministic(t *testing.T) {
	train, _ := fixedDataSet()
	params := map[string]string{
		"max-depth":       "8",
		"min-leaf-size":   "5",
		"gini":            "1.0",
		"dt-sample-ratio": "0.9",
		"seed":            "55",
	}
	build := func() string {
		c := &CART{}
		c.Init(params)
		c.Train(train)
		return string(c.tree.ToString())
	}
	first := build()
	rand.Seed(12345)
	second := build()
	if first != second {
		t.Fatal("standalone CART is not reproducible under global rand perturbation")
	}
	other := map[string]string{}
	for k, v := range params {
		other[k] = v
	}
	other["seed"] = "56"
	c := &CART{}
	c.Init(other)
	c.Train(train)
	if string(c.tree.ToString()) == first {
		t.Fatal("different CART seeds produced identical trees")
	}
}
