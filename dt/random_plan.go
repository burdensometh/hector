package dt

import (
	"math/rand"
)

// DefaultRandomSeed is used when no "seed" parameter is supplied. Fixing it
// keeps repeated runs on the same data reproducible; passing a different
// "seed" still changes every bootstrap draw and feature draw below.
const DefaultRandomSeed int64 = 42

// TreeRandomPlan bundles every independent random decision a single tree is
// allowed to make. One plan is pre-derived per tree identity (its index in the
// forest) before any worker goroutine starts, so the stream of random numbers a
// tree consumes depends only on the model seed and the tree index, never on
// goroutine scheduling, the number of workers, or another forest training in
// the same process.
//
// Each tree gets its own *rand.Rand: the global rand source is never touched
// during training, so two forests built concurrently cannot interleave draws
// on shared state.
type TreeRandomPlan struct {
	// Rng drives bootstrap sampling, row subsampling and the random choices
	// inside a single tree. It is used by one goroutine at most and needs no
	// locking.
	Rng *rand.Rand
	// Salt selects the candidate feature subset at each node through
	// RandByFeatureId. It is derived from Rng at planning time, so it is fixed
	// for this tree identity regardless of execution timing.
	Salt int64
}

// NewTreeRandomPlan derives the plan for the tree with the given stable
// identity. treeSeed is itself drawn sequentially from the forest's master
// source, which makes per-tree streams independent yet fully reproducible.
func NewTreeRandomPlan(treeSeed int64) TreeRandomPlan {
	rng := rand.New(rand.NewSource(treeSeed))
	return TreeRandomPlan{
		Rng:  rng,
		Salt: rng.Int63(),
	}
}

// BuildTreeRandomPlans deterministically derives one plan per tree from the
// forest-level seed. Derivation happens up front in index order, so plans[i]
// is always the plan for tree i, no matter in which order workers finish.
func BuildTreeRandomPlans(seed int64, treeCount int) []TreeRandomPlan {
	master := rand.New(rand.NewSource(seed))
	plans := make([]TreeRandomPlan, treeCount)
	for i := 0; i < treeCount; i++ {
		plans[i] = NewTreeRandomPlan(master.Int63())
	}
	return plans
}
