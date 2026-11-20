package lr

import "github.com/xlvector/hector/core"

// Ni returns the accumulated squared-gradient accumulator ni for feature id.
// It is primarily exposed for testing and diagnostics of resume correctness.
func (w FTRLFeatureWeight) Ni() float64 { return w.ni }

// Zi returns the dual accumulator zi for feature id.
func (w FTRLFeatureWeight) Zi() float64 { return w.zi }

// UpdateSample exposes one deterministic FTRL update for callers that need to
// reconstruct an exact training position (e.g. seeding a crash-resume test).
// Production training goes through Train / TrainBatched.
func (algo *FTRLLogisticRegression) UpdateSample(sample *core.Sample) {
	algo.updateSample(sample)
}

// NewCheckpoint builds a checkpoint envelope from explicit resume fields. It
// is intended for tooling and tests; normal training writes checkpoints itself.
func NewCheckpoint(model map[int64]FTRLFeatureWeight, params FTRLLogisticRegressionParams,
	epoch, nextSampleIndex, batchSize int, updates int64,
	inputFingerprint string) *ftrlCheckpoint {
	return &ftrlCheckpoint{
		Version:           ftrlCheckpointVersion,
		Model:             model,
		Params:            params,
		Epoch:             epoch,
		NextSampleIndex:   nextSampleIndex,
		BatchSize:         batchSize,
		Updates:           updates,
		InputFingerprint:  inputFingerprint,
		ConfigFingerprint: FTRLConfigFingerprint(params, batchSize),
	}
}

// SaveCheckpoint writes cp to path using the same atomic path as training.
func SaveCheckpoint(path string, cp *ftrlCheckpoint) error {
	return saveCheckpoint(path, cp)
}
