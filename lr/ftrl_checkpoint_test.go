package lr

import (
	"encoding/json"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/xlvector/hector/core"
)

func ftrlTestParams() map[string]string {
	return map[string]string{
		"alpha":   "0.1",
		"beta":    "1.0",
		"lambda1": "0.1",
		"lambda2": "1.0",
		"steps":   "3",
	}
}

// sameModel reports whether two FTRL models hold identical (ni, zi) state for
// the same feature set.
func sameModel(a, b map[int64]FTRLFeatureWeight) bool {
	if len(a) != len(b) {
		return false
	}
	for k, va := range a {
		vb, ok := b[k]
		if !ok || va.ni != vb.ni || va.zi != vb.zi {
			return false
		}
	}
	return true
}

// sampleFingerprint mimics what the runner computes over the source file:
// a stable content hash of the serialized dataset used in the test.
func sampleFingerprint(t *testing.T, dataset *core.DataSet) string {
	content := datasetToBytes(dataset)
	fp, err := FingerprintReader(strings.NewReader(string(content)))
	if err != nil {
		t.Fatalf("fingerprint: %v", err)
	}
	return fp
}

func datasetToBytes(dataset *core.DataSet) []byte {
	var sb strings.Builder
	for _, s := range dataset.Samples {
		sb.WriteString(string(s.ToString(false)))
		sb.WriteString("\n")
	}
	return []byte(sb.String())
}

func makeLinearDataSet(n int) *core.DataSet {
	dataset := core.NewDataSet()
	for i := 0; i < n; i++ {
		sample := core.NewSample()
		sample.Label = 0
		for f := 0; f < 20; f++ {
			if rand.Intn(3) == 0 {
				sample.Label = 1
				sample.AddFeature(core.Feature{Id: int64(f), Value: 1.0})
			}
		}
		dataset.AddSample(sample)
	}
	return dataset
}

// 1. Batched checkpointed training must produce exactly the same parameters
// as the plain, uninterrupted Train loop, across every batch size and across
// epoch boundaries.
func TestTrainBatchedMatchesUninterrupted(t *testing.T) {
	rand.Seed(42)
	dataset := makeLinearDataSet(57) // deliberately prime: partial batches
	fingerprint := sampleFingerprint(t, dataset)

	reference := &FTRLLogisticRegression{}
	reference.Init(ftrlTestParams())
	reference.Train(dataset)

	for _, batchSize := range []int{1, 7, 57, 100} {
		dir := t.TempDir()
		cpPath := filepath.Join(dir, "ftrl.ckpt")

		batched := &FTRLLogisticRegression{}
		batched.Init(ftrlTestParams())
		if err := batched.TrainBatched(dataset, batchSize, cpPath, fingerprint); err != nil {
			t.Fatalf("batchSize=%d: %v", batchSize, err)
		}
		if !sameModel(reference.Model, batched.Model) {
			t.Fatalf("batchSize=%d: batched model diverged from uninterrupted training", batchSize)
		}

		// Final checkpoint must describe a fully completed run.
		cp, err := loadCheckpoint(cpPath, fingerprint,
			ftrlConfigFingerprint(reference.Params, batchSize))
		if err != nil {
			t.Fatalf("reload final checkpoint: %v", err)
		}
		if cp.Epoch != reference.Params.Steps || cp.NextSampleIndex != 0 {
			t.Fatalf("batchSize=%d: final position epoch=%d index=%d, want %d/0",
				batchSize, cp.Epoch, cp.NextSampleIndex, reference.Params.Steps)
		}
		if cp.Updates != int64(reference.Params.Steps*len(dataset.Samples)) {
			t.Fatalf("batchSize=%d: updates=%d", batchSize, cp.Updates)
		}
	}
}

// 2. Simulate a crash after every possible confirmed batch position inside
// the FIRST epoch: the process dies with a checkpoint describing
// (epoch=0, next=crashAfter). A brand-new receiver ("new process") resumes on
// the full dataset, replays only unconfirmed batches, and must end in exactly
// the uninterrupted state — neither skipping nor double-counting samples.
func TestResumeReplaysOnlyUnconfirmedBatches(t *testing.T) {
	rand.Seed(7)
	params := ftrlTestParams()
	dataset := makeLinearDataSet(30)
	fingerprint := sampleFingerprint(t, dataset)
	batchSize := 8
	n := len(dataset.Samples)

	reference := &FTRLLogisticRegression{}
	reference.Init(params)
	reference.Train(dataset)

	for crashAfter := batchSize; crashAfter <= n; crashAfter += batchSize {
		// Reproduce what trainBatched would durably record at this exact
		// crash point: update samples [0, crashAfter) in order, checkpoint.
		crashed := &FTRLLogisticRegression{}
		crashed.Init(params)
		for i := 0; i < crashAfter; i++ {
			crashed.updateSample(dataset.Samples[i])
		}
		crashDir := t.TempDir()
		crashPath := filepath.Join(crashDir, "ftrl.ckpt")
		cp := &ftrlCheckpoint{
			Version:           ftrlCheckpointVersion,
			Model:             crashed.Model,
			Params:            crashed.Params,
			Epoch:             0,
			NextSampleIndex:   crashAfter,
			BatchSize:         batchSize,
			Updates:           int64(crashAfter),
			InputFingerprint:  fingerprint,
			ConfigFingerprint: FTRLConfigFingerprint(crashed.Params, batchSize),
		}
		if err := saveCheckpoint(crashPath, cp); err != nil {
			t.Fatalf("seed checkpoint at %d: %v", crashAfter, err)
		}

		// New process: fresh receiver, only the checkpoint file survives.
		resumed := &FTRLLogisticRegression{}
		if err := resumed.ResumeOrTrainBatched(dataset, params, batchSize,
			crashPath, fingerprint); err != nil {
			t.Fatalf("resume at %d: %v", crashAfter, err)
		}
		if !sameModel(reference.Model, resumed.Model) {
			t.Fatalf("resume after crash at sample %d diverged from uninterrupted run", crashAfter)
		}

		// Resuming the now-complete checkpoint again must be a no-op.
		again := &FTRLLogisticRegression{}
		if err := again.ResumeOrTrainBatched(dataset, params, batchSize,
			crashPath, fingerprint); err != nil {
			t.Fatal(err)
		}
		if !sameModel(reference.Model, again.Model) {
			t.Fatal("repeated resume of completed checkpoint changed the model")
		}
	}
}

// 3. A checkpoint bound to a different input file must be rejected, and the
// receiver must not be mutated by the failed resume.
func TestResumeRejectsChangedInput(t *testing.T) {
	rand.Seed(11)
	dataset := makeLinearDataSet(20)
	fingerprint := sampleFingerprint(t, dataset)
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")

	a := &FTRLLogisticRegression{}
	if err := a.ResumeOrTrainBatched(dataset, ftrlTestParams(), 5, cpPath, fingerprint); err != nil {
		t.Fatal(err)
	}

	other := &FTRLLogisticRegression{}
	err := other.ResumeOrTrainBatched(dataset, ftrlTestParams(), 5, cpPath, fingerprint+"deadbeef")
	if err == nil || !strings.Contains(err.Error(), "input data changed") {
		t.Fatalf("expected input-changed rejection, got %v", err)
	}
	if len(other.Model) != 0 {
		t.Fatal("failed resume must not populate model state")
	}
}

// 4. A checkpoint bound to different hyper-parameters or batch size (i.e. a
// changed training definition) must be rejected.
func TestResumeRejectsChangedTrainingDefinition(t *testing.T) {
	rand.Seed(13)
	dataset := makeLinearDataSet(20)
	fingerprint := sampleFingerprint(t, dataset)
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")

	a := &FTRLLogisticRegression{}
	if err := a.ResumeOrTrainBatched(dataset, ftrlTestParams(), 5, cpPath, fingerprint); err != nil {
		t.Fatal(err)
	}

	changed := ftrlTestParams()
	changed["alpha"] = "0.2"
	b := &FTRLLogisticRegression{}
	err := b.ResumeOrTrainBatched(dataset, changed, 5, cpPath, fingerprint)
	if err == nil || !strings.Contains(err.Error(), "training definition changed") {
		t.Fatalf("expected definition-changed rejection, got %v", err)
	}

	// Same hyper-parameters, different batch size -> different definition.
	c := &FTRLLogisticRegression{}
	err = c.ResumeOrTrainBatched(dataset, ftrlTestParams(), 10, cpPath, fingerprint)
	if err == nil || !strings.Contains(err.Error(), "training definition changed") {
		t.Fatalf("expected batch-size rejection, got %v", err)
	}
	if len(c.Model) != 0 {
		t.Fatal("failed resume must not populate model state")
	}
}

// 5. A corrupted/malformed checkpoint is rejected rather than treated as a
// fresh start.
func TestResumeRejectsCorruptedCheckpoint(t *testing.T) {
	dataset := makeLinearDataSet(10)
	fingerprint := sampleFingerprint(t, dataset)
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")
	if err := ioutil.WriteFile(cpPath, []byte("{not json"), 0600); err != nil {
		t.Fatal(err)
	}
	a := &FTRLLogisticRegression{}
	if err := a.ResumeOrTrainBatched(dataset, ftrlTestParams(), 4, cpPath, fingerprint); err == nil {
		t.Fatal("expected malformed checkpoint to be rejected")
	}
	if len(a.Model) != 0 {
		t.Fatal("failed resume must not populate model state")
	}
}

// 6. If the checkpoint cannot be written (path inside a non-existent
// directory), the error must surface and no file must appear; and when a
// valid checkpoint already exists, a later failed save leaves the previous
// complete version byte-for-byte intact.
func TestSaveFailureKeepsPreviousCompleteVersion(t *testing.T) {
	rand.Seed(17)
	dataset := makeLinearDataSet(24)
	fingerprint := sampleFingerprint(t, dataset)

	// Unwritable path: parent directory does not exist.
	tmpRoot := t.TempDir()
	badPath := filepath.Join(tmpRoot, "no-such-dir", "ftrl.ckpt")
	a := &FTRLLogisticRegression{}
	err := a.ResumeOrTrainBatched(dataset, ftrlTestParams(), 8, badPath, fingerprint)
	if err == nil {
		t.Fatal("expected checkpoint write failure")
	}
	if _, statErr := os.Stat(badPath); !os.IsNotExist(statErr) {
		t.Fatal("no checkpoint file should exist after failed save")
	}
	matches, _ := filepath.Glob(filepath.Join(tmpRoot, "*.tmp-*"))
	if len(matches) != 0 {
		t.Fatalf("temporary checkpoint files leaked: %v", matches)
	}

	// Good first version, then force a failure during the resumed run's next
	// save and ensure the first version survives byte-for-byte. The
	// checkpoint is deliberately left mid-first-epoch so resume must train
	// and therefore attempt another checkpoint write.
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")
	seed := &FTRLLogisticRegression{}
	seed.Init(ftrlTestParams())
	for _, s := range dataset.Samples[:8] {
		seed.updateSample(s)
	}
	seedCP := &ftrlCheckpoint{
		Version:           ftrlCheckpointVersion,
		Model:             seed.Model,
		Params:            seed.Params,
		Epoch:             0,
		NextSampleIndex:   8,
		BatchSize:         8,
		Updates:           8,
		InputFingerprint:  fingerprint,
		ConfigFingerprint: FTRLConfigFingerprint(seed.Params, 8),
	}
	if err := saveCheckpoint(cpPath, seedCP); err != nil {
		t.Fatal(err)
	}
	goodBytes, err := ioutil.ReadFile(cpPath)
	if err != nil {
		t.Fatal(err)
	}

	// Make the directory read-only so temp-file creation fails during the
	// next checkpoint save.
	if err := os.Chmod(dir, 0500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0700) })

	second := &FTRLLogisticRegression{}
	err = second.ResumeOrTrainBatched(dataset, ftrlTestParams(), 8, cpPath, fingerprint)
	if err == nil {
		t.Fatal("expected checkpoint save to fail in read-only directory")
	}
	survivingBytes, err := ioutil.ReadFile(cpPath)
	if err != nil {
		t.Fatalf("previous checkpoint vanished: %v", err)
	}
	if string(survivingBytes) != string(goodBytes) {
		t.Fatal("previous complete checkpoint was modified by failed save")
	}
}

// 7. The plain model file (ni/zi only) remains an inference artifact: it can
// be loaded for prediction, but it cannot masquerade as a resumed optimizer.
func TestInferenceModelIsNotResumeState(t *testing.T) {
	rand.Seed(19)
	dataset := makeLinearDataSet(20)
	fingerprint := sampleFingerprint(t, dataset)
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")
	modelPath := filepath.Join(dir, "model.tsv")

	trainer := &FTRLLogisticRegression{}
	if err := trainer.ResumeOrTrainBatched(dataset, ftrlTestParams(), 5, cpPath, fingerprint); err != nil {
		t.Fatal(err)
	}
	trainer.SaveModel(modelPath)

	// The exported model file has no checkpoint envelope: trying to decode it
	// as a checkpoint must fail, proving prediction weights alone cannot
	// resume optimization.
	var cp ftrlCheckpoint
	raw, err := ioutil.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &cp); err == nil {
		t.Fatal("inference model file must not parse as a checkpoint")
	}

	// But it still loads correctly for prediction.
	infer := &FTRLLogisticRegression{}
	infer.Init(ftrlTestParams())
	infer.LoadModel(modelPath)
	if !sameModel(trainer.Model, infer.Model) {
		t.Fatal("loaded inference model differs from trained weights")
	}
	for _, s := range dataset.Samples[:5] {
		if infer.Predict(s) != trainer.Predict(s) {
			t.Fatal("inference predictions differ after loading model-only file")
		}
	}
}

// 8. Checkpoint content must carry every piece required for resume:
// optimizer state, hyper-parameters, epoch, next source record, batch size
// and both fingerprints.
func TestCheckpointCarriesFullResumeState(t *testing.T) {
	rand.Seed(23)
	dataset := makeLinearDataSet(13)
	fingerprint := sampleFingerprint(t, dataset)
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")

	trainer := &FTRLLogisticRegression{}
	if err := trainer.ResumeOrTrainBatched(dataset, ftrlTestParams(), 5, cpPath, fingerprint); err != nil {
		t.Fatal(err)
	}
	raw, err := ioutil.ReadFile(cpPath)
	if err != nil {
		t.Fatal(err)
	}
	var cp ftrlCheckpoint
	if err := json.Unmarshal(raw, &cp); err != nil {
		t.Fatalf("checkpoint is not valid json: %v", err)
	}
	if cp.Version != ftrlCheckpointVersion {
		t.Fatalf("version=%d", cp.Version)
	}
	if !sameModel(cp.Model, trainer.Model) {
		t.Fatal("checkpoint missing optimizer state")
	}
	if cp.Params != trainer.Params {
		t.Fatalf("checkpoint params %+v != %+v", cp.Params, trainer.Params)
	}
	if cp.BatchSize != 5 || cp.InputFingerprint != fingerprint || cp.ConfigFingerprint == "" {
		t.Fatal("checkpoint missing batching or fingerprint fields")
	}
	if cp.Epoch != trainer.Params.Steps || cp.NextSampleIndex != 0 ||
		cp.Updates != int64(trainer.Params.Steps*len(dataset.Samples)) {
		t.Fatalf("checkpoint progress wrong: epoch=%d index=%d updates=%d",
			cp.Epoch, cp.NextSampleIndex, cp.Updates)
	}
}

// 9. Resume across an epoch boundary: a run stopped exactly at an epoch
// boundary (dataset length divisible by the batch size) must carry
// epoch+1/sample 0 in its checkpoint, and a fresh receiver resuming from it
// converges to the uninterrupted multi-epoch model.
func TestResumeAtEpochBoundary(t *testing.T) {
	rand.Seed(29)
	params := ftrlTestParams() // steps = 3
	dataset := makeLinearDataSet(16)
	fingerprint := sampleFingerprint(t, dataset)
	batchSize := 16 // one batch per epoch, so every batch end is an epoch boundary

	reference := &FTRLLogisticRegression{}
	reference.Init(params)
	reference.Train(dataset)

	// First "process": stop after epoch 0 by feeding only the confirmed
	// epoch-0 checkpoint, which ResumeOrTrainBatched writes with epoch=1,
	// next=0 after the epoch-boundary rewrite.
	dir := t.TempDir()
	cpPath := filepath.Join(dir, "ftrl.ckpt")
	first := &FTRLLogisticRegression{}
	first.Init(params)
	// Run only the first epoch directly, then checkpoint the boundary.
	for _, s := range dataset.Samples {
		first.updateSample(s)
	}
	cp := &ftrlCheckpoint{
		Version:           ftrlCheckpointVersion,
		Model:             first.Model,
		Params:            first.Params,
		Epoch:             1,
		NextSampleIndex:   0,
		BatchSize:         batchSize,
		Updates:           int64(len(dataset.Samples)),
		InputFingerprint:  fingerprint,
		ConfigFingerprint: FTRLConfigFingerprint(first.Params, batchSize),
	}
	if err := saveCheckpoint(cpPath, cp); err != nil {
		t.Fatal(err)
	}
	reloaded, err := loadCheckpoint(cpPath, fingerprint,
		FTRLConfigFingerprint(first.Params, batchSize))
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Epoch != 1 || reloaded.NextSampleIndex != 0 ||
		reloaded.Updates != int64(len(dataset.Samples)) {
		t.Fatalf("epoch-boundary checkpoint wrong: %+v", reloaded)
	}

	// New process resumes from epoch 1 and finishes epochs 1 and 2.
	resumed := &FTRLLogisticRegression{}
	if err := resumed.ResumeOrTrainBatched(dataset, params, batchSize, cpPath, fingerprint); err != nil {
		t.Fatal(err)
	}
	if !sameModel(reference.Model, resumed.Model) {
		t.Fatal("epoch-boundary resume did not converge to uninterrupted model")
	}
}

// 10. No checkpoint path means the batched entry point degrades to the plain
// Train semantics.
func TestTrainBatchedWithoutPath(t *testing.T) {
	rand.Seed(31)
	dataset := makeLinearDataSet(25)
	reference := &FTRLLogisticRegression{}
	reference.Init(ftrlTestParams())
	reference.Train(dataset)

	a := &FTRLLogisticRegression{}
	a.Init(ftrlTestParams())
	if err := a.TrainBatched(dataset, 7, "", "anything"); err != nil {
		t.Fatal(err)
	}
	if !sameModel(reference.Model, a.Model) {
		t.Fatal("no-checkpoint path diverged from Train")
	}

	if err := a.TrainBatched(dataset, 0, filepath.Join(t.TempDir(), "x"), "fp"); err == nil {
		t.Fatal("expected error for non-positive batch size")
	}
}
