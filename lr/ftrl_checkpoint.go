package lr

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
)

// ftrlCheckpointVersion is bumped whenever the on-disk checkpoint layout or
// the FTRL update definition changes in an incompatible way.
const ftrlCheckpointVersion = 1

// MarshalJSON / UnmarshalJSON expose the otherwise unexported optimizer
// accumulators (ni, zi) in the checkpoint envelope. The text model file
// written by SaveModel is unaffected and keeps serving inference-only use.
func (w FTRLFeatureWeight) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Ni float64 `json:"ni"`
		Zi float64 `json:"zi"`
	}{Ni: w.ni, Zi: w.zi})
}

func (w *FTRLFeatureWeight) UnmarshalJSON(data []byte) error {
	v := struct {
		Ni float64 `json:"ni"`
		Zi float64 `json:"zi"`
	}{}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	w.ni, w.zi = v.Ni, v.Zi
	return nil
}

// ftrlCheckpoint is the full state required to RESUME FTRL optimization in a
// new process. It is strictly more than the prediction model: besides the
// per-feature optimizer state (ni, zi) it carries the hyper-parameters, the
// current epoch, the position of the next source record to consume, and
// fingerprints that bind the checkpoint to one concrete training definition
// and one concrete input file.
type ftrlCheckpoint struct {
	Version int `json:"version"`

	// Optimizer state: one (ni, zi) pair per feature id. This is the same
	// information written by SaveModel, plus everything below which a bare
	// model file does not have.
	Model map[int64]FTRLFeatureWeight `json:"model"`

	// Hyper-parameters in effect when the state was produced. Resuming with
	// different values would silently change the optimization trajectory, so
	// they are fingerprinted and verified on load.
	Params FTRLLogisticRegressionParams `json:"params"`

	// Epoch is the outer loop counter (0-based). NextSampleIndex is the
	// position inside dataset.Samples of the first record NOT yet consumed.
	// Together they identify the next input record.
	Epoch           int `json:"epoch"`
	NextSampleIndex int `json:"next_sample_index"`

	// Number of samples per confirmed batch. Checkpoints are only written on
	// batch boundaries, so NextSampleIndex is always a multiple of BatchSize
	// (or equals the dataset length at an epoch boundary).
	BatchSize int `json:"batch_size"`

	// Number of optimizer updates already applied, i.e. samples consumed
	// across all epochs. Handy for observability and sanity checks.
	Updates int64 `json:"updates"`

	// InputFingerprint hashes the raw bytes of the training source file.
	// ConfigFingerprint hashes the training definition (method name,
	// hyper-parameters and batch size). A checkpoint is rejected when either
	// fingerprint changes between runs.
	InputFingerprint  string `json:"input_fingerprint"`
	ConfigFingerprint string `json:"config_fingerprint"`
}

// FingerprintReader returns the hex SHA-256 of everything readable from r.
func FingerprintReader(r io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// FingerprintFile returns the hex SHA-256 of the file at path.
func FingerprintFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return FingerprintReader(f)
}

// FTRLConfigFingerprint binds a checkpoint to a training definition: the FTRL
// method name, every hyper-parameter and the batch size.
func FTRLConfigFingerprint(p FTRLLogisticRegressionParams, batchSize int) string {
	h := sha256.New()
	fmt.Fprintf(h, "method=ftrl\n")
	fmt.Fprintf(h, "alpha=%g\nbeta=%g\nlambda1=%g\nlambda2=%g\nsteps=%d\nbatch_size=%d\n",
		p.Alpha, p.Beta, p.Lambda1, p.Lambda2, p.Steps, batchSize)
	return hex.EncodeToString(h.Sum(nil))
}

// ftrlConfigFingerprint is the package-internal alias used by the training
// loop; callers outside the package use FTRLConfigFingerprint.
func ftrlConfigFingerprint(p FTRLLogisticRegressionParams, batchSize int) string {
	return FTRLConfigFingerprint(p, batchSize)
}

// saveCheckpoint atomically writes cp to path. The data is first written and
// fsynced to a temporary file in the SAME directory, then renamed over path.
// If anything fails before the rename, the previous file at path is left
// untouched, so the last complete checkpoint always remains available.
func saveCheckpoint(path string, cp *ftrlCheckpoint) error {
	dir := filepath.Dir(path)
	if dir == "" {
		dir = "."
	}
	// TempFile in the same directory guarantees os.Rename is atomic (same
	// filesystem) and never crosses a mount point.
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create checkpoint temp file: %w", err)
	}
	tmpName := tmp.Name()
	cleanup := func() {
		tmp.Close()
		os.Remove(tmpName)
	}

	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cp); err != nil {
		cleanup()
		return fmt.Errorf("encode checkpoint: %w", err)
	}
	// Flush contents to disk before the rename so a crash cannot leave an
	// empty or partial file under the final name.
	if err := tmp.Sync(); err != nil {
		cleanup()
		return fmt.Errorf("sync checkpoint: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("close checkpoint: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("commit checkpoint: %w", err)
	}
	return nil
}

// LoadFTRLCheckpoint reads and validates a checkpoint from path. It returns
// an error (without touching any model state) when the file is missing,
// malformed, from an incompatible version, or produced for a different input
// file or training definition.
func LoadFTRLCheckpoint(path string, inputFingerprint, configFingerprint string) (*ftrlCheckpoint, error) {
	return loadCheckpoint(path, inputFingerprint, configFingerprint)
}

// FTRLConfigFingerprintFromParams builds the training-definition fingerprint
// directly from the string parameter map used by the CLI/runner, so the
// resume check does not depend on how Init parses defaults.
func FTRLConfigFingerprintFromParams(params map[string]string, batchSize int) string {
	p := FTRLLogisticRegressionParams{}
	p.Alpha, _ = strconv.ParseFloat(params["alpha"], 64)
	p.Beta, _ = strconv.ParseFloat(params["beta"], 64)
	p.Lambda1, _ = strconv.ParseFloat(params["lambda1"], 64)
	p.Lambda2, _ = strconv.ParseFloat(params["lambda2"], 64)
	steps, _ := strconv.ParseInt(params["steps"], 10, 32)
	p.Steps = int(steps)
	return FTRLConfigFingerprint(p, batchSize)
}

// loadCheckpoint reads and validates a checkpoint from path.
func loadCheckpoint(path string, inputFingerprint, configFingerprint string) (*ftrlCheckpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open checkpoint: %w", err)
	}
	defer f.Close()

	cp := &ftrlCheckpoint{}
	if err := json.NewDecoder(f).Decode(cp); err != nil {
		return nil, fmt.Errorf("decode checkpoint %s: %w", path, err)
	}
	if cp.Version != ftrlCheckpointVersion {
		return nil, fmt.Errorf("incompatible checkpoint version %d, expected %d",
			cp.Version, ftrlCheckpointVersion)
	}
	if cp.Model == nil {
		return nil, fmt.Errorf("checkpoint %s carries no optimizer state", path)
	}
	if cp.InputFingerprint != inputFingerprint {
		return nil, fmt.Errorf("checkpoint %s rejected: input data changed "+
			"(checkpoint fingerprint %s, current input %s)",
			path, cp.InputFingerprint, inputFingerprint)
	}
	if cp.ConfigFingerprint != configFingerprint {
		return nil, fmt.Errorf("checkpoint %s rejected: training definition changed "+
			"(checkpoint fingerprint %s, current definition %s)",
			path, cp.ConfigFingerprint, configFingerprint)
	}
	if cp.Epoch < 0 || cp.NextSampleIndex < 0 || cp.BatchSize <= 0 {
		return nil, fmt.Errorf("checkpoint %s has corrupted progress fields", path)
	}
	return cp, nil
}
