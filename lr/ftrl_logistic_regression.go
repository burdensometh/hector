package lr

import (
	"bufio"
	"errors"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

type FTRLLogisticRegressionParams struct {
	Alpha, Beta, Lambda1, Lambda2 float64
	Steps                         int
}

type FTRLFeatureWeight struct {
	ni, zi float64
}

func (w *FTRLFeatureWeight) Wi(p FTRLLogisticRegressionParams) float64 {
	wi := 0.0
	if math.Abs(w.zi) > p.Lambda1 {
		wi = (util.Signum(w.zi)*p.Lambda1 - w.zi) / (p.Lambda2 + (p.Beta+math.Sqrt(w.ni))/p.Alpha)
	}
	return wi
}

type FTRLLogisticRegression struct {
	Model  map[int64]FTRLFeatureWeight
	Params FTRLLogisticRegressionParams
}

func (algo *FTRLLogisticRegression) SaveModel(path string) {
	sb := util.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g.ni)
		sb.Write("\t")
		sb.Float(g.zi)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *FTRLLogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		ni, _ := strconv.ParseFloat(tks[1], 64)
		zi, _ := strconv.ParseFloat(tks[2], 64)
		g := FTRLFeatureWeight{ni: ni, zi: zi}
		algo.Model[fid] = g
	}
}

func (algo *FTRLLogisticRegression) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value.Wi(algo.Params) * feature.Value
		}
	}
	return util.Sigmoid(ret)
}

func (algo *FTRLLogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]FTRLFeatureWeight)
	algo.Params.Alpha, _ = strconv.ParseFloat(params["alpha"], 64)
	algo.Params.Lambda1, _ = strconv.ParseFloat(params["lambda1"], 64)
	algo.Params.Lambda2, _ = strconv.ParseFloat(params["lambda2"], 64)
	algo.Params.Beta, _ = strconv.ParseFloat(params["beta"], 64)
	steps, _ := strconv.ParseInt(params["steps"], 10, 32)
	algo.Params.Steps = int(steps)
}

func (algo *FTRLLogisticRegression) Clear() {
	algo.Model = nil
	algo.Model = make(map[int64]FTRLFeatureWeight)
}

// updateSample applies one FTRL-proximal update for a single sample. FTRL is
// an inherently sequential, order-sensitive algorithm, so every code path
// (plain training, batched training and checkpoint resume) must funnel
// through this method in exactly the same order.
func (algo *FTRLLogisticRegression) updateSample(sample *core.Sample) {
	prediction := algo.Predict(sample)
	err := sample.LabelDoubleValue() - prediction
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if !ok {
			model_feature_value = FTRLFeatureWeight{0.0, 0.0}
		}
		zi := model_feature_value.zi
		ni := model_feature_value.ni
		gi := -1 * err * feature.Value
		sigma := (math.Sqrt(ni+gi*gi) - math.Sqrt(ni)) / algo.Params.Alpha
		wi := model_feature_value.Wi(algo.Params)
		zi += gi - sigma*wi
		ni += gi * gi
		algo.Model[feature.Id] = FTRLFeatureWeight{zi: zi, ni: ni}
	}
}

func (algo *FTRLLogisticRegression) Train(dataset *core.DataSet) {
	for step := 0; step < algo.Params.Steps; step++ {
		for _, sample := range dataset.Samples {
			algo.updateSample(sample)
		}
	}
}

// ResumeOrTrainBatched is the process-entry point for resumable FTRL
// training. If checkpointPath does not exist yet, params initialize a fresh
// model and training starts from the first record. If it exists, the
// checkpoint is fully validated (version, input fingerprint, training
// definition fingerprint) and the optimizer state, hyper-parameters, epoch
// and next-record position are restored; only unconfirmed batches are then
// replayed. A validation failure returns an error and leaves the receiver
// untouched, so callers cannot mistake prediction weights for a resumed
// optimization process.
func (algo *FTRLLogisticRegression) ResumeOrTrainBatched(dataset *core.DataSet,
	params map[string]string, batchSize int, checkpointPath string,
	inputFingerprint string) error {

	if batchSize <= 0 {
		return errors.New("batch size must be positive")
	}
	configFingerprint := FTRLConfigFingerprintFromParams(params, batchSize)

	if _, err := os.Stat(checkpointPath); err == nil {
		cp, err := loadCheckpoint(checkpointPath, inputFingerprint, configFingerprint)
		if err != nil {
			return err
		}
		algo.Model = cp.Model
		algo.Params = cp.Params
		log.Printf("ftrl: resumed from %s at epoch=%d next_sample=%d updates=%d",
			checkpointPath, cp.Epoch, cp.NextSampleIndex, cp.Updates)
	} else {
		if !os.IsNotExist(err) {
			return err
		}
		algo.Init(params)
	}
	return algo.trainBatched(dataset, batchSize, checkpointPath, inputFingerprint, configFingerprint)
}

// TrainBatched runs the same epoch/sample loop as Train, but groups samples
// into batches of batchSize and durably checkpoints progress only on
// CONFIRMED batch boundaries: a batch is confirmed only after every one of
// its samples has been updated and the checkpoint has been atomically
// written. On restart the checkpoint is validated against the current input
// and training definition, then only the not-yet-confirmed batches are
// replayed, producing exactly the parameters an uninterrupted run would have
// reached.
//
// checkpointPath == "" means no checkpointing: behavior is identical to
// Train. inputFingerprint must identify the exact training source content
// (e.g. its SHA-256); it is embedded in every checkpoint and re-checked on
// resume.
func (algo *FTRLLogisticRegression) TrainBatched(dataset *core.DataSet,
	batchSize int, checkpointPath string, inputFingerprint string) error {

	if batchSize <= 0 {
		return errors.New("batch size must be positive")
	}
	if checkpointPath == "" {
		algo.Train(dataset)
		return nil
	}
	return algo.trainBatched(dataset, batchSize, checkpointPath,
		inputFingerprint, ftrlConfigFingerprint(algo.Params, batchSize))
}

// trainBatched contains the shared batch loop. The config fingerprint is
// passed in so that both the fresh-start path (fingerprint of current
// Params) and the resume path (fingerprint already validated against the
// parameter map) write exactly what was verified on load.
func (algo *FTRLLogisticRegression) trainBatched(dataset *core.DataSet,
	batchSize int, checkpointPath string, inputFingerprint, configFingerprint string) error {

	epoch, nextIndex, updates := 0, 0, int64(0)
	if _, err := os.Stat(checkpointPath); err == nil {
		cp, err := loadCheckpoint(checkpointPath, inputFingerprint, configFingerprint)
		if err != nil {
			// Refuse to continue: a stale or incompatible checkpoint must
			// never be silently treated as a fresh start or as a valid
			// optimization resume.
			return err
		}
		algo.Model = cp.Model
		algo.Params = cp.Params
		epoch, nextIndex, updates = cp.Epoch, cp.NextSampleIndex, cp.Updates
		log.Printf("ftrl: resumed from %s at epoch=%d next_sample=%d updates=%d",
			checkpointPath, epoch, nextIndex, updates)
	} else if !os.IsNotExist(err) {
		return err
	}

	n := len(dataset.Samples)
	for ; epoch < algo.Params.Steps; epoch++ {
		// Defensive: a checkpoint must never point past the end of the
		// dataset it was fingerprinted against.
		if nextIndex > n {
			return errors.New("checkpoint position is beyond the dataset size")
		}
		for nextIndex < n {
			end := nextIndex + batchSize
			if end > n {
				end = n
			}
			// Apply the whole batch first; no state is durable until every
			// sample below has been consumed, so a crash inside this loop
			// simply replays the entire batch on restart.
			for i := nextIndex; i < end; i++ {
				algo.updateSample(dataset.Samples[i])
			}
			updates += int64(end - nextIndex)
			nextIndex = end

			cp := &ftrlCheckpoint{
				Version:           ftrlCheckpointVersion,
				Model:             algo.Model,
				Params:            algo.Params,
				Epoch:             epoch,
				NextSampleIndex:   nextIndex,
				BatchSize:         batchSize,
				Updates:           updates,
				InputFingerprint:  inputFingerprint,
				ConfigFingerprint: configFingerprint,
			}
			// On failure the previous complete checkpoint file is retained;
			// the in-memory model is still correct for this process, but the
			// durable resume point rolls back to the last confirmed batch.
			if err := saveCheckpoint(checkpointPath, cp); err != nil {
				return err
			}
		}
		// Epoch boundary confirmed: next run starts at the first sample of
		// the next epoch. The checkpoint written for the final batch already
		// carries NextSampleIndex == n; rewrite it with the advanced epoch so
		// the resume position is unambiguous. This also runs after the final
		// epoch, marking the run complete (Epoch == Steps, index 0); resuming
		// a completed run is then an idempotent no-op.
		nextIndex = 0
		cp := &ftrlCheckpoint{
			Version:           ftrlCheckpointVersion,
			Model:             algo.Model,
			Params:            algo.Params,
			Epoch:             epoch + 1,
			NextSampleIndex:   0,
			BatchSize:         batchSize,
			Updates:           updates,
			InputFingerprint:  inputFingerprint,
			ConfigFingerprint: configFingerprint,
		}
		if err := saveCheckpoint(checkpointPath, cp); err != nil {
			return err
		}
	}
	return nil
}
