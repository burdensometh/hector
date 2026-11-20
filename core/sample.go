package core

import (
	"github.com/xlvector/hector/util"
)

/*
Sample - for classification
Here, label should be int value started from 0

Weight is the optional representative weight of this row. Prefer supplying
weights through DataSet.Weights (or DataSet.SetWeights / LoadWeights), which
validates the length and rejects an all-zero vector. Sample.Weight is a
convenience for callers that build datasets by hand: a dataset counts as
weighted when it carries a DataSet.Weights slice or when at least one sample
has a non-zero Weight. Because NewSample leaves Weight at 0, an all-zero
per-sample vector is indistinguishable from "no weights" and is silently
treated as unweighted; use DataSet.SetWeights when you need all-zero input to
be rejected. A zero weight row is excluded from learning and from weighted
metrics.
*/
type Sample struct {
	Features   []Feature
	Label      int
	Prediction float64
	Weight     float64
}

func NewSample() *Sample {
	ret := Sample{}
	ret.Features = []Feature{}
	ret.Label = 0
	ret.Prediction = 0.0
	ret.Weight = 0.0
	return &ret
}

func (s *Sample) Clone() *Sample {
	ret := NewSample()
	ret.Label = s.Label
	ret.Prediction = s.Prediction
	ret.Weight = s.Weight
	for _, feature := range s.Features {
		clone_feature := Feature{feature.Id, feature.Value}
		ret.Features = append(ret.Features, clone_feature)
	}

	return ret
}

func (s *Sample) ToString(includePrediction bool) []byte {
	sb := util.StringBuilder{}
	sb.Int(s.Label)
	sb.Write(" ")
	if includePrediction {
		sb.Float(s.Prediction)
		sb.Write(" ")
	}
	for _, feature := range s.Features {
		sb.Int64(feature.Id)
		sb.Write(":")
		sb.Float(feature.Value)
		sb.Write(" ")
	}
	return sb.Bytes()
}

func (s *Sample) LabelDoubleValue() float64 {
	if s.Label > 0 {
		return 1.0
	} else {
		return 0.0
	}
}

func (s *Sample) AddFeature(f Feature) {
	s.Features = append(s.Features, f)
}

/* RawSample */
type RawSample struct {
	Label      int
	Prediction float64
	Features   map[string]string
}

func NewRawSample() *RawSample {
	ret := RawSample{}
	ret.Features = make(map[string]string)
	ret.Label = 0
	ret.Prediction = 0.0
	return &ret
}

func (s *RawSample) GetFeatureValue(key string) string {
	value, ok := s.Features[key]
	if ok {
		return value
	} else {
		return "nil"
	}
}

/* MapBasedSample */
type MapBasedSample struct {
	Label      int
	Prediction float64
	Features   map[int64]float64
}

func (s *MapBasedSample) LabelDoubleValue() float64 {
	return float64(s.Label)
}

func (s *Sample) ToMapBasedSample() *MapBasedSample {
	ret := MapBasedSample{}
	ret.Features = make(map[int64]float64)
	ret.Label = s.Label
	ret.Prediction = s.Prediction
	for _, feature := range s.Features {
		ret.Features[feature.Id] = feature.Value
	}
	return &ret
}

func (s *Sample) GetFeatureVector() *Vector {
	ret := NewVector()
	for _, f := range s.Features {
		ret.SetValue(f.Id, f.Value)
	}
	return ret
}

/*
RealSample
Real valued samples for regression
*/
type RealSample struct {
	Features   []Feature
	Prediction float64
	Value      float64
}

func NewRealSample() *RealSample {
	ret := RealSample{}
	ret.Features = []Feature{}
	ret.Value = 0.0
	ret.Prediction = 0.0
	return &ret
}

func (rs *RealSample) GetFeatureVector() *Vector {
	ret := NewVector()
	for _, f := range rs.Features {
		ret.SetValue(f.Id, f.Value)
	}
	return ret
}

func (s *RealSample) AddFeature(f Feature) {
	s.Features = append(s.Features, f)
}
