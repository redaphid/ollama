package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"io/fs"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/x448/float16"
)

func parseTorch(fsys fs.FS, replacer *strings.Replacer, ps ...string) ([]Tensor, error) {
	var ts []Tensor
	for _, p := range ps {
		pt, err := pytorch.Load(p)
		if err != nil {
			return nil, err
		}

		for _, k := range pt.(*types.Dict).Keys() {
			t := pt.(*types.Dict).MustGet(k)

			var shape []uint64
			for dim := range t.(*pytorch.Tensor).Size {
				shape = append(shape, uint64(dim))
			}

			ts = append(ts, torch{
				storage: t.(*pytorch.Tensor).Source,
				tensorBase: &tensorBase{
					name:  replacer.Replace(k.(string)),
					shape: shape,
				},
			})
		}
	}

	return ts, nil
}

type torch struct {
	storage pytorch.StorageInterface
	*tensorBase
}

func (t torch) Clone() Tensor {
	return torch{
		storage: t.storage,
		tensorBase: &tensorBase{
			name:     t.name,
			shape:    t.shape,
			repacker: t.repacker,
		},
	}
}

func (pt torch) WriteTo(w io.Writer) (int64, error) {
	var f32s []float32
	switch s := pt.storage.(type) {
	case *pytorch.FloatStorage:
		f32s = s.Data
	case *pytorch.HalfStorage:
		f32s = s.Data
	case *pytorch.BFloat16Storage:
		f32s = s.Data
	case *pytorch.DoubleStorage:
		f32s = make([]float32, len(s.Data))
		for i, v := range s.Data {
			f32s[i] = float32(v)
		}
	default:
		return 0, fmt.Errorf("unsupported pytorch storage type: %T", pt.storage)
	}

	if pt.repacker != nil {
		var err error
		f32s, err = pt.repacker(pt.Name(), f32s, pt.Shape())
		if err != nil {
			return 0, err
		}
	}

	switch pt.Kind() {
	case tensorKindFP32:
		return int64(len(f32s) * 4), binary.Write(w, binary.LittleEndian, f32s)
	case tensorKindFP16:
		u16s := make([]uint16, len(f32s))
		for i, v := range f32s {
			u16s[i] = float16.Fromfloat32(v).Bits()
		}
		return int64(len(u16s) * 2), binary.Write(w, binary.LittleEndian, u16s)
	default:
		return 0, fmt.Errorf("unsupported output tensor kind: %d", pt.Kind())
	}
}
