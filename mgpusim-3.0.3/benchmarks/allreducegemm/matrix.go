package allreducegemm

// matrix is a row-major Width x Height matrix: element (row, col) lives at
// Data[row*Width+col].
type matrix struct {
	Data          []float32
	Width, Height uint32
}

func newMatrix(width, height uint32) *matrix {
	return &matrix{
		Data:   make([]float32, width*height),
		Width:  width,
		Height: height,
	}
}

// columnSlice extracts columns [colStart, colEnd) from every row, returning
// a compact (colEnd-colStart) x Height matrix. Used to give each GPU its own
// K-dimension slice of matrix A.
func (m *matrix) columnSlice(colStart, colEnd uint32) *matrix {
	sliceWidth := colEnd - colStart
	s := newMatrix(sliceWidth, m.Height)

	for row := uint32(0); row < m.Height; row++ {
		srcOffset := row*m.Width + colStart
		dstOffset := row * sliceWidth
		copy(s.Data[dstOffset:dstOffset+sliceWidth],
			m.Data[srcOffset:srcOffset+sliceWidth])
	}

	return s
}

// rowSlice extracts rows [rowStart, rowEnd), returning a compact
// Width x (rowEnd-rowStart) matrix. Used to give each GPU its own
// K-dimension slice of matrix B. Since rows are contiguous in row-major
// layout, this is a single contiguous copy.
func (m *matrix) rowSlice(rowStart, rowEnd uint32) *matrix {
	s := newMatrix(m.Width, rowEnd-rowStart)
	copy(s.Data, m.Data[rowStart*m.Width:rowEnd*m.Width])
	return s
}
