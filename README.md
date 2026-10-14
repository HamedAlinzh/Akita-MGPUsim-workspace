# Akita / MGPUSim Workspace

A research workspace for multi-GPU experiments on [MGPUSim](https://github.com/sarchlab/mgpusim),
which runs on the [Akita](https://github.com/sarchlab/akita) simulation engine.

## Layout

| Folder | Contents | In git? |
|---|---|---|
| `simulator/akita-3.0.0/` | Akita discrete-event simulation engine (Go module) | yes |
| `simulator/mgpusim-3.0.3/` | MGPUSim GPU simulator, plus local additions (see below) | yes |
| `scripts/` | Experiment drivers and analysis scripts | yes |
| `results/` | Raw simulation output, one subfolder per experiment | no (local only) |
| `diagrams/` | Plots and figures made from `results/` | yes |
| `reports/` | Standalone HTML write-ups | yes |

`go.work` (at the root) points both modules at the local copies, so a change in Akita is
picked up by MGPUSim right away.

## Local additions to MGPUSim

- `benchmarks/mccl/allreduce_naive.go`: a naive (gather, reduce, broadcast) All-Reduce,
  added alongside the existing Ring All-Reduce.
- `benchmarks/allreducegemm/` and `samples/allreducegemm/`: a distributed GEMM that splits
  the K dimension across GPUs and sums the partial products with All-Reduce.
- `samples/runner`: new `-l1-cache-size` and `-dram-size` flags. This also fixes
  `WithDRAMSize()`, which the DRAM controller used to ignore.

## Quick start

```bash
# Build the simulator
(cd simulator/mgpusim-3.0.3 && go build ./...)

# Run the distributed GEMM sample. MGPUSim writes metrics.csv to the current
# directory, so run it from results/ to keep the source tree clean.
mkdir -p results/allreducegemm
(cd results/allreducegemm && go run github.com/sarchlab/mgpusim/v3/samples/allreducegemm \
    -timing -gpus=1,2 -m=256 -k=256 -n=256 -algorithm=ring)

# Sweep GEMM work-group tile sizes (writes to results/gemm_tile_tuning/)
python3 scripts/tune_gemm_tile.py --tiles 4,8,16 --gpus 2 --algorithm ring
```

## Reports

- `reports/allreduce-gemm-ring-vs-naive.html`: Ring vs. naive All-Reduce for distributed GEMM
- `reports/gemm-tile-tuning.html`: GEMM work-group tile tuning results
- `reports/gemm-tile-tuning-validity.html`: corrections to the tile-tuning experiment and
  caveats about how far the simulator can be trusted
- `reports/gemm-tiling-concepts-fa.html`: background on tiling and the memory hierarchy in GEMM (Persian)
