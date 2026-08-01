# Dead ends

Nothing in this directory is on the path to a working port. Each file's conclusion is
already recorded in a document; the code is here so nobody re-runs the experiment to find
out.

These predate this repo's git history — they were carried across from before it was
initialised, which is why they are files rather than reverted commits. New refutations do
not belong here: they belong in a probe under `crates/a8-probe/` that can be re-run, with
the conclusion in its module docstring. See that crate's README for the list.

- `convopt.rs` — conv-as-GEMM (im2col / taps / chunked). All three correct, all
  0.73–0.86x of direct `conv1d`, i.e. slower. See `../../performance/candle-on-metal.md`.
- `dilation.rs` — de-interleaving a dilated conv into a batched dense conv. Exact,
  and slower. Also showed dilation itself is free (d9/d1 = 0.95–1.06x). Same doc.
  This is the file whose 2x-off timings exposed the thermal problem.
- `main.rs` — the original op-coverage probe. Verdict (20/20 Metal ops, GO) is in
  `../coreml-and-op-coverage.md`. Superseded by the real cascades.
- `export_codec.py`, `bench_onnx.py`, `bench_coreml_one.py` — the ONNX/ORT
  evaluation. The export itself worked and was bit-exact; every runtime
  configuration was the slowest option tested. See `../onnx.md`, including
  the four export patches, which are described there in enough detail to redo.
- `a8-standalone-cli.rs` — the original single-engine `a8` binary. Superseded by
  `crates/tts-cli` (`tts speak --engine audio8`), which does the same thing through the
  `tts_core::Engine` trait. Two code paths to the same synthesis is one too many, and
  the segment loop it contained now lives in `crates/a8/src/engine.rs`.
