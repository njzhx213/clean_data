# clean_data — final clean H200 profiling data (assembled 2026-09-17, explorer added 2026-09-24)

One self-contained **real copy** (no symlinks into the shared tree, except `hw_raw_samples`) of every
run that counts. 3,345 runs; 580 MB of traces + 130 MB of explorer extracts. GitHub mirror: https://github.com/njzhx213/clean_data Built by
`bench_agentic/architecture/6_measurement/clean_export/build_clean.py` (also copied here as `tools/build_clean.py`) (idempotent; `--dry-run` prints the table only).

| folder | what | source tree (read-only) |
|---|---|---|
| `pro_250/<mode>/<fw>/` | SWE-bench-Pro 250 × {thinking, instruct, coder} × {hermes, langgraph}; official prompt, `--network none`, git strip | `profile_all/results_nonet` |
| `tb2_89/<mode>/<fw>/` | Terminal-Bench-2 89, tmux interactive arm | `results_tb2t`, with 90 runs taken from `results_tb2v` (see below) |
| `swebench_verified_200/…`, `swebench_lite_100/…` | Pro protocol on Verified 200 / Lite 100; lane was stopped 09-07 17:08, so 143–146 / 200 and 72–75 / 100 per cell | `results_swev`, `results_swel` |
| `task_sets/` | the exact task records fed to the agents | `bench_agentic/subsets` |
| `grading/` | official-harness result files the verdicts come from | `bench_agentic/grading/{full,swe}` |
| `hw_raw_samples` | symlink: cgroup + nvidia-smi samples (28 GB, not copied, not in the GitHub mirror) | `profile_all/hw` |
| `explorer/` | the data explorer (TAG graphs, timelines, wall times, predictor layer) rebuilt on this folder only; serve with `python3 -m http.server 8913 --directory clean_data/explorer` → http://130.207.125.95:8913/index.html; `explorer/build/rebuild_all.sh` regenerates `explorer/data/` | `bench_agentic/technique/explorer` |

Per cell the source layout is kept, so existing loaders work by pointing at `clean_data/<dataset>`:
`logs/<iid>.log` (`.tmux.log` for TB2; receipts), `traj/<iid>/` (`*tool_calls.jsonl`,
`*live_vllm_openai.parquet`, langgraph transcripts/node events), `traj/<iid>_traj.jsonl` (hermes conversation),
`stats/<iid>.json`, plus two files written by the builder:
`preds_w0_clean.jsonl` (ONE line per task = the patch the official harness graded; TB2: reward from the current log)
and `verdicts.tsv`.

## What "clean" means here

1. **Only current runs.** Anything the campaign archived with a `__suffix` is a superseded or buggy run and is
   not copied; its rerun (plain name) is. 326 archived names were left out, listed in `excluded_archived.tsv`
   (`__tmp_noexec`, `__loop600`, `__bare_prompt`, `__roottmpfs`, `__noenv`, `__prewait_bug`, `__netguard_verify`, `__infra_cut`, …).
   Runs archived for a pipeline bug but already judged resolved were restored under the plain name during the campaign (09-07 rule) and are therefore included.
2. **Reruns replace the runs they were made for.** TB2: all 90 (cell, task) pairs of `tb2v_tasks.tsv`
   (hermes runs hit by the quiet-long-command trap + the same tasks on langgraph, rerun with `TMUX_BUSY_NOTICE=1`)
   use the rerun instead of the main-table run. `replaced_runs.tsv` gives old and new verdict per pair.
   Effect: hermes +8 (44→46, 32→34, 27→31), langgraph unchanged, total 203→211 / 534.
   The shared tree is untouched; the original main-table runs stay in `results_tb2t`.
3. **Verdicts are official.** Pro: `swe_bench_pro_eval.py` results; Verified/Lite: swebench 5.0.2 `report.json`,
   empty patch = unresolved (official protocol); TB2: the verifier reward in the run's own log.

## Accuracy (from `MANIFEST.tsv`; also in `accuracy_summary.md`)

Pro matches the 09-07 final table exactly (685 / 1500). Verified/Lite have more graded runs than the
09-07 17:52 final tables because the grading loop finished afterwards.

| dataset | thinking·hermes | thinking·lg | instruct·hermes | instruct·lg | coder·hermes | coder·lg |
|---|---|---|---|---|---|---|
| pro_250 | 139/250 55.6% | 136/250 54.4% | 122/250 48.8% | 92/250 36.8% | 91/250 36.4% | 105/250 42.0% |
| tb2_89 (reruns substituted) | 46/89 51.7% | 47/89 52.8% | 34/89 38.2% | 29/89 32.6% | 31/89 34.8% | 24/89 27.0% |
| verified_200 (partial) | 113/146 77.4% | 95/145 65.5% | 91/146 62.3% | 50/146 34.2% | 93/142 65.5% | 96/143 67.1% |
| lite_100 (partial) | 49/75 65.3% | 30/64 46.9% | 27/65 41.5% | 19/68 27.9% | 34/71 47.9% | 34/70 48.6% |

## What takes the space

| kind | size | note |
|---|---|---|
| hermes conversations `traj/<iid>_traj.jsonl` | 143 MB | the full transcript the agent saw (system prompt + every tool output) |
| `*tool_calls.jsonl` | 112 MB | every tool call: command, start/end on the gpu6 monotonic clock |
| `*live_vllm_openai.parquet` | 96 MB | every LLM request: tokens, timing, vLLM/GPU counters (100+ columns) |
| langgraph `*llm_transcripts.jsonl` / `node_events.jsonl` | 69 MB | the langgraph equivalent of the hermes transcript |
| `logs/*.log` | 45 MB | per-run receipts (anti-cheat, network block, prompt protocol, verdict) |
| `explorer/data/` | 130 MB | derived: 3,341 TAG graphs + timelines + opp predictor layer + wall table; rebuildable |

`explorer/data/opp/.tagcache.jsonl` (81 MB cache) and the run logs are git-ignored.

## Redaction

The Terminal-Bench-2 task `sanitize-git-repo` plants fake credentials in its repository (the task is to scrub them), and the
agents echoed them into commands and transcripts. GitHub secret scanning flagged one. In this copy every `hf_…`, `ghp_…` and
`AKIA…` string in that task's 22 files (tool_calls, transcripts, logs, stats, TAG graphs, all six cells) is replaced by
`*_REDACTED_BENCHMARK_FIXTURE`; the source tree under `profile_all` is untouched. No other file matched any credential shape.

## Known gaps (recorded per run in `MANIFEST.tsv`)

- 3 `pro_250/instruct/langgraph` runs (flipt-967855b4, vuls-bff6b755, teleport-8302d467) were hard-killed at the 3 h timeout: log and verdict only, no tool_calls/parquet.
- 30 Verified/Lite runs finished but were never graded (blank verdict): 28 in Lite, 2 in Verified.
- Verified/Lite are incomplete task sets; runs in flight at the stop were archived `__infra_cut` and are excluded.
- Old campaigns (`data_1/2/3`: bare prompt + networked containers) are NOT here; they remain valid only for hardware/latency profiles.
- Byte-equality with the source was spot-checked on 60 random runs (230 files, 0 mismatches).

`explorer/` — the data explorer rebuilt on this folder only (TAG graphs, timelines, wall times for all four datasets), served on :8913; see `explorer/README.md`.
