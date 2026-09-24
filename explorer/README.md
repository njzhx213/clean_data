# clean_data/explorer — the data explorer, rebuilt on H200_profiling_portal/clean_data only

A real copy of `bench_agentic/technique/explorer` (page + builders), pointed at `../` (the clean data) instead of
CLEAN_0905. Nothing here links into `tool_speculation/`.

Serve: `python3 -m http.server 8913 --directory /eic/data/haoxiong_gpu6/bench_agentic/H200_profiling_portal/clean_data/explorer`
→ http://130.207.125.95:8913/index.html   (started 2026-09-24 with nohup; restart with the same command if it is gone)

| card | data | built from |
|---|---|---|
| Tool Analysis Graphs | `data/tag/<ds>/<cell>/<iid>.json`, `index.json`, `census.json`, `TAG_CENSUS.md` | `../<ds>/<mode>/<fw>/traj/<iid>/*tool_calls.jsonl` |
| Timelines | `data/tl/…` | `…/*live_vllm_openai.parquet` |
| Wall times | `data/wall.json` | tag + tl + `../<ds>/<mode>/<fw>/verdicts.tsv` (official verdicts; TB2 = verifier reward) |
| predictor layer of the TAG card | `data/opp/…` | the tag graphs (`build_oppdata.py`) |
| Grid-12 run / speculation lane / hardware rows | absent on purpose | A/B-campaign extracts (`data/run`, `data/spec`, `data/hw`); not profiling data — the page shows "no … yet" |
| Figures / Command logs / Execution graphs / Dependency graphs | embedded in `index.html` | the June 227-task corpus, unchanged from the original page |

Datasets: `pro_250`, `tb2_89` (busy-notice reruns already substituted), `swebench_verified_200`, `swebench_lite_100`
(the last two partial: 143–146 / 72–75 per cell). Cells: `<mode>__<fw>`. 3,341 of the 3,345 clean runs have graphs
(3 hard-killed instruct·langgraph Pro runs have no tool_calls; 1 TB2 run has only bookkeeping calls).

Rebuild everything (after `build_clean.py` refreshes `../`): `build/rebuild_all.sh` (tag → tl → wall → opp; log in `data/build.log`).
Changes vs the campaign copy: Timelines header shows the verdict of the run being viewed (badge) plus the same task's verdict under the other five settings (from `data/wall.json`); `build_tagdata.py` ROOT/DATASETS/SUBSETS point at clean; `build_walldata.py` reads
`verdicts.tsv` for all four datasets instead of grading_snapshot/preds; page DSL labels and headings.
