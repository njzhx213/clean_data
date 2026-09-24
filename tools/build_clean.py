#!/usr/bin/env python3
"""Assemble H200_profiling_portal/clean_data/ = ONE self-contained copy of the final clean profiling data.

Sources (read-only, shared tree): tool_speculation/warmstack/profile_all/results_{nonet,tb2t,tb2v,swev,swel}
Rules
  * only CURRENT runs are copied: anything renamed with an archive suffix (__loop600, __roottmpfs,
    __prewait_bug, ...) is a superseded/buggy run and is left out -- its rerun (plain name) is what lands here;
  * TB2: for every (cell, task) in tb2v_tasks.tsv whose busy-notice rerun reached a verdict, the rerun
    (results_tb2v) REPLACES the main-table run hit by the quiet-long-command trap (user 09-17);
  * per cell the layout of the source is kept (logs/, traj/<iid>/, traj/<iid>_traj.jsonl, stats/<iid>.json)
    so existing loaders work by pointing at clean/<dataset>; predictions are collapsed to ONE line per task
    in preds_w0_clean.jsonl (= the patch the official harness graded; TB2: reward from the current log);
  * verdicts.tsv per cell + MANIFEST.tsv / replaced_runs.tsv / excluded_archived.tsv at the top.
Idempotent: rebuilds each dataset dir from scratch.   Usage: build_clean.py [--dry-run] [dataset ...]
"""
import glob, json, os, re, shutil, sys, time
P = "/eic/data/haoxiong_gpu6/tool_speculation/warmstack/profile_all"
B = "/eic/data/haoxiong_gpu6/bench_agentic"
OUT = f"{B}/H200_profiling_portal/clean_data"
GF = f"{B}/grading/full"; GS = f"{B}/grading/swe"
SNAP = f"{B}/H200_profiling_portal/CLEAN_0905_official_prompt_nonet"
DRY = "--dry-run" in sys.argv
ARCH = re.compile(r"__[A-Za-z0-9_]+$")          # archive suffix (instance ids end in -<sha>/-<n>, never in __word)
MODES = ["thinking", "instruct", "coder"]; FWS = ["hermes", "langgraph"]
DATASETS = {  # name -> (source tree, log extension, kind)
    "pro_250":               ("results_nonet", ".log", "pro"),
    "tb2_89":                ("results_tb2t", ".tmux.log", "tb2"),
    "swebench_verified_200": ("results_swev", ".log", "swev"),
    "swebench_lite_100":     ("results_swel", ".log", "swel"),
}
SWE_DONE = ("completed=", "patch captured in", "loop-break")

def tb2_verdict(log):
    try: t = open(log, errors="replace").read()
    except OSError: return None
    m = re.findall(r"tb2 verified.*?(?:'reward': |-> )([0-9.]+)", t)
    return float(m[-1]) if m else None

def last_pred(tree, mode, fw, iid):
    hit = None
    for pf in sorted(glob.glob(f"{P}/{tree}/{mode}/{fw}/preds_w*.jsonl"), key=os.path.getmtime):
        for l in open(pf, errors="replace"):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("instance_id") == iid: hit = r
    return hit

def cp(src, dst):
    if DRY: return
    if os.path.isdir(src): shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)

manifest, replaced, excluded = [], [], []
tb2v_pairs = {tuple(l.rstrip("\n").split("\t")) for l in open(f"{P}/tb2v_tasks.tsv") if l.strip()}
want = [a for a in sys.argv[1:] if a in DATASETS] or list(DATASETS)
t00 = time.time()
for ds in want:
    tree, ext, kind = DATASETS[ds]
    if not DRY and os.path.isdir(f"{OUT}/{ds}"): shutil.rmtree(f"{OUT}/{ds}")
    for mode in MODES:
        for fw in FWS:
            cell = f"{mode}_{fw}"; src = f"{P}/{tree}/{mode}/{fw}"
            if not os.path.isdir(src): continue
            dst = f"{OUT}/{ds}/{mode}/{fw}"
            names = [n for n in os.listdir(f"{src}/traj") if os.path.isdir(f"{src}/traj/{n}")]
            logs = {n[:-len(ext)] for n in os.listdir(f"{src}/logs") if n.endswith(ext)}
            for n in sorted(set(names) | logs):
                if ARCH.search(n): excluded.append((ds, cell, n, tree))
            cur = sorted(n for n in (set(names) | logs) if not ARCH.search(n))
            # graded verdicts / predictions
            if kind == "pro":
                ev = f"{GF}/eval_{cell}_v2/eval_results.json"
                ev = ev if os.path.exists(ev) else f"{SNAP}/grading_snapshot/eval_results_{cell}.json"
                verd = json.load(open(ev))
                pj = f"{GF}/predictions_{cell}_v2.json"
                pj = pj if os.path.exists(pj) else f"{SNAP}/grading_snapshot/predictions_{cell}.json"
                graded = {p["instance_id"]: p for p in json.load(open(pj))}
            elif kind in ("swev", "swel"):
                graded = {}
                gp = f"{GS}/predictions_{cell}.jsonl"
                if not os.path.exists(gp): gp = f"{SNAP}/grading_swe/predictions_{cell}.jsonl"
                for l in open(gp, errors="replace"):
                    try: r = json.loads(l)
                    except Exception: continue
                    graded[r["instance_id"]] = {"instance_id": r["instance_id"], "patch": r.get("model_patch", ""), "prefix": r.get("model_name_or_path", fw)}
            preds, vrows = [], []
            for iid in cur:
                s_tree, s_dir, note = tree, src, ""
                if kind == "tb2" and (cell, iid) in tb2v_pairs:
                    vlog = f"{P}/results_tb2v/{mode}/{fw}/logs/{iid}{ext}"
                    if tb2_verdict(vlog) is not None:
                        old = tb2_verdict(f"{src}/logs/{iid}{ext}")
                        s_tree, s_dir = "results_tb2v", f"{P}/results_tb2v/{mode}/{fw}"
                        note = "busy-notice rerun replaces main-table run (quiet-long-command trap)"
                        replaced.append((ds, cell, iid, f"{tree}/{mode}/{fw}", old, "results_tb2v", tb2_verdict(vlog), note))
                log = f"{s_dir}/logs/{iid}{ext}"
                has_log = os.path.exists(log); has_traj = os.path.isdir(f"{s_dir}/traj/{iid}")
                if has_log: cp(log, f"{dst}/logs/{iid}{ext}")
                if has_traj: cp(f"{s_dir}/traj/{iid}", f"{dst}/traj/{iid}")
                for extra in (f"traj/{iid}_traj.jsonl", f"stats/{iid}.json"):
                    if os.path.exists(f"{s_dir}/{extra}"): cp(f"{s_dir}/{extra}", f"{dst}/{extra}")
                tc = bool(glob.glob(f"{s_dir}/traj/{iid}/*tool_calls.jsonl")); pq = bool(glob.glob(f"{s_dir}/traj/{iid}/*live_vllm_openai.parquet"))
                # verdict + prediction
                if kind == "pro":
                    v = verd.get(iid); p = graded.get(iid) or last_pred(s_tree, mode, fw, iid)
                elif kind == "tb2":
                    v = tb2_verdict(log); p = last_pred(s_tree, mode, fw, iid) or {"instance_id": iid, "dataset": "tb2"}
                    p = dict(p); p["reward"] = v
                else:
                    rep = f"{GS}/out_{kind}/logs/run_evaluation/{kind}_{cell}/{cell}/{iid}/report.json"
                    v = None
                    if os.path.exists(rep):
                        try: v = bool(json.load(open(rep))[iid].get("resolved"))
                        except Exception: v = None
                    t = open(log, errors="replace").read() if has_log else ""
                    done = any(k in t for k in SWE_DONE)
                    if v is None and re.search(r"patch captured in [^:]*: 0 bytes|patch captured \(0 bytes\)", t): v = False
                    p = graded.get(iid) or last_pred(s_tree, mode, fw, iid)
                    if not done: note = (note + " " if note else "") + "log has no completion marker"
                if p: preds.append(p)
                patch_bytes = len((p or {}).get("patch") or "") if kind != "tb2" else ""
                vs = "" if v is None else (v if kind == "tb2" else int(bool(v)))
                vrows.append((iid, vs, patch_bytes, s_tree, int(has_log), int(tc), int(pq), note))
                manifest.append((ds, mode, fw, iid, s_tree, vs, patch_bytes, int(has_log), int(tc), int(pq), note))
            if not DRY:
                os.makedirs(dst, exist_ok=True)
                with open(f"{dst}/preds_w0_clean.jsonl", "w") as fh:
                    for p in preds: fh.write(json.dumps(p) + "\n")
                with open(f"{dst}/verdicts.tsv", "w") as fh:
                    fh.write("instance_id\tverdict\tpatch_bytes\tsource_tree\thas_log\thas_tool_calls\thas_llm_parquet\tnote\n")
                    for r in vrows: fh.write("\t".join(str(x) for x in r) + "\n")
            nv = [r[1] for r in vrows if r[1] != ""]
            ok = sum(1 for x in nv if float(x) >= 1.0)
            print(f"{ds:22s} {cell:20s} runs={len(cur):3d} graded={len(nv):3d} pass={ok:3d} ({100*ok/max(len(nv),1):.1f}%) "
                  f"from_tb2v={sum(1 for r in vrows if r[3]=='results_tb2v'):2d} no_tool_calls={sum(1 for r in vrows if not r[5]):2d} "
                  f"no_parquet={sum(1 for r in vrows if not r[6]):2d}  [{time.time()-t00:.0f}s]", flush=True)
if not DRY:
    def dump(name, hdr, rows):
        with open(f"{OUT}/{name}", "w") as fh:
            fh.write(hdr + "\n")
            for r in rows: fh.write("\t".join("" if x is None else str(x) for x in r) + "\n")
    dump("MANIFEST.tsv", "dataset\tmode\tframework\tinstance_id\tsource_tree\tverdict\tpatch_bytes\thas_log\thas_tool_calls\thas_llm_parquet\tnote", manifest)
    dump("replaced_runs.tsv", "dataset\tcell\tinstance_id\treplaced_source\treplaced_verdict\tnew_source\tnew_verdict\treason", replaced)
    dump("excluded_archived.tsv", "dataset\tcell\tarchived_name\tsource_tree", excluded)
print(f"done: {len(manifest)} runs, {len(replaced)} replaced by reruns, {len(excluded)} archived names excluded  [{time.time()-t00:.0f}s]")
