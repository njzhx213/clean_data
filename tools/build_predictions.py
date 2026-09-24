#!/usr/bin/env python3
"""Build a SWE-bench-Pro predictions file from a results tree.

    build_predictions.py <results_dir> <prefix> <out.json> [--ids pro_250.norm.jsonl] [--no-hygiene]

* reads every preds_w*.jsonl (and preds*.jsonl) under <results_dir>, keeps the
  LAST line per instance_id (reruns append; the newest run wins)
* restricts to the task list (default pro_250) and emits one entry per task,
  empty patch for tasks without an answer (the official harness scores those 0)
* patch hygiene: drops whole `diff --git` sections whose path matches the
  shared runtime-junk list (warmstack/patch_hygiene.py) -- dump.rdb, *.aof,
  caches... -- exactly what the live capture now excludes, applied offline to
  patches captured before the fix (6 thinking-hermes apply failures, 09-04)
* prints a summary: answered / empty / sections dropped
"""
import fnmatch
import glob
import json
import os
import re
import sys

sys.path.insert(0, "/eic/data/haoxiong_gpu6/tool_speculation/warmstack")
import patch_hygiene  # noqa: E402

DEFAULT_IDS = "/eic/data/haoxiong_gpu6/bench_agentic/subsets/pro_250.norm.jsonl"


def junk(path):
    for pat in patch_hygiene.RUNTIME_JUNK:
        p = pat.replace("/**", "")
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, p) \
                or path.startswith(p + "/") or ("/" + p + "/") in ("/" + path):
            return True
    return False


def clean_patch(patch):
    if not patch or not patch.strip():
        return patch, []
    sections = re.split(r"(?=^diff --git )", patch, flags=re.M)
    kept, dropped = [], []
    for s in sections:
        if not s.strip():
            continue
        m = re.match(r"diff --git a/(\S+) b/", s)
        path = m.group(1) if m else ""
        if path and junk(path):
            dropped.append(path)
            continue
        # binary sections (a `go build` executable, images...): the official
        # harness strips them itself (swe_bench_pro_eval.strip_binary_hunks),
        # so dropping here changes nothing for grading and keeps the
        # predictions file small (coder-lg flipt-b6ce: 27 MB `flipt` binary)
        if re.search(r"^GIT binary patch$", s, re.M) or re.search(r"^Binary files .* differ$", s, re.M):
            dropped.append(path + " [binary]")
            continue
        kept.append(s)
    out = "".join(kept)
    if out and not out.endswith("\n"):
        out += "\n"
    return out, dropped


def main():
    args = sys.argv[1:]
    hygiene = True
    ids_path = DEFAULT_IDS
    if "--no-hygiene" in args:
        hygiene = False
        args.remove("--no-hygiene")
    if "--ids" in args:
        i = args.index("--ids")
        ids_path = args[i + 1]
        del args[i:i + 2]
    rdir, prefix, out = args
    ids = [json.loads(l)["instance_id"] for l in open(ids_path) if l.strip()]
    latest = {}
    files = sorted(glob.glob(os.path.join(rdir, "preds_w*.jsonl")) + glob.glob(os.path.join(rdir, "preds*.jsonl")),
                   key=os.path.getmtime)
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            iid = j.get("instance_id")
            if iid in ids:
                latest[iid] = j.get("patch") or j.get("model_patch") or ""
    # A rerun (archived+requeued task) lands in the preds file of WHICHEVER
    # worker picked it up, so "last line in mtime-sorted files" can still
    # return the stale line from the earlier run (fec1 09:51: rerun's 1195 B
    # patch in preds_w7 lost to the old empty line in a later-touched file).
    # The authoritative run is the latest `uni-N done` in progress.log: take
    # that worker's last line for the task.
    by_worker = {}
    try:
        m_cell = re.search(r"results_nonet/([a-z]+)/([a-z]+)", os.path.abspath(rdir))
        if m_cell:
            cell = f"nonet_{m_cell.group(1)}_{m_cell.group(2)}"
            plog = "/eic/data/haoxiong_gpu6/tool_speculation/warmstack/profile_all/progress.log"
            for line in open(plog):
                m = re.search(r"uni-(\d+) done " + re.escape(cell) + r"/(instance_\S+) rc=", line)
                if m and not (m.group(1).isdigit() and int(m.group(1)) >= 90):   # smoke workers uni-94/97/8xx only; uni-9 is a real card (09-06 fix)
                    by_worker[m.group(2)] = m.group(1)
        for iid, w in by_worker.items():
            f = os.path.join(rdir, f"preds_w{w}.jsonl")
            if not os.path.exists(f):
                continue
            for line in open(f):
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if j.get("instance_id") == iid:
                    latest[iid] = j.get("patch") or j.get("model_patch") or ""
    except Exception as e:
        print(f"warning: worker-ordered selection skipped: {e}", file=sys.stderr)
    preds, empty, dropped_total = [], 0, []
    for iid in ids:
        p = latest.get(iid, "")
        if hygiene:
            p, dropped = clean_patch(p)
            dropped_total += [(iid, d) for d in dropped]
        if not p.strip():
            empty += 1
        preds.append({"instance_id": iid, "patch": p, "prefix": prefix})
    json.dump(preds, open(out, "w"))
    print(f"{out}: {len(preds)} entries, answered={len(preds) - empty}, empty={empty}, "
          f"junk sections dropped={len(dropped_total)} in {len({i for i, _ in dropped_total})} tasks")
    for iid, d in dropped_total[:20]:
        print("   dropped", iid[:50], d)


if __name__ == "__main__":
    main()
