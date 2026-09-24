#!/usr/bin/env python3
"""Wall-time table for the explorer's "Wall times" card (explorer/data/wall.json).

Every task of H200_profiling_portal/clean_data (Pro, TB2, Verified, Lite) across the six settings
(thinking/instruct/coder × hermes/langgraph), built only from the explorer's own extracts:
  data/tag/index.json          tool-call span, tool seconds, wait seconds, calls  (build_tagdata.py)
  data/tl/<ds>/<cell>/<iid>.json   first LLM request / last LLM end, LLM seconds, requests (build_tldata.py)
plus the outcome of each run:
  Pro : <root>/grading_snapshot/resolved_<mode>_<fw>.txt   (official harness, one iid per line)
  TB2 : <root>/tb2_89_tmux[_busynotice]/<mode>/<fw>/preds_w*.jsonl  last line per instance_id, `reward`
        (tasks listed in grading_snapshot/tb2/cheated_tb2.tsv are marked "cheated")

wall = first LLM request -> last event of the run  (max(span, last) - min(0, first));
tool = tool seconds minus waiting; wait = tmux WAIT keystrokes + sleep polls; idle = the rest.

Output: {ds: {"tasks": {iid: {repo, cells: {cell: [wall, llm, tool, wait, idle, ncalls, nreq, resolved]}}},
              "summary": {cell: {metric: {n, total_h, mean, median, p90, max}}, ...}}}
Usage: python build_walldata.py [--root DIR] [--out FILE]
"""
import argparse, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tagdata as T   # noqa: E402  (ROOT, MODES, FWS)

DATA = os.path.join(T.EXPL, 'data')
DATASETS = ['pro_250', 'tb2_89', 'swebench_verified_200', 'swebench_lite_100']
METRICS = ['wall', 'llm', 'tool', 'wait', 'idle', 'ncalls', 'nreq']


def clean_verdicts(root, ds):
    """{cell: {iid: 1/0/None}} from clean/<ds>/<mode>/<fw>/verdicts.tsv (build_clean.py): official
    harness verdict for Pro/Verified/Lite, verifier reward for TB2; '' (never graded) -> None."""
    out = {}
    for mode in T.MODES:
        for fw in T.FWS:
            f = os.path.join(root, ds, mode, fw, 'verdicts.tsv')
            if not os.path.exists(f):
                continue
            cell = '%s__%s' % (mode, fw)
            out[cell] = {}
            with open(f) as fh:
                hdr = fh.readline().rstrip('\n').split('\t')
                for l in fh:
                    p = l.rstrip('\n').split('\t')
                    r = dict(zip(hdr, p))
                    v = r.get('verdict', '')
                    try:
                        out[cell][r['instance_id']] = None if v == '' else (1 if float(v) >= 1 else 0)
                    except ValueError:
                        out[cell][r['instance_id']] = None
    return out


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'total_h': round(sum(xs) / 3600, 2), 'mean': round(sum(xs) / len(xs), 1),
            'median': round(statistics.median(xs), 1), 'p90': round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 1),
            'max': round(xs[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=T.ROOT)
    ap.add_argument('--out', default=os.path.join(DATA, 'wall.json'))
    a = ap.parse_args()
    idx = json.load(open(os.path.join(DATA, 'tag', 'index.json')))
    result = {}
    for ds in DATASETS:
        if ds not in idx:
            continue
        verd = clean_verdicts(a.root, ds)
        tasks, summ = {}, {}
        for cell, ts in idx[ds].items():
            vals = {m: [] for m in METRICS}
            nres = 0
            for x in ts:
                f = os.path.join(DATA, 'tl', ds, cell, x['iid'] + '.json')
                m = x['m']
                if os.path.exists(f):
                    tm = json.load(open(f))['m']
                else:
                    tm = {'first': 0, 'last': 0, 'llm_s': 0, 'n': 0}
                wall = max(m['span'], tm.get('last') or 0) - min(0, tm.get('first') or 0)
                llm = tm.get('llm_s') or 0.0
                tool = max(0.0, m['serial'] - m['wait'])
                wait = m['wait']
                idle = max(0.0, wall - llm - tool - wait)
                resolved = verd.get(cell, {}).get(x['iid'])
                if resolved == 1:
                    nres += 1
                rec = [round(wall, 1), round(llm, 1), round(tool, 1), round(wait, 1), round(idle, 1),
                       x['ncalls'], tm.get('n') or 0, resolved]
                tasks.setdefault(x['iid'], {'repo': x['repo'], 'cells': {}})['cells'][cell] = rec
                for k, mname in enumerate(METRICS):
                    vals[mname].append(rec[k])
            summ[cell] = {mname: stats(v) for mname, v in vals.items()}
            summ[cell]['resolved'] = nres
            print('%-24s %-20s tasks=%3d wall=%7.1fh llm=%6.1fh tool=%6.1fh wait=%5.1fh resolved=%d'
                  % (ds, cell, len(ts), sum(vals['wall']) / 3600, sum(vals['llm']) / 3600,
                     sum(vals['tool']) / 3600, sum(vals['wait']) / 3600, nres))
        result[ds] = {'tasks': tasks, 'summary': summ,
                      'wall_def': 'first LLM request -> last event of the run; tool excludes waiting; idle = remainder'}
    json.dump(result, open(a.out, 'w'), separators=(',', ':'))
    print('wrote', a.out, os.path.getsize(a.out) // 1024, 'KB')


if __name__ == '__main__':
    main()
