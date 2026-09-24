#!/usr/bin/env python3
"""LLM-side segments for the explorer timeline block (explorer/data/tl/…).

Companion to build_tagdata.py (which uses only tool_calls.jsonl).  A timeline needs the
model's turns too, so this reads, per canonical run, the LLM request log
  <root>/<dataset>/<mode>/<fw>/traj/<iid>/swe_bench_pro__live_vllm_openai.parquet
and keeps only what a timeline draws: request start/end (same monotonic clock as the
tool calls -- verified: requests interleave with tool calls, no overlap on langgraph),
input/output tokens, hit_max_tokens.  Times are relative to the SAME t0 as the TAG task
file (first agent tool call after bookkeeping removal), so the two files overlay directly.

Output per run: data/tl/<dataset>/<mode>__<fw>/<iid>.json
  {iid, llm:[[start, end, in_tok, out_tok, hitmax], ...], m:{n, llm_s, in_tok, out_tok, first, last}}
Usage: python build_tldata.py [--root DIR] [--datasets a,b] [--out DIR]
"""
import argparse, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tagdata as T   # noqa: E402  (loader, BOOK, manifests)

PQ = 'swe_bench_pro__live_vllm_openai.parquet'
COLS = ['request_start_s', 'request_end_s', 'input_tokens', 'output_tokens', 'hit_max_tokens', 'wall_ms']


def main():
    ap = argparse.ArgumentParser()
    T.add_common_args(ap)
    ap.add_argument('--out', default=os.path.join(T.EXPL, 'data', 'tl'))
    a = ap.parse_args()
    tot = {'tasks': 0, 'llm_rows': 0, 'no_pq': 0}
    for ds, mode, fw, arm, cell, traj in T.iter_cells(a):
        man = T.load_manifest(ds)
        if man is None:
            continue
        outdir = T.guard_out(os.path.join(a.out, ds, cell))
        os.makedirs(outdir, exist_ok=True)
        n = 0
        for d in sorted(os.listdir(traj)):
            if d not in man:
                continue
            tc = os.path.join(traj, d, T.TC); pq = os.path.join(traj, d, PQ)
            if not os.path.exists(tc):
                continue
            if not os.path.exists(pq):
                tot['no_pq'] += 1
                continue
            rows = [json.loads(l) for l in open(tc) if l.strip()]
            rows = [r for r in rows if not T.BOOK.search(T.cmd_of(r))]
            if not rows:
                continue
            t0 = min(float(r['start_s']) for r in rows)      # == build_tagdata's t0
            try:
                df = pd.read_parquet(pq, columns=COLS)
            except Exception:
                df = pd.read_parquet(pq)
            df = df.sort_values('request_start_s')
            llm = [[round(float(r.request_start_s) - t0, 2), round(float(r.request_end_s) - t0, 2),
                    int(r.input_tokens or 0), int(r.output_tokens or 0), 1 if bool(r.hit_max_tokens) else 0]
                   for r in df.itertuples()]
            m = {'n': len(llm), 'llm_s': round(sum(e - s for s, e, *_ in llm), 1),
                 'in_tok': int(df.input_tokens.sum()), 'out_tok': int(df.output_tokens.sum()),
                 'first': llm[0][0] if llm else None, 'last': llm[-1][1] if llm else None}
            json.dump({'iid': d, 'llm': llm, 'm': m}, open(os.path.join(outdir, d + '.json'), 'w'),
                      separators=(',', ':'))
            n += 1; tot['tasks'] += 1; tot['llm_rows'] += len(llm)
        print('%-24s %-28s tasks=%3d' % (ds, cell, n))
    print('wrote', a.out, tot)


if __name__ == '__main__':
    main()
