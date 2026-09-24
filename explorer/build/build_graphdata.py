#!/usr/bin/env python
"""Build the Markov x data-dependency multiplex graphs (explorer/data/graphdata.json)
at three granularities (whole corpus / per repo / per task) and two vocabularies
(specific tool type via verb_of, high-level function Read/Edit/Execute/Navigate/Setup).
Adjacent call pairs are classified RAW/WAR/WAW/NONE from heuristic per-invocation
read/write sets (files + coarse resources: worktree 'wt', 'deps', 'art').
Usage: TOOLSPEC_CORPUS=<corpus dir> python build_graphdata.py"""
import sys, os, json, re
from collections import defaultdict, Counter
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'spork_repro')); sys.path.insert(0, os.path.join(BASE, 'paste_repro'))
os.environ.setdefault('SIG_LANG', 'en')
from plot_signature_stats import verb_of, BOOK, CORPUS

PATH = re.compile(r'(?<![\w@])((?:\.{0,2}/)?[\w.-]+(?:/[\w.-]+)+|[\w-]+\.(?:ts|tsx|js|jsx|py|go|json|txt|md|css|pcss|yaml|yml|sh|tsb|log|snap))\b')
REDIR = re.compile(r'>>?\s*([\w./-]+)')

def rw_of(c):
    s = re.sub(r'^\s*(cd\s+\S+\s*&&\s*)+', '', c).strip()
    s = re.sub(r'^\s*timeout\s+\d+\S*\s+', '', s)
    s = re.sub(r'^\s*[A-Z_][A-Z0-9_]*=\S+\s+', '', s)
    head = s.split('|')[0]
    R, W = set(), set()
    for t in REDIR.findall(c.split('\n')[0]):
        if t != '/dev/null':
            W.add(t)
            if not t.startswith('/tmp'): W.add('wt')
    v = head.split()[:1]; vb = os.path.basename(v[0]) if v else ''
    files = [f for f in PATH.findall(head) if not f.startswith('-')]
    if re.match(r'(npm|npx|yarn|pnpm|go|jest|mocha|vitest|pytest|tsc|make|cargo|node|python3?|bash|sh)\b', vb) and not re.search(r'python3? -c', head):
        if re.search(r'\b(python3?|node|bash|sh)\s+/tmp/', head):
            W.add('wt'); R.add('wt'); return R, W, 'Edit'
        R.add('wt')
        if re.search(r'\b(build|compile|install)\b', head): W.add('art')
        if re.search(r'\b(npm|pip|yarn|pnpm)\s+(install|ci|add)\b', head): W.add('deps')
        return R, W, ('Setup' if re.search(r'\b(install|ci|add)\b', head) else 'Execute')
    if '<<' in c or re.search(r'\bsed\s+-i\b|\b(cp|mv|rm|patch|tee|touch|mkdir|truncate)\b|\bgit (apply|checkout|stash|reset|restore)\b', head):
        for f in files: W.add(f)
        if [f for f in files if not f.startswith('/tmp')] or '<<' in c: W.add('wt')
        return R, W, 'Edit'
    if re.match(r'(cat|head|tail|less|wc|grep|rg|find|ls|diff|git|pwd|which|tree|file|stat|du)\b', vb):
        for f in files: R.add(f)
        if re.search(r'\bgrep\s+-r|\bfind\b', head): R.add('wt')
        return R, W, 'Read'
    if re.match(r'(cd|export|source|echo|sleep|env|set)\b', vb):
        return R, W, 'Navigate'
    R.add('wt'); return R, W, 'Other'

def dep_of(Ra, Wa, Rb, Wb):
    if Wa & Rb: return 'RAW'
    if Ra & Wb: return 'WAR'
    if Wa & Wb: return 'WAW'
    return 'NONE'

def build(seq):
    nodes = Counter(l for l, _, _ in seq)
    edges = defaultdict(lambda: {'n': 0, 'RAW': 0, 'WAR': 0, 'WAW': 0, 'NONE': 0})
    for (la, Ra, Wa), (lb, Rb, Wb) in zip(seq, seq[1:]):
        e = edges[(la, lb)]; e['n'] += 1; e[dep_of(Ra, Wa, Rb, Wb)] += 1
    return nodes, edges

def pack(nodes, edges, topk=12):
    keep = set(k for k, _ in nodes.most_common(topk))
    nn = Counter()
    for k, v in nodes.items(): nn[k if k in keep else 'other'] += v
    ee = defaultdict(lambda: {'n': 0, 'RAW': 0, 'WAR': 0, 'WAW': 0, 'NONE': 0})
    for (a, b), v in edges.items():
        t = ee[(a if a in keep else 'other', b if b in keep else 'other')]
        for f in ('n', 'RAW', 'WAR', 'WAW', 'NONE'): t[f] += v[f]
    out_tot = Counter()
    for (a, b), v in ee.items(): out_tot[a] += v['n']
    return {'nodes': [[k, v] for k, v in nn.most_common()],
            'edges': [[a, b, v['n'], round(v['n'] / out_tot[a], 3), v['RAW'], v['WAR'], v['WAW'], v['NONE']]
                      for (a, b), v in sorted(ee.items(), key=lambda kv: -kv[1]['n'])]}

def agg(seqs):
    nodes = Counter(); edges = defaultdict(lambda: {'n': 0, 'RAW': 0, 'WAR': 0, 'WAW': 0, 'NONE': 0})
    for s in seqs:
        n, e = build(s); nodes.update(n)
        for k, v in e.items():
            t = edges[k]
            for f in ('n', 'RAW', 'WAR', 'WAW', 'NONE'): t[f] += v[f]
    return nodes, edges

per_task = {}; per_repo = defaultdict(list); at = []; af = []
for d in sorted(os.listdir(CORPUS)):
    m = re.match(r'instance_([^_]+)__([A-Za-z0-9.-]+?)-([0-9a-f]{7,})', d)
    if not m: continue
    f = os.path.join(CORPUS, d, 'swe_bench_pro__tool_calls.jsonl')
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if not BOOK.search(str(r.get('command') or ''))]
    rows.sort(key=lambda r: r.get('start_s') or 0)
    st, sf = [], []
    for r in rows:
        c = str(r.get('command') or ''); R, W, fn = rw_of(c)
        st.append((verb_of(c), R, W)); sf.append((fn, R, W))
    if not st: continue
    per_task[(m.group(2), m.group(3)[:8])] = (st, sf)
    per_repo[m.group(2)].append((st, sf)); at.append(st); af.append(sf)

OUT = {'all': {'tool': pack(*agg(at)), 'func': pack(*agg(af), topk=8)}, 'repos': {}, 'tasks': {}}
for repo, lst in per_repo.items():
    OUT['repos'][repo] = {'tool': pack(*agg([s for s, _ in lst])), 'func': pack(*agg([f for _, f in lst]), topk=8)}
for (repo, sha), (st, sf) in per_task.items():
    OUT['tasks'].setdefault(repo, {})[sha] = {'tool': pack(*build(st), topk=10), 'func': pack(*build(sf), topk=8)}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'graphdata.json')
json.dump(OUT, open(out, 'w'), separators=(',', ':'))
print('wrote', out)
