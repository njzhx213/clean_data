#!/usr/bin/env python3
"""build_oppdata.py -- the CPU-side opportunity assessment read off the priced TAG graphs.

WHY THIS FILE EXISTS.  The census (build_tagdata.py) prices every slow call as if the identity
of the next node were known for free: L1 (reuse), L2/L3 (known identity, only timing open) and
PASTE/T1 (identity or its trigger must be PREDICTED) all get oracle seconds.  This builder adds
what a predictor actually has to get right for the last two -- the tool TYPE, the ARGUMENTS and
the WINDOW -- plus the tool->tool transition frequency that a graph predictor would use, and
composes the oracle price with those predicates.  Everything is decided ONLINE: at call i only
calls 0..i-1 are looked at (PLAN.md G5).  Cross-task pools are computed too, but labelled as
what they are on a 10-repo corpus: leaky.

WHAT IT READS.  data/tag/index.json -> data/tag/<ds>/<cell>/<iid>.json, i.e. the already-priced
rows t[i] = [i, tool, dur, st, est, slack, cell, val, first, win, val_w], the fn class in
n[i][1] (6 = Wait), the display text cmd[i], cwd and repo.  It never re-prices -- the oracle
numbers stay identical to the digit (the page checks this: snap_score.py) -- and it never
walks tool_calls.jsonl again (302 s over NFS for the 3333 graphs; a --cache file makes reruns
cost seconds).

WHAT IT WRITES (all under data/opp/, guard_out on every path):
  index.json                      {meta, tasks:{ds:{cell:{iid: X}}}}  -- per-task extension, joined by iid
  <ds>/<cell>/<iid>.json          per-call predicate rows for the node panel and the timeline
  <ds>/<cell>/_cell.json          per-cell aggregates: markov, accuracy, window, probe race, speedup
  census.json + TAG_OPP_CENSUS.md the tables the assessment document quotes
A task absent from index.json is NOT COMPUTED; a task with no PASTE/T1 rows has every key with
n = 0 -- missing and zero are different things and the page keeps them apart.

LABEL VOCABULARY = verb_of(key_cwd(cmd, cwd).core): a function of the TAG identity (every
identity has exactly one label), it separates go build / go test / npm run X / python -m pytest
-- the families the L2/L3 pricing keys on -- and it is the vocabulary of the existing Markov
card, so the transition graph stays comparable.  verb_of is AST-lifted from the shared tree
(importing that module would write a .pyc into the write-guarded tree).  signature_of (PASTE's
coarser label) is scored alongside for comparability with its held-out 0.26.

Usage:
  python3 build_oppdata.py --datasets pro_250_nonet --cells instruct__langgraph     # one cell
  python3 build_oppdata.py                                                           # everything
  python3 build_oppdata.py --force          # recompute cells whose _cell.json is up to date
"""
import argparse, ast, json, math, os, re, statistics, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tagdata as T   # noqa: E402  (key_cwd, TOOLS, SPINE, stats, ceiling, guard_out, s_share, INCR, TESTP)

PASTE_DIR = '/eic/data/haoxiong_gpu6/bench_agentic/technique/base/paste_repro'
sys.path.insert(0, PASTE_DIR)
from paste.signatures import canonicalize, core_of, extract_paths, signature_of   # noqa: E402

GH_SIG = '/eic/data/haoxiong_gpu6/tool_speculation/gh_staging/spork_repro/plot_signature_stats.py'

PROBE_C = (0.0, 1.2, 3.3, 5.0, 9.7)     # probe cost sweep, s: 0 = no probe, 1.2 = KV-cached decode,
                                        # 3.3 = measured early-stopped probe, 9.7 = the 09-08 live probe
LOOKBACK = (1, 2, 3, 5, 10, None)       # None = unlimited (task start)
KMAX = 4                                # top-K width for type / argument coverage
PRED_CELLS = {4: 'PASTE', 5: 'T1'}      # the two predicted-speculation cells (build_tagdata.CELLS)
CHEAP = {'echo', 'ls', 'cat', 'pwd', 'head', 'tail', 'wc', 'which', 'true', 'date', 'printf', 'stat',
         'file', 'du', 'df'}            # the tmux-artefact column: emitted for EVERY cell, no dataset special case
FAM = {'const': 1, 'repeat': 2, 'copy_path': 4, 'copy_path_ub': 8}
VN_KEYS = ['%d%d%d' % (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]   # TYPE ARGS FIT
WAIT_FN = T.FNS.index('Wait')
CELL_L1, CELL_L2, CELL_L3 = 1, 2, 3
MODEL_P = (0.30, 0.44)                  # replay f=0 slow hit / f=1 slow hit (EXPERIMENT_FORK_REPLAY.md)
S_MEAS = tuple(k.lower() for k in T.S_SHARE)


# ---- verb_of lifted from the shared tree without importing it (same trick as build_tagdata._lift_rw_of)
def _lift_verb_of():
    tree = ast.parse(open(GH_SIG).read())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == 'verb_of')
            or (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == '_ENV')]
    tree.body = keep
    ns = {'re': re, 'os': os, 'canonicalize': canonicalize, 'core_of': core_of, '__file__': GH_SIG}
    exec(compile(tree, GH_SIG, 'exec'), ns)
    return ns['verb_of']


verb_of = _lift_verb_of()
_COMMENT = re.compile(r'^\s*#[^\n]*\n')


def label_of(tool_idx, fn_idx, cmd, cwd):
    """one label per TAG identity; Wait rows are 'wait'; non-bash tools are their tool name."""
    tool = T.TOOLS[tool_idx]
    if fn_idx == WAIT_FN or tool == 'wait':
        return 'wait'
    if tool != 'bash':
        return tool
    core = T.key_cwd(cmd, cwd)[1]
    for _ in range(3):
        core2 = _COMMENT.sub('', core)
        if core2 == core:
            break
        core = core2
    return verb_of(core)


def ident_of(tool_idx, fn_idx, cmd, cwd):
    tool = T.TOOLS[tool_idx]
    if fn_idx == WAIT_FN or tool == 'wait':
        return None
    if tool != 'bash':
        return (tool, cmd)
    return ('bash',) + T.key_cwd(cmd, cwd)


def sig_of(tool_idx, fn_idx, cmd):
    tool = T.TOOLS[tool_idx]
    if fn_idx == WAIT_FN or tool == 'wait':
        return 'wait'
    return signature_of('bash' if tool == 'bash' else tool, cmd if tool == 'bash' else None)


# ---- n-gram machinery (k = 1..3 contexts, tie -> most recent) --------------------------------
class Gram:
    """counts[k][ctx][label] and last[k][ctx][label] = index of the last occurrence (tie-break)."""
    def __init__(self):
        self.c = {k: defaultdict(Counter) for k in (1, 2, 3)}
        self.last = {k: defaultdict(dict) for k in (1, 2, 3)}
        self.uni = Counter(); self.ulast = {}

    def add(self, seq, j, target):
        for k in (1, 2, 3):
            if j >= k:
                ctx = tuple(seq[j - k:j])
                self.c[k][ctx][target] += 1; self.last[k][ctx][target] = j
        self.uni[target] += 1; self.ulast[target] = j

    def ranked(self, k, ctx):
        row = self.c[k].get(ctx)
        if not row:
            return None
        last = self.last[k][ctx]
        return [l for l, _ in sorted(row.items(), key=lambda kv: (-kv[1], -last.get(kv[0], -1)))]

    def backoff(self, seq, i):
        """-> (ranked labels, k_used); k_used 0 = unigram, -1 = nothing seen yet."""
        for k in (3, 2, 1):
            if i >= k:
                r = self.ranked(k, tuple(seq[i - k:i]))
                if r:
                    return r, k
        if self.uni:
            return [l for l, _ in sorted(self.uni.items(), key=lambda kv: (-kv[1], -self.ulast.get(kv[0], -1)))], 0
        return [], -1

    def fixed(self, seq, i, k):
        if i < k:
            return []
        return self.ranked(k, tuple(seq[i - k:i])) or []


class PoolGram:
    """cell-level counts minus an excluded set of tasks (leave-one-out / leave-repo-out)."""
    def __init__(self, total, minus):
        self.total, self.minus = total, minus     # both: {k: {ctx: Counter}}, plus 'uni'

    def _row(self, k, ctx):
        row = Counter(self.total[k].get(ctx, ()))
        row.subtract(self.minus[k].get(ctx, ()))
        return [l for l, n in sorted(row.items(), key=lambda kv: (-kv[1], kv[0])) if n > 0]

    def backoff(self, seq, i):
        for k in (3, 2, 1):
            if i >= k:
                r = self._row(k, tuple(seq[i - k:i]))
                if r:
                    return r, k
        u = Counter(self.total['uni']); u.subtract(self.minus['uni'])
        r = [l for l, n in sorted(u.items(), key=lambda kv: (-kv[1], kv[0])) if n > 0]
        return r, (0 if r else -1)


def empty_counts():
    return {1: defaultdict(Counter), 2: defaultdict(Counter), 3: defaultdict(Counter), 'uni': Counter()}


def add_counts(dst, g):
    for k in (1, 2, 3):
        for ctx, row in g.c[k].items():
            dst[k][ctx].update(row)
    dst['uni'].update(g.uni)


def gram_counts(g):
    return {1: g.c[1], 2: g.c[2], 3: g.c[3], 'uni': g.uni}


# ---- per-task annotation ------------------------------------------------------------------------
def task_labels(g):
    """label / signature / identity per row, plus the label-id sequence for the task file."""
    t, n, cmd, cwd = g['t'], g['fn'], g['cmd'], g['cwd']
    labs, sigs, ids = [], [], []
    for i, r in enumerate(t):
        labs.append(label_of(r[1], n[i], cmd[i], cwd))
        sigs.append(sig_of(r[1], n[i], cmd[i]))
        ids.append(ident_of(r[1], n[i], cmd[i], cwd))
    return labs, sigs, ids


def final_grams(labs, sigs, ids):
    """the task's own transition counts after seeing everything (for the cell pools)."""
    gl, gs = Gram(), Gram()
    for j in range(len(labs)):
        if labs[j] != 'wait':
            gl.add(labs, j, labs[j])
        if sigs[j] != 'wait':
            gs.add(sigs, j, sigs[j])
    return gl, gs


def _zero_cell():
    return {'g': [0.0, 0.0], 'gt': [0.0, 0.0], 'ga': [0.0, 0.0], 'p': [0.0, 0.0],
            'pr': [0.0] * len(PROBE_C), 'vn': {k: 0.0 for k in VN_KEYS}, 'n': [0] * 6,
            'tK': [0] * KMAX, 'gK': [0.0] * KMAX, 'aK': [0.0] * KMAX}


def annotate(g, labs, sigs, ids, pool_loo, pool_xrepo, sigpool_loo):
    """returns (X: per-task extension for index.json, pv: per-call rows, acc: per-row accuracy
    records for the cell tables).  Online: at row i only rows < i feed the predictors."""
    t, fn, cmd, cwd, repo = g['t'], g['fn'], g['cmd'], g['cwd'], g['repo']
    N = len(t)
    gl, gs = Gram(), Gram()                 # in-task label / signature n-grams
    gid = {k: defaultdict(Counter) for k in (1, 2, 3)}   # label-context -> identity counts (const)
    last_by_label, ids_by_label = {}, defaultdict(list)   # label -> last row idx; label -> recency list of identities
    X = {c: _zero_cell() for c in PRED_CELLS.values()}
    X.update({'lb': [0.0] * len(LOOKBACK), 'lbv': [0.0] * len(LOOKBACK), 'cheap': [0.0, 0, 0.0, 0],
              'sa': {'l1': [0.0, 0.0], 'known': [0.0, 0.0], 'pred': [0.0, 0.0], 'predT': [0.0, 0.0]},
              # cheap = [sw pred, n pred, sw L1, n L1]: slow echo/ls/cat/... rows -- under tmux their latency is the previous program's tail
              'smeas': int(any(k in (repo or '').lower() for k in S_MEAS)), 'parl1': 0.0,
              'nolabel': 0, 'trunc': 0, 'nrows': N, 'nslowpred': 0})
    share = T.s_share(repo or '')
    pv, acc = [], []
    ends = [r[3] + r[2] for r in t]
    par_rows = set()
    for p in g.get('par', []):
        par_rows.update(range(p[0], p[1] + 1))
    for i in range(N):
        r = t[i]; dur, st, slack, cell, val, win, valw = r[2], r[3], r[5], r[6], r[7], r[9], r[10]
        lab, sig, ident = labs[i], sigs[i], ids[i]
        if len(cmd[i]) >= 1200:
            X['trunc'] += 1
        if lab == '(empty)':
            X['nolabel'] += 1
        is_wait = lab == 'wait'
        slow = (not is_wait) and dur >= T.SLOW
        # ---- predictions for this row (before it is added to any counter)
        if not is_wait:
            rk, kused = gl.backoff(labs, i)
            fixed = {k: gl.fixed(labs, i, k) for k in (1, 2, 3)}
            rp, kp = pool_loo.backoff(labs, i)
            rx, _ = pool_xrepo.backoff(labs, i)
            rh = rk if kused > 0 else rp
            rs, _ = gs.backoff(sigs, i)
            rsp, _ = sigpool_loo.backoff(sigs, i)
            top1 = bool(rk) and rk[0] == lab
            rec = {'i': i, 'dur': dur, 'cell': cell, 'slow': slow,
                   'backoff': (top1, lab in rk[:3]), 'pool_loo': (bool(rp) and rp[0] == lab, lab in rp[:3]),
                   'pool_xrepo': (bool(rx) and rx[0] == lab, lab in rx[:3]),
                   'hybrid': (bool(rh) and rh[0] == lab, lab in rh[:3]),
                   'sig_backoff': (bool(rs) and rs[0] == sig, sig in rs[:3]),
                   'sig_pool_loo': (bool(rsp) and rsp[0] == sig, sig in rsp[:3])}
            for k in (1, 2, 3):
                rec['k%d' % k] = (bool(fixed[k]) and fixed[k][0] == lab, lab in fixed[k][:3])
            acc.append(rec)
        # ---- lookback and S/A on every slow row
        if slow:
            for q, k in enumerate(LOOKBACK):
                Wk = st if (k is None or i < k) else st - ends[i - k]
                Wk = max(0.0, Wk)
                X['lb'][q] += min(dur, Wk)
                X['lbv'][q] += min(dur, Wk, slack)
            fam_sa = T.TOOLS[r[1]] == 'bash' and bool(T.INCR.search(cmd[i]) or T.TESTP.search(cmd[i]))
            if fam_sa:
                grp = 'l1' if cell == CELL_L1 else ('known' if cell in (CELL_L2, CELL_L3) else ('pred' if cell in PRED_CELLS else None))
                if grp:
                    X['sa'][grp][0] += share * dur
                    X['sa'][grp][1] += min((1 - share) * dur, win)
            if cell == CELL_L1 and i in par_rows:
                X['parl1'] += valw
            if cell == CELL_L1 and lab in CHEAP:
                X['cheap'][2] += valw; X['cheap'][3] += 1
        # ---- the three predicates, only on slow predicted-speculation rows
        if slow and cell in PRED_CELLS:
            c = PRED_CELLS[cell]; Xc = X[c]
            X['nslowpred'] += 1
            # TYPE
            top1 = bool(rk) and rk[0] == lab
            topK = [lab in rk[:K] for K in range(1, KMAX + 1)]
            top1p = bool(rp) and rp[0] == lab
            # ARGS (given the TRUE label; when TYPE hits the predicted label IS the true label)
            fam = 0
            recent = ids_by_label.get(lab, [])
            arank_r = 0
            for rank, idd in enumerate(reversed(recent), 1):     # most recent first
                if idd == ident:
                    arank_r = rank; break
            freq = Counter(recent)
            arank_f = 0
            if ident in freq:
                arank_f = 1 + sum(1 for idd, nn in freq.items() if nn > freq[ident] or (nn == freq[ident] and idd < ident))
            acs = len(freq)
            if recent and recent[-1] == ident:
                fam |= FAM['repeat']
            for k in (3, 2, 1):
                if i >= k:
                    row = gid[k].get(tuple(labs[i - k:i]))
                    if row:
                        best = max(row.items(), key=lambda kv: kv[1])[0]
                        if best == ident:
                            fam |= FAM['const']
                        break
            j = last_by_label.get(lab)
            srcs = []
            for back in (1, 2):
                if i - back >= 0:
                    srcs += extract_paths(cmd[i - back])
            srcs = list(dict.fromkeys(srcs))
            if j is not None and not (fam & FAM['repeat']):
                tmpl = cmd[j]; slots = extract_paths(tmpl)[:3]
                cands = [s for s in srcs if s not in slots][:8]
                hit = False
                for slot in slots:
                    for src in cands:
                        cand = tmpl.replace(slot, src, 1)
                        if ident_of(r[1], fn[i], cand, cwd) == ident:
                            hit = True; break
                    if hit:
                        break
                if hit:
                    fam |= FAM['copy_path']
            mypaths = extract_paths(cmd[i])
            if mypaths and set(mypaths) <= set(srcs):
                fam |= FAM['copy_path_ub']
            args = bool(fam & 7)
            # WINDOW
            fit = win >= dur
            frac = min(dur, win) / dur if dur > 0 else 0.0
            # composed values
            add = lambda key, cond: (Xc[key].__setitem__(0, Xc[key][0] + val), Xc[key].__setitem__(1, Xc[key][1] + valw)) if cond else None
            add('gt', top1); add('g', top1 and args); add('ga', args); add('p', top1p and args)
            for q, cc in enumerate(PROBE_C):
                Xc['pr'][q] += min(val, max(0.0, win - cc))
            Xc['vn']['%d%d%d' % (top1, args, fit)] += valw
            nn = Xc['n']; nn[0] += 1; nn[1] += top1; nn[2] += args; nn[3] += fit; nn[4] += top1 and args; nn[5] += top1 and args and fit
            for K in range(KMAX):
                Xc['tK'][K] += topK[K]
                if topK[K] and args:
                    Xc['gK'][K] += valw
                if arank_r and arank_r <= K + 1:
                    Xc['aK'][K] += valw
            if lab in CHEAP:
                X['cheap'][0] += valw; X['cheap'][1] += 1
            if fam_sa and top1:          # hint -> S-stage prewarm is actionable only when the type was named
                X['sa']['predT'][0] += share * dur
                X['sa']['predT'][1] += min((1 - share) * dur, win)
            pv.append([i, rk[0] if rk else '', kused, int(top1), int(lab in rk[:3]), int(top1p), fam, int(args),
                       int(fit), round(frac, 3), acs, arank_r, arank_f])
        # ---- now this row becomes history
        if not is_wait:
            gl.add(labs, i, lab); gs.add(sigs, i, sig)
            for k in (1, 2, 3):
                if i >= k:
                    gid[k][tuple(labs[i - k:i])][ident] += 1
            last_by_label[lab] = i
            ids_by_label[lab].append(ident)
    # round for the index
    for c in PRED_CELLS.values():
        Xc = X[c]
        for key in ('g', 'gt', 'ga', 'p'):
            Xc[key] = [round(v, 1) for v in Xc[key]]
        Xc['pr'] = [round(v, 1) for v in Xc['pr']]
        Xc['vn'] = {k: round(v, 1) for k, v in Xc['vn'].items()}
        Xc['gK'] = [round(v, 1) for v in Xc['gK']]; Xc['aK'] = [round(v, 1) for v in Xc['aK']]
    X['lb'] = [round(v, 1) for v in X['lb']]; X['lbv'] = [round(v, 1) for v in X['lbv']]
    X['cheap'][0] = round(X['cheap'][0], 1); X['cheap'][2] = round(X['cheap'][2], 1); X['parl1'] = round(X['parl1'], 1)
    for k in X['sa']:
        X['sa'][k] = [round(v, 1) for v in X['sa'][k]]
    return X, pv, acc


# ---- loading: tag index -> task files, with a single-file cache ----------------------------------
def load_tag_task(path):
    g = json.load(open(path))
    return {'t': g['t'], 'fn': [r[1] for r in g['n']], 'cmd': g['cmd'], 'cwd': g.get('cwd') or '/app',
            'repo': g.get('repo') or '', 'par': g.get('par', []), 'rc': g['rc'], 'span': g['m']['span']}


def load_cell(tagdir, ds, cell, rows, cache, cache_fh):
    """rows = the index rows for the cell; returns {iid: g}, reading the cache first."""
    out, missing = {}, []
    for x in rows:
        key = '%s/%s/%s' % (ds, cell, x['iid'])
        if key in cache:
            out[x['iid']] = cache[key]
        else:
            missing.append(x['iid'])
    for iid in missing:
        p = os.path.join(tagdir, ds, cell, iid + '.json')
        if not os.path.exists(p):
            continue
        g = load_tag_task(p)
        out[iid] = g
        if cache_fh is not None:
            cache_fh.write(json.dumps({'k': '%s/%s/%s' % (ds, cell, iid), 'g': g}, separators=(',', ':')) + '\n')
            cache_fh.flush()
    return out, len(missing)


# ---- per-cell aggregates ---------------------------------------------------------------------
def pct(a, b):
    return round(100.0 * a / b, 2) if b else 0.0


def acc_table(accs):
    """accs: list of per-row records -> {target: {pred: {n, s, top1_call, top3_call, top1_time, top3_time}}}"""
    targets = {'all': lambda r: True, 'slow': lambda r: r['slow'], 'T1': lambda r: r['slow'] and r['cell'] == 5,
               'PASTE': lambda r: r['slow'] and r['cell'] == 4, 'rep': lambda r: r['slow'] and r['cell'] in (1, 2, 3)}
    preds = ('k1', 'k2', 'k3', 'backoff', 'pool_loo', 'pool_xrepo', 'hybrid')
    out = {}
    for tn, sel in targets.items():
        rows = [r for r in accs if sel(r)]
        S = sum(r['dur'] for r in rows)
        out[tn] = {}
        for p in preds:
            if not rows:
                out[tn][p] = {'n': 0, 's': 0.0}
                continue
            out[tn][p] = {'n': len(rows), 's': round(S, 1),
                          'top1_call': round(sum(r[p][0] for r in rows) / len(rows), 4),
                          'top3_call': round(sum(r[p][1] for r in rows) / len(rows), 4),
                          'top1_time': round(sum(r['dur'] for r in rows if r[p][0]) / S, 4) if S else 0.0,
                          'top3_time': round(sum(r['dur'] for r in rows if r[p][1]) / S, 4) if S else 0.0}
    sig = {}
    for tn in ('all', 'slow'):
        rows = [r for r in accs if targets[tn](r)]
        S = sum(r['dur'] for r in rows)
        sig[tn] = {p: ({'n': len(rows), 'top1_call': round(sum(r[p][0] for r in rows) / len(rows), 4),
                        'top1_time': round(sum(r['dur'] for r in rows if r[p][0]) / S, 4) if S else 0.0}
                       if rows else {'n': 0}) for p in ('sig_backoff', 'sig_pool_loo')}
    return out, sig


def markov(seqs, top=15):
    """seqs: list of (labels, durs) per task -> nodes / edges / succ over ADJACENT calls."""
    nodes = defaultdict(lambda: [0, 0.0, 0, 0.0]); edges = defaultdict(lambda: [0, 0.0])
    for labs, durs in seqs:
        for j, l in enumerate(labs):
            nd = nodes[l]; nd[0] += 1; nd[1] += durs[j]
            if durs[j] >= T.SLOW and l != 'wait':
                nd[2] += 1; nd[3] += durs[j]
            if j:
                e = edges[(labs[j - 1], l)]; e[0] += 1; e[1] += durs[j]
    keep = {l for l, _ in sorted(nodes.items(), key=lambda kv: -kv[1][0])[:top] if l != 'wait'} | {'wait'}
    fold = lambda l: l if l in keep else 'other'
    nn = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for l, v in nodes.items():
        f = nn[fold(l)]
        for q in range(4):
            f[q] += v[q]
    ee = defaultdict(lambda: [0, 0.0])
    for (a, b), v in edges.items():
        f = ee[(fold(a), fold(b))]; f[0] += v[0]; f[1] += v[1]
    out_tot = Counter()
    for (a, b), v in ee.items():
        out_tot[a] += v[0]
    succ = defaultdict(list)
    for (a, b), v in sorted(ee.items(), key=lambda kv: -kv[1][0]):
        if len(succ[a]) < 5:
            succ[a].append([b, round(v[0] / out_tot[a], 3), v[0]])
    return {'vocab': 'verb_of(key_cwd core); wait = tmux WAIT/READ/KEYS + sleep-polls',
            'nodes': [[l, v[0], round(v[1], 1), v[2], round(v[3], 1)] for l, v in sorted(nn.items(), key=lambda kv: -kv[1][0])],
            'edges': [[a, b, v[0], round(v[0] / out_tot[a], 3), round(v[1], 1)] for (a, b), v in sorted(ee.items(), key=lambda kv: -kv[1][0])],
            'succ': dict(succ)}


def cell_aggregate(ds, cell, tasks, Xs, accs, seqs, slowrows):
    """everything the taxonomy section, the plots and the census tables need for one cell."""
    span = sum(x['m']['span'] for x in tasks)
    rc = lambda x, k: x['rc'].get(k, 0.0) or 0.0
    A = {}
    # --- taxonomy seconds (uncapped / capped)
    tax = {}
    for name, keys in (('reuse', ('L1',)), ('known', ('L2', 'L3')), ('pred', ('PASTE', 'T1')), ('PASTE', ('PASTE',)), ('T1', ('T1',))):
        u = sum(rc(x, k) for x in tasks for k in keys); c = sum(rc(x, k + 'w') for x in tasks for k in keys)
        tax[name] = {'s': round(u), 'sw': round(c), 'pct': pct(u, span), 'pctw': pct(c, span)}
    tax['reorder'] = {'s': round(sum(rc(x, 'par') for x in tasks)), 'pct': pct(sum(rc(x, 'par') for x in tasks), span)}
    tax['loop_K3'] = {'s': round(sum(T.reclaim(x['rc'], {'loop'}, 3) for x in tasks))}
    tax['parl1'] = round(sum(X['parl1'] for X in Xs.values()), 1)
    tax['cheap'] = {'s': round(sum(X['cheap'][0] for X in Xs.values()), 1), 'n': sum(X['cheap'][1] for X in Xs.values()),
                    's_l1': round(sum(X['cheap'][2] for X in Xs.values()), 1), 'n_l1': sum(X['cheap'][3] for X in Xs.values())}
    A['span'] = round(span); A['tasks'] = len(tasks); A['ntasks_opp'] = len(Xs); A['tax'] = tax
    # --- predicted-speculation decomposition
    dec = {'vn': {k: round(sum(X[c]['vn'][k] for X in Xs.values() for c in PRED_CELLS.values()), 1) for k in VN_KEYS}}
    for c in PRED_CELLS.values():
        dec[c] = {key: [round(sum(X[c][key][q] for X in Xs.values()), 1) for q in (0, 1)] for key in ('g', 'gt', 'ga', 'p')}
        dec[c]['pr'] = [round(sum(X[c]['pr'][q] for X in Xs.values()), 1) for q in range(len(PROBE_C))]
        dec[c]['n'] = [sum(X[c]['n'][q] for X in Xs.values()) for q in range(6)]
        dec[c]['tK'] = [sum(X[c]['tK'][q] for X in Xs.values()) for q in range(KMAX)]
        dec[c]['gK'] = [round(sum(X[c]['gK'][q] for X in Xs.values()), 1) for q in range(KMAX)]
        dec[c]['aK'] = [round(sum(X[c]['aK'][q] for X in Xs.values()), 1) for q in range(KMAX)]
    fams = {f: [0, 0.0] for f in FAM}
    fams['output_path'] = 'n/a (tool_calls.jsonl carries no output text)'
    acs_all, arank_all = [], []
    for row in slowrows:
        fam, valw = row['fam'], row['valw']
        for f, bit in FAM.items():
            if fam & bit:
                fams[f][0] += 1; fams[f][1] += valw
        acs_all.append(row['acs'])
        if row['arank_r']:
            arank_all.append(row['arank_r'])
    for f in FAM:
        fams[f][1] = round(fams[f][1], 1)
    dec['mappers'] = fams
    dec['acs'] = {'median': statistics.median(acs_all) if acs_all else 0, 'p90': (sorted(acs_all)[int(0.9 * (len(acs_all) - 1))] if acs_all else 0),
                  'arank_median': statistics.median(arank_all) if arank_all else 0, 'n_ranked': len(arank_all), 'n': len(acs_all)}
    A['pred_decomp'] = dec
    # --- transition accuracy
    A['acc'], A['acc_sig'] = acc_table(accs)
    A['markov'] = markov(seqs)
    # --- window / lookback
    lb = [round(sum(X['lb'][q] for X in Xs.values()), 1) for q in range(len(LOOKBACK))]
    lbv = [round(sum(X['lbv'][q] for X in Xs.values()), 1) for q in range(len(LOOKBACK))]
    per = {q: [X['lb'][q] / x['m']['span'] for x in tasks if x['m']['span'] > 0 and (X := Xs.get(x['iid'])) is not None] for q in range(len(LOOKBACK))}
    A['lookback'] = {'k': [k if k is not None else 'inf' for k in LOOKBACK], 's': lb, 'pct_span': [pct(v, span) for v in lb],
                     'sv': lbv, 'pctv_span': [pct(v, span) for v in lbv],
                     'per_task_pct': {'median': [round(100 * statistics.median(per[q]), 2) if per[q] else 0 for q in per],
                                      'p90': [round(100 * sorted(per[q])[int(0.9 * (len(per[q]) - 1))], 2) if per[q] else 0 for q in per]}}
    bins = [0] * 11; secs = [0.0] * 11
    DUR_B = [5, 10, 20, 40, 80, 160, float('inf')]; WIN_B = [0, 5, 10, 20, 40, 80, 160, float('inf')]
    heat_n = [[0] * len(WIN_B) for _ in DUR_B]; heat_s = [[0.0] * len(WIN_B) for _ in DUR_B]
    fit_n = fit_s = 0.0; tot_n = tot_s = 0.0
    for row in slowrows:
        dur, win = row['dur'], row['win']
        q = min(10, int(10 * min(dur, win) / dur)) if dur > 0 else 0
        bins[q] += 1; secs[q] += dur
        di = next(k for k, b in enumerate(DUR_B) if dur < b or b == float('inf'))
        wi = next(k for k, b in enumerate(WIN_B) if win < b or b == float('inf'))
        heat_n[di][wi] += 1; heat_s[di][wi] += dur
        tot_n += 1; tot_s += dur
        if win >= dur:
            fit_n += 1; fit_s += dur
    A['gap'] = {'bins': [round(0.1 * q, 1) for q in range(11)], 'calls': bins, 'secs': [round(v, 1) for v in secs],
                'p_fit_call': round(fit_n / tot_n, 4) if tot_n else 0.0, 'p_fit_time': round(fit_s / tot_s, 4) if tot_s else 0.0,
                'heat': {'dur_bins': [b if b != float('inf') else 'inf' for b in DUR_B], 'win_bins': [b if b != float('inf') else 'inf' for b in WIN_B],
                         'n': heat_n, 's': [[round(v, 1) for v in rr] for rr in heat_s]}, 'n_slow_pred': int(tot_n)}
    # --- probe race
    orc = [round(sum(X[c]['pr'][q] for X in Xs.values() for c in PRED_CELLS.values()), 1) for q in range(len(PROBE_C))]
    A['probe'] = {'c': list(PROBE_C), 'oracle_s': orc, 'share': [round(v / orc[0], 4) if orc[0] else 0.0 for v in orc],
                  'model': {'%.2f' % p: [round(p * v, 1) for v in orc] for p in MODEL_P}}
    # --- speedup under predictors (window-capped spine; +par variants)
    def f_pred(x, X, c, how):
        if how == 'oracle':
            return rc(x, c + 'w')
        if how == 'graph':
            return X[c]['g'][1] if X else None
        if how == 'pool':
            return X[c]['p'][1] if X else None
        if how.startswith('model'):
            p = float(how[5:9]); pc = int(how.split('pc')[1]) if 'pc' in how else 0
            base = rc(x, c + 'w') if pc == 0 else (X[c]['pr'][pc] if X else None)
            return None if base is None else p * base
        raise ValueError(how)
    sp = {}
    for how in ('oracle', 'graph', 'pool', 'model0.30', 'model0.44', 'model0.30pc2', 'model0.44pc2'):
        for par in (False, True):
            pairs = []
            miss = 0
            for x in tasks:
                X = Xs.get(x['iid'])
                fp, ft = f_pred(x, X, 'PASTE', how), f_pred(x, X, 'T1', how)
                if fp is None or ft is None:
                    miss += 1; continue
                r = rc(x, 'L1w') + rc(x, 'L2w') + rc(x, 'L3w') + fp + ft + (rc(x, 'par') if par else 0.0)
                pairs.append((x['m']['span'], r))
            st = T.stats(pairs); st['miss'] = miss
            sp[how + ('+par' if par else '')] = st
    A['speedup'] = sp
    # --- verbs: alone / marginal / share (window-capped)
    V = {'reuse': lambda x: rc(x, 'L1w'), 'known': lambda x: rc(x, 'L2w') + rc(x, 'L3w'),
         'pred': lambda x: rc(x, 'PASTEw') + rc(x, 'T1w'), 'reorder': lambda x: rc(x, 'par')}
    allf = lambda x: sum(f(x) for f in V.values())
    agg_all = T.stats([(x['m']['span'], allf(x)) for x in tasks])
    verbs = {'all': agg_all}
    tot_s_all = sum(allf(x) for x in tasks)
    for name, f in V.items():
        alone = T.stats([(x['m']['span'], f(x)) for x in tasks])
        without = T.stats([(x['m']['span'], allf(x) - f(x)) for x in tasks])
        verbs[name] = {'alone': alone, 'without': without,
                       'marginal_agg': round((agg_all.get('agg', 1) - without.get('agg', 1)), 4),
                       'share': round(sum(f(x) for x in tasks) / tot_s_all, 4) if tot_s_all else 0.0}
    A['verbs'] = verbs
    # --- S/A
    sa = {grp: [round(sum(X['sa'][grp][0] for X in Xs.values()), 1), round(sum(X['sa'][grp][1] for X in Xs.values()), 1)] for grp in ('l1', 'known', 'pred', 'predT')}
    sa['smeas_tasks'] = sum(X['smeas'] for X in Xs.values()); sa['pct_S_known_pred'] = pct(sa['known'][0] + sa['pred'][0], span)
    A['sa'] = sa
    return A


def repo_aggregates(tasks, Xs, accs_by_task, seqs_by_task, slow_by_task):
    out = {}
    byrepo = defaultdict(list)
    for x in tasks:
        byrepo[x['repo']].append(x['iid'])
    for repo, iids in byrepo.items():
        accs = [r for iid in iids for r in accs_by_task.get(iid, [])]
        rows = [r for r in accs if r['slow']]
        S = sum(r['dur'] for r in rows)
        mk = markov([seqs_by_task[iid] for iid in iids if iid in seqs_by_task], top=10)
        sl = [r for iid in iids for r in slow_by_task.get(iid, [])]
        ts = sum(r['dur'] for r in sl)
        out[repo] = {'tasks': len(iids), 'succ': mk['succ'],
                     'acc': {p: {'top1_time': round(sum(r['dur'] for r in rows if r[p][0]) / S, 4) if S else 0.0,
                                 'top3_time': round(sum(r['dur'] for r in rows if r[p][1]) / S, 4) if S else 0.0}
                             for p in ('backoff', 'pool_loo', 'pool_xrepo')},
                     'gap': {'p_fit_time': round(sum(r['dur'] for r in sl if r['win'] >= r['dur']) / ts, 4) if ts else 0.0, 'n': len(sl)}}
    return out


# ---- census markdown --------------------------------------------------------------------------
def write_census(out, cells):
    md = ['# TAG CPU-side opportunity census (build_oppdata.py over the priced CLEAN_0905 graphs)\n',
          'Window-capped seconds unless a column says uncapped. reuse = L1; known-identity speculation = L2+L3; '
          'predicted speculation = PASTE (identity known, trigger turn must be predicted) + T1 (identity must be predicted); '
          'reorder = read-parallelism. TYPE = in-task n-gram top-1 over verb labels (backoff k=3..1), ARGS = an argument '
          'mapper (const / repeat / copy_path) reconstructs the exact identity from the task prefix, WINDOW = preceding LLM gap >= duration. '
          'Pools are leave-one-task-out (pool_loo) or leave-repo-out (pool_xrepo) within the cell and are leaky on a 10-repo corpus. '
          'output_path is not computable here (no tool output in tool_calls.jsonl). cheap = predicted-speculation seconds on '
          'echo/ls/cat/... (under tmux their latency is the previous program\'s tail). Missing != zero: a cell absent here was not built.\n']
    md.append('## 1. Taxonomy per cell (s = uncapped, sw = window-capped, % of span)\n')
    md.append('| dataset | cell | tasks | span s | reuse L1 s / sw / %w | known L2+L3 s / sw / %w | predicted PASTE+T1 s / sw / %w | PASTE sw | T1 sw | reorder s / % | par∩L1 sw | cheap pred sw (n) | cheap L1 sw (n) | loop K3 s (not summed) |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        t = A['tax']
        md.append('| %s | %s | %d | %d | %d / %d / %.2f | %d / %d / %.2f | %d / %d / %.2f | %d | %d | %d / %.2f | %.0f | %.0f (%d) | %.0f (%d) | %d |' % (
            ds, cell, A['tasks'], A['span'], t['reuse']['s'], t['reuse']['sw'], t['reuse']['pctw'], t['known']['s'], t['known']['sw'], t['known']['pctw'],
            t['pred']['s'], t['pred']['sw'], t['pred']['pctw'], t['PASTE']['sw'], t['T1']['sw'], t['reorder']['s'], t['reorder']['pct'],
            t['parl1'], t['cheap']['s'], t['cheap']['n'], t['cheap'].get('s_l1', 0), t['cheap'].get('n_l1', 0), t['loop_K3']['s']))
    md.append('\n## 2. Predicted speculation: what a predictor must get right (window-capped s; T=type A=args F=window)\n')
    md.append('| dataset | cell | pred sw | TAF | TA¬F | T¬AF | T¬A¬F | ¬TAF | ¬TA¬F | ¬T¬AF | ¬T¬A¬F | type-only (gt−g) | type+args (g) | args given type (ga) | pool T∧A (p) | const n/s | repeat n/s | copy_path n/s | copy_path_ub n/s | acs median / p90 | arank median |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        d = A['pred_decomp']; vn = d['vn']; m = d['mappers']
        g = sum(d[c]['g'][1] for c in PRED_CELLS.values()); gt = sum(d[c]['gt'][1] for c in PRED_CELLS.values())
        ga = sum(d[c]['ga'][1] for c in PRED_CELLS.values()); p = sum(d[c]['p'][1] for c in PRED_CELLS.values())
        md.append('| %s | %s | %d | %s | %.0f | %.0f | %.0f | %.0f | %d/%.0f | %d/%.0f | %d/%.0f | %d/%.0f | %s / %s | %s |' % (
            ds, cell, A['tax']['pred']['sw'], ' | '.join('%.0f' % vn[k] for k in ('111', '110', '101', '100', '011', '010', '001', '000')),
            gt - g, g, ga, p, m['const'][0], m['const'][1], m['repeat'][0], m['repeat'][1], m['copy_path'][0], m['copy_path'][1],
            m['copy_path_ub'][0], m['copy_path_ub'][1], d['acs']['median'], d['acs']['p90'], d['acs']['arank_median']))
    md.append('\n## 3. Transition-frequency predictors: top-1 / top-3, time-weighted on slow calls (call-weighted in census.json)\n')
    md.append('| dataset | cell | slow n | k=1 | k=2 | k=3 | backoff (in-task) | pool_loo | pool_xrepo | hybrid | signature_of backoff | signature_of pool_loo | first-sight T1 backoff top1 | PASTE backoff top1 |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        a = A['acc']['slow']; s = A['acc_sig']['slow']
        f = lambda p: '%.0f / %.0f' % (100 * a[p]['top1_time'], 100 * a[p]['top3_time']) if a[p].get('n') else '-'
        md.append('| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ds, cell, a['backoff'].get('n', 0), f('k1'), f('k2'), f('k3'), f('backoff'), f('pool_loo'), f('pool_xrepo'), f('hybrid'),
            ('%.0f' % (100 * s['sig_backoff']['top1_time'])) if s['sig_backoff'].get('n') else '-',
            ('%.0f' % (100 * s['sig_pool_loo']['top1_time'])) if s['sig_pool_loo'].get('n') else '-',
            ('%.0f' % (100 * A['acc']['T1']['backoff']['top1_time'])) if A['acc']['T1']['backoff'].get('n') else '-',
            ('%.0f' % (100 * A['acc']['PASTE']['backoff']['top1_time'])) if A['acc']['PASTE']['backoff'].get('n') else '-'))
    md.append('\n## 4. The window: P(gap ≥ dur) and the k-turn lookback (% of span; v = validity-aware, capped by dataflow slack)\n')
    md.append('| dataset | cell | slow pred n | P(fit) calls | P(fit) time | lb k=1 | k=2 | k=3 | k=5 | k=10 | ∞ | lbv k=1 | k=2 | k=3 | k=5 | k=10 | ∞ | per-task median % k=1 / ∞ | p90 % k=1 / ∞ |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        L = A['lookback']; G = A['gap']
        md.append('| %s | %s | %d | %.2f | %.2f | %s | %s | %.2f / %.2f | %.2f / %.2f |' % (
            ds, cell, G['n_slow_pred'], G['p_fit_call'], G['p_fit_time'], ' | '.join('%.2f' % v for v in L['pct_span']),
            ' | '.join('%.2f' % v for v in L['pctv_span']), L['per_task_pct']['median'][0], L['per_task_pct']['median'][-1],
            L['per_task_pct']['p90'][0], L['per_task_pct']['p90'][-1]))
    md.append('\n## 5. Probe race: predicted-speculation seconds that survive a probe costing c seconds (oracle identity)\n')
    md.append('| dataset | cell | c=0 s | c=1.2 | c=3.3 | c=5.0 | c=9.7 | share 1.2 | 3.3 | 5.0 | 9.7 |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        P = A['probe']
        md.append('| %s | %s | %s | %s |' % (ds, cell, ' | '.join('%d' % v for v in P['oracle_s']), ' | '.join('%.2f' % v for v in P['share'][1:])))
    md.append('\n## 6. Predictor-composed speedup (window-capped spine; aggregate / mean / geomean); pc2 = probe cost 3.3 s\n')
    md.append('| dataset | cell | oracle | graph n-gram | graph pool (leaky) | model p=0.30 | model p=0.44 | model 0.30 @3.3 s | model 0.44 @3.3 s | oracle +par | reuse alone / marginal | known alone / marginal | pred alone / marginal | reorder alone / marginal |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        sp = A['speedup']; V = A['verbs']
        f = lambda k: '%.3f / %.3f / %.3f' % (sp[k]['agg'], sp[k]['mean'], sp[k]['geomean']) if sp[k].get('n') else '-'
        v = lambda k: '%.3f / %+.3f' % (V[k]['alone'].get('agg', 1), V[k]['marginal_agg'])
        md.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ds, cell, f('oracle'), f('graph'), f('pool'), f('model0.30'), f('model0.44'), f('model0.30pc2'), f('model0.44pc2'), f('oracle+par'),
            v('reuse'), v('known'), v('pred'), v('reorder')))
    md.append('\n## 7. S/A split: uncapped S-stage seconds (survive edits) vs window-capped A-stage, INCR/TESTP families only\n')
    md.append('| dataset | cell | S known (L2+L3) | A known | S pred (PASTE+T1) | A pred | S pred with TYPE named (hint→S prewarm) | S L1 | S known+pred % span | tasks with measured S-share |')
    md.append('|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), A in cells.items():
        s = A['sa']
        md.append('| %s | %s | %d | %d | %d | %d | %d | %d | %.2f | %d/%d |' % (ds, cell, s['known'][0], s['known'][1], s['pred'][0], s['pred'][1], s.get('predT', [0, 0])[0], s['l1'][0],
                                                                             s['pct_S_known_pred'], s['smeas_tasks'], A['ntasks_opp']))
    open(T.guard_out(os.path.join(out, 'TAG_OPP_CENSUS.md')), 'w').write('\n'.join(md) + '\n')
    slim = {}
    for (ds, cell), A in cells.items():
        B = dict(A); B['markov'] = {'nodes': A['markov']['nodes'][:8], 'vocab': A['markov']['vocab']}   # census stays small; full markov in _cell.json
        B.pop('repos', None)
        slim['%s/%s' % (ds, cell)] = B
    json.dump({'meta': {'built': time.strftime('%Y-%m-%d %H:%M'), 'probe_c': list(PROBE_C), 'lookback': [k if k is not None else 'inf' for k in LOOKBACK]},
               'groups': slim}, open(T.guard_out(os.path.join(out, 'census.json')), 'w'), indent=1)


# ---- main ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    T.add_common_args(ap)
    ap.add_argument('--tag', default=os.path.join(T.EXPL, 'data', 'tag'))
    ap.add_argument('--out', default=os.path.join(T.EXPL, 'data', 'opp'))
    ap.add_argument('--cells', default='', help='comma list of cells (<mode>__<fw>) to build; default all in the index')
    ap.add_argument('--force', action='store_true', help='rebuild cells whose _cell.json is newer than the tag data')
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    os.makedirs(T.guard_out(a.out), exist_ok=True)
    tag_index = json.load(open(os.path.join(a.tag, 'index.json')))
    idx_path = os.path.join(a.out, 'index.json')
    try:
        index = json.load(open(idx_path))
    except (OSError, ValueError):
        index = {'meta': {}, 'tasks': {}}
    index['meta'] = {'v': 1, 'built': time.strftime('%Y-%m-%d %H:%M'), 'vocab': 'verb_of(key_cwd core)',
                     'probe': {'default': 0.30, 'marks': list(MODEL_P)}, 'probe_c': list(PROBE_C),
                     'lookback': [k if k is not None else 'inf' for k in LOOKBACK], 'kmax': KMAX,
                     'cells': list(PRED_CELLS.values()),
                     'keys': {'g': 'TYPE∧ARGS [s, sw]', 'gt': 'TYPE [s, sw]', 'ga': 'ARGS given true type [s, sw]', 'p': 'pool_loo TYPE∧ARGS [s, sw]',
                              'pr': 'oracle sw under probe cost probe_c', 'vn': 'TYPE ARGS FIT bits -> sw', 'n': '[n, nT, nA, nF, nTA, nTAF]',
                              'tK': 'true type within top-K (counts)', 'gK': 'sw with type in top-K ∧ ARGS', 'aK': 'sw with true identity within K most recent same-label identities',
                              'lb': 'Σ min(dur, W_k) over slow rows, k = lookback', 'lbv': 'same, also capped by dataflow slack', 'cheap': '[sw pred, n pred, sw L1, n L1] on echo/ls/cat/...',
                              'sa': 'S-stage (uncapped) / A-stage (capped) s by cell group', 'parl1': 'sw of L1 rows inside read-par runs'}}
    # cache
    cache, cache_fh = {}, None
    cpath = os.path.join(a.out, '.tagcache.jsonl')
    if not a.no_cache:
        if os.path.exists(cpath):
            t0 = time.time()
            for l in open(cpath):
                if l.strip():
                    o = json.loads(l); cache[o['k']] = o['g']
            print('cache: %d tasks in %.1f s' % (len(cache), time.time() - t0))
        cache_fh = open(T.guard_out(cpath), 'a')
    want = set(x for x in a.cells.split(',') if x)
    dss = [d.strip() for d in a.datasets.split(',') if d.strip()]
    built = 0
    for ds in dss:
        if ds not in tag_index:
            print('skip %s: not in tag index' % ds); continue
        for cell, rows in tag_index[ds].items():
            if want and cell not in want:
                continue
            outdir = T.guard_out(os.path.join(a.out, ds, cell)); os.makedirs(outdir, exist_ok=True)
            cellf = os.path.join(outdir, '_cell.json')
            tagdir = os.path.join(a.tag, ds, cell)
            if (not a.force) and os.path.exists(cellf) and os.path.getmtime(cellf) > os.path.getmtime(tagdir) \
                    and cell in index['tasks'].get(ds, {}):
                print('%-24s %-22s up to date (use --force)' % (ds, cell)); continue
            t0 = time.time()
            tasks, nread = load_cell(a.tag, ds, cell, rows, cache, cache_fh)
            # pass 1: labels + final grams; cell / repo totals
            L, cell_lab, cell_sig = {}, empty_counts(), empty_counts()
            repo_lab = defaultdict(empty_counts)
            for x in rows:
                g = tasks.get(x['iid'])
                if g is None:
                    continue
                labs, sigs, ids = task_labels(g)
                gl, gs = final_grams(labs, sigs, ids)
                L[x['iid']] = (labs, sigs, ids, gl, gs)
                add_counts(cell_lab, gl); add_counts(cell_sig, gs); add_counts(repo_lab[g['repo']], gl)
            # pass 2: annotate
            Xs, accs_by, seqs_by, slow_by = {}, {}, {}, {}
            for x in rows:
                if x['iid'] not in L:
                    continue
                g = tasks[x['iid']]; labs, sigs, ids, gl, gs = L[x['iid']]
                pool_loo = PoolGram(cell_lab, gram_counts(gl))
                pool_xrepo = PoolGram(cell_lab, repo_lab[g['repo']])
                sigpool = PoolGram(cell_sig, gram_counts(gs))
                X, pv, acc = annotate(g, labs, sigs, ids, pool_loo, pool_xrepo, sigpool)
                Xs[x['iid']] = X; accs_by[x['iid']] = acc
                seqs_by[x['iid']] = (labs, [r[2] for r in g['t']])
                slow_by[x['iid']] = [{'dur': g['t'][row[0]][2], 'win': g['t'][row[0]][9], 'valw': g['t'][row[0]][10], 'fam': row[6],
                                      'acs': row[10], 'arank_r': row[11]} for row in pv]
                vocab = sorted(set(labs)); vid = {l: q for q, l in enumerate(vocab)}
                json.dump({'v': 1, 'iid': x['iid'], 'ds': ds, 'cell': cell, 'lab': vocab, 'L': [vid[l] for l in labs],
                           'pv': pv, 'tr': [[vid[labs[j - 1]], vid[labs[j]]] for j in range(1, len(labs))],
                           'cols': ['i', 'pred_label', 'k_used', 'top1', 'top3', 'top1_pool', 'fam', 'args', 'fit', 'frac', 'acs', 'arank_r', 'arank_f']},
                          open(os.path.join(outdir, x['iid'] + '.json'), 'w'), separators=(',', ':'))
            A = cell_aggregate(ds, cell, [x for x in rows if x['iid'] in Xs], Xs, [r for v in accs_by.values() for r in v],
                               list(seqs_by.values()), [r for v in slow_by.values() for r in v])
            A['repos'] = repo_aggregates([x for x in rows if x['iid'] in Xs], Xs, accs_by, seqs_by, slow_by)
            A['ds'], A['cell'], A['built'] = ds, cell, time.strftime('%Y-%m-%d %H:%M')
            json.dump(A, open(cellf, 'w'), separators=(',', ':'))
            index['tasks'].setdefault(ds, {})[cell] = Xs
            json.dump(index, open(T.guard_out(idx_path), 'w'), separators=(',', ':'))
            built += 1
            sp = A['speedup']
            print('%-24s %-22s tasks=%3d read=%3d  pred sw=%6d  g=%6.0f gt=%6.0f ga=%6.0f  slow-top1 backoff=%.2f pool=%.2f  '
                  'oracle %.3f graph %.3f model.30 %.3f  lb1=%.2f%%  %.0fs'
                  % (ds, cell, len(Xs), nread, A['tax']['pred']['sw'],
                     sum(A['pred_decomp'][c]['g'][1] for c in PRED_CELLS.values()), sum(A['pred_decomp'][c]['gt'][1] for c in PRED_CELLS.values()),
                     sum(A['pred_decomp'][c]['ga'][1] for c in PRED_CELLS.values()),
                     A['acc']['slow']['backoff'].get('top1_time', 0), A['acc']['slow']['pool_loo'].get('top1_time', 0),
                     sp['oracle'].get('agg', 0), sp['graph'].get('agg', 0), sp['model0.30'].get('agg', 0), A['lookback']['pct_span'][0], time.time() - t0))
    if cache_fh:
        cache_fh.close()
    # census over every _cell.json present (not just this run's)
    cells = {}
    for ds in sorted(os.listdir(a.out)):
        dsd = os.path.join(a.out, ds)
        if not os.path.isdir(dsd) or ds.startswith('.'):
            continue
        for cell in sorted(os.listdir(dsd)):
            f = os.path.join(dsd, cell, '_cell.json')
            if os.path.exists(f):
                cells[(ds, cell)] = json.load(open(f))
    if cells:
        write_census(a.out, cells)
    print('wrote', a.out, ': index.json (%d cells), census over %d cells; built %d this run'
          % (sum(len(v) for v in index['tasks'].values()), len(cells), built))


if __name__ == '__main__':
    main()
