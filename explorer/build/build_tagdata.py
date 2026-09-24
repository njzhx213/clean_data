#!/usr/bin/env python3
"""Build per-task Tool Analysis Graphs (TAG) from the CLEAN_0905 profiling results
into explorer/data/tag/ -- the execution dependency graph of build_execdata.py
re-read along the three TAG axes (bench_agentic/TAG_ARGUMENT.md; definitions in
cshih63/ASPLOS2027/TOOL_ANALYSIS_GRAPH.md s2.2/s4/s5/s9; census rules copied from
m4b/build_bipartite.py, identity policy from m4b/m4blib.py, langgraph tool rules from
m4b/build_arm_graphs.py -- cited inline, not imported across trees).

Input (and nothing else from the portal):
  <root>/<dataset>/<mode>/<fw>/traj/<iid>/swe_bench_pro__tool_calls.jsonl
  bench_agentic/subsets/*.norm*.jsonl   (canonical task ids + default cwd; repo = manifest repo,
                                         or the task.toml category for Terminal-Bench-2)
A traj dir is used iff its name is a manifest instance_id (archived reruns carry a
__suffix and are skipped).  Harness bookkeeping rows inside hermes logs (git strip /
ancestry gate / fileMode / baseline ref / patch capture / TB2 gate + netguard) are
removed by BOOK wherever they occur (they repeat on continuation restarts).  2026-09-08: the runners' patch
capture became `git -C /app -c core.fileMode=false add -A` -- the optional `(?: -c \S+)*` keeps that row a harness row
(CLEAN_0905 logs never carried `-c`, so the CLEAN census is unchanged).

Per task (rows sorted by start_s, i = call index):
  transition node = one invocation (tool, identity); state node = (resource, version)
  with availability = its writer's earliest finish; read edge state->transition,
  write edge transition->state(v+1).  est = max avail over reads, slack = issue-est,
  cp = max(est+dur).  identity: bash -> ('bash', cwd-folded core); read_file/grep/glob/
  ls/edit_file per rw_and_identity; tmux keystrokes (WAIT/READ/KEYS/TYPE/C-x/empty) and
  sleep-polling (`sleep N`, optionally followed by a cheap peek) -> class Wait (no R/W, no
  cell, summed as wait_s).
  cells (slow >= 5 s, mutually exclusive):
    L1 dup>=0 (same identity AND same reaching versions)      value = dur
    L2 identity recurs, incremental-build family (INCR)       value = S-share x dur
    L3 identity recurs, test/process family (TESTP)           value = min(slack, dur)
    PASTE identity recurs, other                              value = min(slack, dur)
    T1 first-sight identity                                   value = dur if slack>=dur else slack
  Track-C signatures (TAG_ARGUMENT s3; numbers recomputed on this data, s6):
    loop-guard   consecutive same-identity non-mutating runs; per surplus turn we keep
                 the unclaimed tool seconds and the LLM gap before it, so the viewer can
                 charge turns beyond any K (K = identical results shown before warning)
    prefill      re-reads of an unchanged (resource, version): count only (pricing needs
                 the LLM parquet, excluded by decision)
    read-par     adjacent independent read-class runs: sum(lat)-max(lat) minus the overlap
                 the scaffold already realised with parallel tool calls
  Predicted e2e ceiling = span / (span - reclaim), span = first tool start -> last tool
  end (omits the leading/trailing LLM turns: slightly optimistic), reclaim capped at
  0.95 span, no double counting between cells and loop-guard.
Usage: python build_tagdata.py [--root DIR] [--datasets a,b] [--out DIR]
"""
import argparse, ast, json, math, os, re, statistics, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EXPL = os.path.dirname(HERE)
ROOT = '/eic/data/haoxiong_gpu6/bench_agentic/H200_profiling_portal/clean_data'   # the final clean copy (build_clean.py)
SUBSETS = '/eic/data/haoxiong_gpu6/bench_agentic/H200_profiling_portal/clean_data/task_sets'
BENCH = '/eic/data/haoxiong_gpu6/bench_agentic'
# A/B campaign trajectories: <arm>/<pro|tb2>/<mode>/<fw>/traj/<iid>.  The campaign RUNS off
# local disk (tech_worker.local.sh:52 RES=/tmp/tech/results) and ops/archive_traces.sh copies
# the same relative layout to final_test/traces on NFS -- that archive is the durable copy and
# is therefore the default here.  To build off a run that is still going (or a card-local
# validation run) pass --camp-root /tmp/tech/results.  technique/results holds only ledger.tsv
# and progress.log (sync_back.sh:43-45), never a trajectory, so it must NOT be the default.
CAMP = os.path.join(BENCH, 'final_test', 'traces')
CAMP_LIVE = '/tmp/tech/results'                      # the same layout, on the worker's local disk
FORBID = '/eic/data/haoxiong_gpu6/tool_speculation'  # shared tree (cshih63): never written to
DATASETS = {  # dataset key == directory under clean/; manifest (canonical ids + cwd) in clean/task_sets
    'pro_250': 'pro_250.official_prompt.jsonl',
    'tb2_89': 'tb2_89.norm.jsonl',
    'swebench_verified_200': 'swebench_verified_200.norm.jsonl',
    'swebench_lite_100': 'swebench_lite_100.norm.jsonl',
}
DSDIR = {}   # clean/: key == dir
_DSDIR_OLD = {'grid12v7_pro': 'pro', 'final_pro': 'pro', 'final_tb2': 'tb2', 'grid12_pro': 'pro', 'grid12t0_pro': 'pro', 'grid12v2_pro': 'pro', 'grid12v2c_pro': 'pro', 'grid12v3_pro': 'pro', 'grid12v3c_pro': 'pro', 'grid12v4_pro': 'pro', 'grid12v4r_pro': 'pro', 'grid12v6rec_pro': 'pro', 'grid12v6_pro': 'pro'}    # dataset key -> directory name on disk
DSROOT = {}
_DSROOT_OLD = {'grid12v7_pro': CAMP, 'final_pro': CAMP, 'final_tb2': CAMP, 'grid12_pro': CAMP, 'grid12t0_pro': CAMP, 'grid12v2_pro': CAMP, 'grid12v2c_pro': CAMP, 'grid12v3_pro': CAMP, 'grid12v3c_pro': CAMP, 'grid12v4_pro': CAMP, 'grid12v4r_pro': CAMP, 'grid12v6rec_pro': CAMP, 'grid12v6_pro': CAMP}     # dataset key -> results root (else --root)
# where a cell's traj dir sits under its root.  The campaign adds the arm dimension in front,
# because tech_worker.local.sh:113,152 writes $RES/$ARM/$DS/$MODE/$SC/traj.
TPL = '{dsdir}/{mode}/{fw}/traj'
DSTPL = {}
_DSTPL_OLD = {'final_pro': '{arm}/' + TPL, 'final_tb2': '{arm}/' + TPL, 'grid12_pro': '{arm}/' + TPL, 'grid12t0_pro': '{arm}/' + TPL,
         'grid12v2_pro': '{arm}/' + TPL, 'grid12v2c_pro': '{arm}/' + TPL, 'grid12v3_pro': '{arm}/' + TPL, 'grid12v3c_pro': '{arm}/' + TPL,
         'grid12v4_pro': '{arm}/' + TPL, 'grid12v4r_pro': '{arm}/' + TPL,
         'grid12v6rec_pro': '{arm}/' + TPL, 'grid12v6_pro': '{arm}/' + TPL, 'grid12v7_pro': '{arm}/' + TPL}
ARMS = ('base', 'full')
# 2026-09-09: the six-arm Duet campaign.  Its arms are whatever ops/arms.tsv lists (the one source of
# truth both runners read), so DSARMS names the table rather than copying it; a dataset absent here
# keeps the legacy base/full pair.  An explicit --arms still overrides every dataset (tests, smoke runs).
ARMS_TSV = next((p for p in (os.path.join(os.path.dirname(EXPL), 'ops', 'arms.tsv'),
                             os.path.join(BENCH, 'technique', 'ops', 'arms.tsv')) if os.path.exists(p)), None)


def arms_tsv(path=None):
    """arm tokens of ops/arms.tsv in table order (column 1 of every non-comment row; base/full are not in it)."""
    path = path or ARMS_TSV
    if not path:
        raise SystemExit('ops/arms.tsv not found next to the explorer (the six-arm datasets take their arms from it)')
    return tuple(l.split('\t')[0].strip() for l in open(path) if l.strip() and not l.startswith('#'))


V7ARMS = ('serial', 'tagspecj', 'ngram', 'naivej', 'duetalgja', 'fullduetja')   # 2026-09-09 v7: the six arms of the n-gram-engine campaign (arms.tsv holds 15 tokens incl. study arms)
DSARMS = {}
_DSARMS_OLD = {'grid12v4_pro': arms_tsv, 'grid12v4r_pro': arms_tsv, 'grid12v6rec_pro': arms_tsv, 'grid12v6_pro': arms_tsv, 'grid12v7_pro': V7ARMS}   # dataset key -> arm tuple, or a callable returning one; absent -> ARMS (v4r/v6 = replay passes, same six arms; v6rec holds only serial on disk, absent arms are tolerated)


def arms_of(ds, cli=None):
    """the arms iter_cells walks for a dataset: --arms when given ('' = no arm dimension), else DSARMS / ARMS."""
    if cli is not None:
        return [x for x in cli.split(',') if x] or [None]
    v = DSARMS.get(ds, ARMS)
    return list(v() if callable(v) else v) or [None]


MODES = ('thinking', 'instruct', 'coder')
FWS = ('hermes', 'langgraph')
TC = 'swe_bench_pro__tool_calls.jsonl'
SLOW = 5.0
CELLS = ['fast', 'L1', 'L2', 'L3', 'PASTE', 'T1']
FNS = ['Read', 'Edit', 'Execute', 'Setup', 'Navigate', 'Other', 'Wait']
TOOLS = ['bash', 'read_file', 'edit_file', 'grep', 'glob', 'ls', 'wait', 'other']
K_TABLE = (3, 8)


# ---- rw_of lifted from build_graphdata.py without running its pipeline (m4blib.py:26-34 trick)
def _lift_rw_of():
    src = os.path.join(HERE, 'build_graphdata.py')
    tree = ast.parse(open(src).read())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.FunctionDef)):
            keep.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('PATH', 'REDIR')):
            keep.append(node)   # the two regex constants rw_of uses; nothing that runs the pipeline
    tree.body = keep
    ns = {'__file__': src}
    exec(compile(tree, src, 'exec'), ns)
    return ns['rw_of']


rw_of = _lift_rw_of()

# harness bookkeeping inside the tool log (validated on every hermes cell: prefix 4-7 rows,
# plus trailing patch capture / netguard removal / prefix repeated on restarts)
BOOK = re.compile(r"^\s*true\s*$|git remote remove|git tag -l|git (fetch|for-each-ref|symbolic-ref)\b"
                  r"|git rev-parse --git-dir|git -C \S+(?: -c \S+)* (config core\.fileMode|update-ref refs/ws/baseline|add -A|diff --cached)"
                  r"|^git reset --hard [0-9a-f]{7,}|rm -rf \S*/test_outputs\.py|ws-netguard|\.ws_hosts"
                  r"|-name [\"']solve\.sh[\"']")
# tmux keystroke protocol (TB2 tmux arm): waiting / reading the screen, not a command
WAIT = re.compile(r"^\s*(?:WAIT\b.*|READ\b.*|KEYS\b.*|TYPE\b.*|C-[a-z]>?|)\s*$")
# sleep-polling: `sleep N` alone or followed only by a cheap peek (cat/tail/ps/ls/...) -- the agent
# waiting on a background process through bash, same class as the tmux WAIT keystroke.
POLL = re.compile(r"^\s*sleep\s+\d+\S*\s*(?:(?:&&|;|\|\|)\s*(?:cat|tail|head|ps|ls|jobs|echo|grep|wc|test|\[|true|date|find|du|stat|nvidia-smi|curl|pgrep|kill|wait)\b[^\n]*)?\s*$", re.S)

# ---- identity = (cwd, core): m4blib.eff_cwd / key_cwd, verbatim semantics (live_spork _eff_cwd)
_CD_PFX = re.compile(r"^cd\s+(?P<t>[^\s&|;<>]+)\s*&&\s*")
_TMO = re.compile(r"^\s*timeout\s+\d+\S*\s+")
_ENV = re.compile(r"^\s*(?:export\s+[A-Za-z_]\w*=\S+\s*&&\s*|[A-Za-z_]\w*=\S+\s+)+")


def eff_cwd(core, cwd):
    cwd = (cwd or '/').rstrip('/') or '/'
    while True:
        m = _CD_PFX.match(core)
        if not m:
            return cwd, core
        t = m.group('t').strip('"\'')
        if not t or t.startswith(('-', '$', '~')):
            return cwd, core
        if t != '.':
            cwd = t if t.startswith('/') else cwd.rstrip('/') + '/' + t
            cwd = re.sub(r'/\./', '/', cwd).rstrip('/') or '/'
        core = core[m.end():]


def key_cwd(c, cwd):
    cwd, rest = eff_cwd(c.strip(), cwd)
    for _ in range(3):
        rest = _TMO.sub('', rest)
    rest = re.split(r'\s*\|\s*(?!\|)', rest)[0]
    rest = re.sub(r'\s*2>&1\s*$', '', rest).strip()
    return cwd, rest


def parse_query(q):
    """langgraph rows carry a python-repr dict; ~17% of edit_file queries do not literal_eval."""
    try:
        d = ast.literal_eval(q)
        return d if isinstance(d, dict) else {}
    except Exception:
        d = {}
        for k in ('command', 'file_path', 'path', 'pattern'):
            m = (re.search(r"'%s': '((?:[^'\\]|\\.)*)'" % k, q, re.S)
                 or re.search(r"'%s': \"((?:[^\"\\]|\\.)*)\"" % k, q, re.S))
            if m:
                d[k] = m.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
        return d


def rw_and_identity(tool, args, cwd):
    """(reads, writes, identity, fn class, display text) -- build_arm_graphs.rw_and_identity
    extended with the tmux Wait class."""
    if tool == 'bash':
        cmd = str(args.get('command') or '')
        if WAIT.match(cmd) or POLL.match(_ENV.sub('', cmd)):
            return set(), set(), ('wait', re.sub(r'\d+', 'N', cmd.strip())[:80]), 'Wait', (cmd.strip() or '(empty keystroke)')
        R, W, fn = rw_of(_ENV.sub('', cmd))
        return set(R), set(W), ('bash',) + key_cwd(cmd, cwd), fn, cmd
    if tool == 'read_file':
        p = str(args.get('file_path') or '')
        return ({p} if p else {'wt'}), set(), ('read', p), 'Read', 'read_file %s' % p
    if tool == 'grep':
        p = str(args.get('path') or ''); pat = str(args.get('pattern') or '')
        return ({p} if p else {'wt'}), set(), ('grep', pat, p), 'Read', 'grep %r %s' % (pat, p)
    if tool == 'glob':
        pat = str(args.get('pattern') or '')
        return {'wt'}, set(), ('glob', pat), 'Read', 'glob %r' % pat
    if tool == 'ls':
        p = str(args.get('path') or '')
        return {p or 'wt'}, set(), ('ls', p), 'Read', 'ls %s' % p
    if tool == 'edit_file':
        p = str(args.get('file_path') or '')
        return ({p} if p else set()), ({p, 'wt'} if p else {'wt'}), ('edit', p), 'Edit', 'edit_file %s' % p
    return {'wt'}, set(), (tool or '?',), 'Other', '%s %s' % (tool, json.dumps(args)[:200])


# ---- census families and S-share (build_bipartite.py:37-48; TOOL_ANALYSIS_GRAPH.md s5)
INCR = re.compile(r"\b(tsc|go\s+(build|vet)|cargo\s+check|mypy)\b")
TESTP = re.compile(r"\b(pytest|go\s+test|npm\s+(test|run)|jest|mocha|vitest|make\s+test|python[0-9.]*\s+-m\s+pytest|node\s+\S*test)\b")
S_SHARE = {"vuls": 0.99, "flipt": 0.99, "navidrome": 0.99, "teleport": 0.99,
           "NodeBB": 0.30, "tutanota": 0.62, "element": 0.55, "webclients": 0.55}


def s_share(name):
    for k, v in S_SHARE.items():
        if k.lower() in name.lower():
            return v
    return 0.5   # unmeasured (doc: 0.5)


def cmd_of(r):
    if 'command' in r:
        return str(r.get('command') or '')
    return str(parse_query(str(r.get('query') or '')).get('command') or '')


def build_task(rows, cwd, repo):
    ver, res, ridx = {}, [], {}

    def rid(x):
        if x not in ridx:
            ridx[x] = len(res); res.append(x)
        return ridx[x]

    n, e, t, cmds = [], [], [], []
    ctx_first, seen = {}, {}
    census = {c: {'n': 0, 's': 0.0} for c in CELLS[1:]}
    cp = serial = wait_s = 0.0
    nslow = nhid = 0
    t0 = min(float(r['start_s']) for r in rows)
    read_seen, reread = set(), 0
    starts, ends, fns, idents, Ws, vals, durs = [], [], [], [], [], [], []
    share = s_share(repo)
    censusw = {c: 0.0 for c in CELLS[1:]}   # values capped by the LLM window before the call
    prev_end = 0.0
    for i, r in enumerate(rows):
        tool = r.get('tool') or 'bash'
        args = {'command': r.get('command') or ''} if 'command' in r else parse_query(str(r.get('query') or ''))
        dur = float(r['latency_s'])
        st = float(r['start_s']) - t0
        en = float(r.get('end_s') or 0) - t0
        if en < st:
            en = st + dur
        R, W, ident, fn, disp = rw_and_identity(tool, args, cwd)
        serial += dur
        win = max(0.0, st - prev_end)       # decode window a single-turn pre-run could overlap with
        prev_end = max(prev_end, en)
        ins, est = [], 0.0
        for x in sorted(R):
            v, avail, prod = ver.get(x, (0, 0.0, -1))
            ins.append((x, v, prod, avail)); est = max(est, avail)
        ctx = (ident, tuple((x, v) for x, v, _, _ in ins))
        dup = ctx_first.get(ctx, -1) if R else -1
        if R and ctx not in ctx_first:
            ctx_first[ctx] = i
        ef = est + dur
        cp = max(cp, ef)
        slack = max(0.0, st - est)
        cell, val = 0, 0.0
        if fn == 'Wait':
            wait_s += dur
        elif dur >= SLOW:
            nslow += 1
            if slack >= dur:
                nhid += 1
            cmd = str(args.get('command') or '') if tool == 'bash' else ''
            if dup >= 0:
                cell, val = 1, dur
            elif ident in seen:
                if tool == 'bash' and INCR.search(cmd):
                    cell, val = 2, share * dur
                elif tool == 'bash' and TESTP.search(cmd):
                    cell, val = 3, min(slack, dur)
                else:
                    cell, val = 4, min(slack, dur)
            else:
                cell, val = 5, (dur if slack >= dur else slack)
            census[CELLS[cell]]['n'] += 1
            census[CELLS[cell]]['s'] += val
            censusw[CELLS[cell]] += val if cell == 1 else min(val, win)
        first = seen.get(ident, -1)
        seen.setdefault(ident, i)
        for x, v, prod, avail in ins:
            e.append([prod, i, rid(x), v])
            if (x, v) in read_seen:
                reread += 1
            read_seen.add((x, v))
        for x in sorted(W):
            v, _, _ = ver.get(x, (0, 0.0, -1))
            ver[x] = (v + 1, ef, i)
        tl = 'wait' if fn == 'Wait' else (tool if tool in TOOLS else 'other')
        n.append([i, FNS.index(fn), [rid(x) for x in sorted(W)], dup, ctx_first.get(ctx, i)])
        t.append([i, TOOLS.index(tl), round(dur, 3), round(st, 2), round(est, 2), round(slack, 2),
                  cell, round(val, 1), first, round(win, 1), round(val if cell == 1 else min(val, win), 1)])
        cmds.append(disp[:1200])
        starts.append(st); ends.append(en); fns.append(fn); idents.append(ident)
        Ws.append(W); vals.append(val); durs.append(dur)

    N = len(rows)
    # loop-guard: consecutive same-identity, non-mutating (writes only to art/deps allowed)
    lg = []
    i = 0
    while i < N:
        if fns[i] == 'Wait':
            i += 1; continue
        j = i
        while (j + 1 < N and fns[j + 1] != 'Wait' and idents[j + 1] == idents[i]
               and not (Ws[j] - {'art', 'deps'}) and not (Ws[j + 1] - {'art', 'deps'})):
            j += 1
        if j > i:
            tl_ = [round(max(0.0, durs[k] - vals[k]), 1) for k in range(i + 1, j + 1)]
            gp_ = [round(max(0.0, starts[k] - ends[k - 1]), 1) for k in range(i + 1, j + 1)]
            lg.append([i, j - i + 1, tl_, gp_])
        i = j + 1
    # read-parallelism: adjacent read-class runs, no reaching edge between neighbours,
    # minus the overlap already realised (langgraph parallel tool calls)
    cons = {(p, c) for p, c, _, _ in e if p >= 0}
    par = []
    i = 0
    while i < N:
        j = i
        while j + 1 < N and fns[j] == 'Read' and fns[j + 1] == 'Read' and (j, j + 1) not in cons:
            j += 1
        if j > i:
            lats = durs[i:j + 1]
            save = sum(lats) - max(lats)
            overlap = sum(max(0.0, ends[k] - starts[k + 1]) for k in range(i, j))
            save = max(0.0, save - overlap)
            if save > 0.05:
                par.append([i, j, round(save, 2)])
        i = j + 1
    span = max(ends) if ends else 0.0
    for c in census.values():
        c['s'] = round(c['s'], 1)
    m = {'serial': round(serial, 1), 'span': round(span, 1), 'gap': round(max(0.0, span - serial), 1),
         'wait': round(wait_s, 1), 'cp': round(cp, 1), 'ratio': round(cp / serial, 3) if serial else None,
         'nslow': nslow, 'nhid': nhid, 'census': census, 'reread': reread}
    rc = {c: census[c]['s'] for c in CELLS[1:]}
    for c in CELLS[1:]:
        rc[c + 'w'] = round(censusw[c], 1)
    rc['par'] = round(sum(p[2] for p in par), 1)
    rc['lg'] = [[L, a, b] for _, L, a, b in lg]
    return {'res': res, 'n': n, 'e': e, 't': t, 'cmd': cmds, 'lg': lg, 'par': par,
            'pf': {'reread': reread}, 'm': m, 'rc': rc}


# ---- predicted e2e ceiling (TOOL_ANALYSIS_GRAPH.md s9.2), composable by source
SPINE = ('L1', 'L2', 'L3', 'PASTE', 'T1')


def reclaim(rc, srcs, K):
    w = 'w' if 'win' in srcs else ''
    s = sum(rc[c + w] for c in SPINE if c in srcs)
    if 'par' in srcs:
        s += rc['par']
    if 'loop' in srcs:
        for L, tl_, gp_ in rc['lg']:
            for pos in range(1, L):          # surplus turns; charge those beyond the K-th identical
                if pos >= K:
                    s += tl_[pos - 1] + gp_[pos - 1]
    return s


def ceiling(span, r):
    if span <= 0:
        return 1.0
    r = min(r, 0.95 * span)
    return span / (span - r)


def stats(pairs):
    """pairs = [(span, reclaim)] per task.  agg = the setting's total speedup, time-weighted:
    sum(e2e) / sum(e2e - capped reclaim); mean = arithmetic mean of per-task ceilings; geomean etc."""
    pairs = [(sp, r) for sp, r in pairs if sp > 0]
    if not pairs:
        return {'n': 0}
    xs = sorted(ceiling(sp, r) for sp, r in pairs)
    S = sum(sp for sp, _ in pairs)
    R = sum(min(r, 0.95 * sp) for sp, r in pairs)
    return {'n': len(xs), 'agg': round(S / (S - R), 3), 'mean': round(sum(xs) / len(xs), 3),
            'geomean': round(math.exp(sum(math.log(x) for x in xs) / len(xs)), 3),
            'median': round(statistics.median(xs), 3), 'p90': round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 3),
            'max': round(xs[-1], 3), 'ge102': round(sum(x >= 1.02 for x in xs) / len(xs), 3),
            'ge110': round(sum(x >= 1.10 for x in xs) / len(xs), 3)}


PRESETS = {'spine': (set(SPINE), 3), 'spine+tc3': (set(SPINE) | {'loop', 'par'}, 3),
           'spine+tc8': (set(SPINE) | {'loop', 'par'}, 8),
           'spine_win': (set(SPINE) | {'win'}, 3), 'spine_win+tc3': (set(SPINE) | {'win', 'loop', 'par'}, 3)}
for _c in SPINE:
    PRESETS['only_' + _c] = ({_c}, 3)
PRESETS['only_loop3'] = ({'loop'}, 3)
PRESETS['only_loop8'] = ({'loop'}, 8)
PRESETS['only_par'] = ({'par'}, 3)


def tb2_category(task_dir):
    """Terminal-Bench-2 has no repository; group by the official task.toml [metadata] category."""
    try:
        txt = open(os.path.join(task_dir, 'task.toml')).read()
        m = re.search(r'^category\s*=\s*"([^"]+)"', txt, re.M)
        return m.group(1) if m else 'uncategorized'
    except OSError:
        return 'uncategorized'


def manifest_path(ds):
    v = DATASETS[ds]
    if os.path.isabs(v):
        return v
    return os.path.join(BENCH if '/' in v else SUBSETS, v)


_MAN = {}


def load_manifest(ds):
    """cached; returns None (instead of raising) when the manifest is missing, so that adding a
    dataset key whose files are not on this machine cannot break a build of the other datasets."""
    if ds in _MAN:
        return _MAN[ds]
    out = {}
    try:
        fh = open(manifest_path(ds))
    except OSError as ex:
        print('  manifest missing for %s: %s' % (ds, ex))
        _MAN[ds] = None
        return None
    with fh:
        for l in fh:
            if l.strip():
                r = json.loads(l)
                if r.get('repo'):
                    repo = r['repo'].split('/')[-1]
                elif r.get('task_dir'):
                    repo = tb2_category(r['task_dir'])
                else:
                    repo = r['instance_id']
                out[r['instance_id']] = (r.get('cwd') or '/app', repo)
    _MAN[ds] = out
    return out


def guard_out(path):
    """refuse to write through a symlink into the shared tool_speculation tree (the explorer's
    data/tag/<ds> entries are symlinks back to the read-only CLEAN_0905 output)."""
    real = os.path.realpath(path)
    if real == FORBID or real.startswith(FORBID + os.sep):
        raise SystemExit('refusing to write into the shared tree: %s -> %s' % (path, real))
    return path


def cellparts(cell):
    """'<mode>__<fw>[__<arm>]' -> (mode, fw, arm-or-'-')"""
    p = (cell.split('__') + ['-', '-', '-'])[:3]
    return p[0], p[1], p[2]


def add_common_args(ap):
    """the arguments every builder in this directory shares (tag / tl / spec)."""
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--arms', default=None,
                    help='arms for datasets whose layout has {arm}; default per dataset (DSARMS: ops/arms.tsv '
                         'for the six-arm campaign, else %s); "" disables the dimension' % ','.join(ARMS))
    ap.add_argument('--camp-root', default=CAMP,
                    help='root of the A/B campaign traces (default the NFS archive %s; '
                         'use %s to build off a run in progress)' % (CAMP, CAMP_LIVE))
    ap.add_argument('--manifest', default='', metavar='DS=PATH[,DS=PATH]',
                    help='override the manifest of a dataset for this build only, e.g. '
                         'grid12_pro=../../final_test/final_pro.jsonl to smoke-test the grid12_pro '
                         'wiring against a root that holds final_pro tasks')
    return ap


def apply_manifest_overrides(a):
    """--manifest DS=PATH: rebind DATASETS[DS] (and drop its cache) before any cell is read.
    Called by iter_cells, so every builder that shares the arguments honours it."""
    for kv in (getattr(a, 'manifest', '') or '').split(','):
        if not kv.strip():
            continue
        if '=' not in kv:
            raise SystemExit('--manifest expects DS=PATH, got %r' % kv)
        ds, path = (x.strip() for x in kv.split('=', 1))
        if ds not in DATASETS:
            raise SystemExit('--manifest: unknown dataset %r (known: %s)' % (ds, ', '.join(DATASETS)))
        DATASETS[ds] = os.path.abspath(path)
        _MAN.pop(ds, None)
        print('  manifest override: %s -> %s' % (ds, DATASETS[ds]))


def iter_cells(a):
    """yield (ds, mode, fw, arm, cell, trajdir) for every cell that exists on disk.

    cell = '<mode>__<fw>' without an arm, '<mode>__<fw>__<arm>' with one -- the id used for the
    output directory and for INDEX[ds][cell] in the page."""
    apply_manifest_overrides(a)
    cli = getattr(a, 'arms', '')      # a Namespace without the attribute means no arm dimension, as before
    for ds in a.datasets.split(','):
        ds = ds.strip()
        if not ds or ds not in DATASETS:
            continue
        tpl = DSTPL.get(ds, TPL)
        arms = arms_of(ds, cli)
        root = getattr(a, 'camp_root', CAMP) if ds in DSROOT else a.root
        for arm in (arms if '{arm}' in tpl else [None]):
            for mode in MODES:
                for fw in FWS:
                    # '' for {arm} leaves a leading '/', which os.path.join would read as an
                    # absolute path and silently scan the filesystem root: drop empty components.
                    rel = tpl.format(arm=arm or '', dsdir=DSDIR.get(ds, ds), mode=mode, fw=fw)
                    traj = os.path.join(root, *[c for c in rel.split('/') if c])
                    if not os.path.isdir(traj):
                        continue
                    cell = '%s__%s' % (mode, fw) + ('__%s' % arm if arm else '')
                    yield ds, mode, fw, arm, cell, traj


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument('--out', default=os.path.join(EXPL, 'data', 'tag'))
    ap.add_argument('--merge', action='store_true',
                    help='merge the datasets built now into an existing index.json instead of '
                         'replacing it (and skip the census, which would then be partial)')
    a = ap.parse_args()
    index, groups = {}, {}
    if a.merge:
        try:
            index = json.load(open(os.path.join(a.out, 'index.json')))
        except (OSError, ValueError):
            index = {}
    for ds, mode, fw, arm, cell, traj in iter_cells(a):
        man = load_manifest(ds)
        if man is None:
            continue
        outdir = guard_out(os.path.join(a.out, ds, cell))
        os.makedirs(outdir, exist_ok=True)
        st = Counter(); bookdist = Counter(); tasks = []
        for d in sorted(os.listdir(traj)):
            if d not in man:
                if os.path.isdir(os.path.join(traj, d)):
                    st['skipped_noncanonical'] += 1   # archived rerun (__suffix) or unknown
                continue
            f = os.path.join(traj, d, TC)
            if not os.path.exists(f):
                st['no_toolcalls'] += 1
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            nraw = len(rows)
            rows = [r for r in rows if not BOOK.search(cmd_of(r))]
            rows.sort(key=lambda r: float(r['start_s']))
            st['rows'] += nraw; st['book'] += nraw - len(rows); bookdist[nraw - len(rows)] += 1
            if not rows:
                st['empty'] += 1
                continue
            cwd, repo = man[d]
            g = build_task(rows, cwd, repo)
            g.update({'iid': d, 'repo': repo, 'cwd': cwd, 'ds': ds, 'mode': mode, 'fw': fw,
                      'arm': arm, 'cell': cell})
            json.dump(g, open(os.path.join(outdir, d + '.json'), 'w'), separators=(',', ':'))
            st['tasks'] += 1; st['nodes'] += len(g['n']); st['edges'] += len(g['e'])
            tasks.append({'iid': d, 'repo': repo, 'ncalls': len(g['n']), 'm': g['m'], 'rc': g['rc']})
        tasks.sort(key=lambda x: -sum(x['rc'][c] for c in SPINE))
        index.setdefault(ds, {})[cell] = tasks
        groups[(ds, cell)] = tasks
        bd = ' '.join('%d:%d' % kv for kv in sorted(bookdist.items())[:12])
        print('%-24s %-28s tasks=%3d/%d rows=%6d book=%5d nodes=%6d edges=%6d skipped=%d no_tc=%d empty=%d  bookdist[%s]'
              % (ds, cell, st['tasks'], len(man), st['rows'], st['book'], st['nodes'], st['edges'],
                 st['skipped_noncanonical'], st['no_toolcalls'], st['empty'], bd))
    json.dump(index, open(guard_out(os.path.join(a.out, 'index.json')), 'w'), separators=(',', ':'))
    if a.merge:   # a partial census would overwrite the full one with a subset: don't write it
        print('wrote', a.out, ': index.json (merged),',
              sum(len(v) for g in index.values() for v in g.values()), 'tasks indexed')
        return

    # ---- census + predicted speedup, per dataset x mode x fw
    cen, md = {}, []
    md.append('# TAG census on H200_profiling_portal/clean_data (tool_calls.jsonl only)\n')
    md.append('Slow = tool call >= 5 s (tmux WAIT/READ/KEYS keystrokes and sleep-polling excluded, summed as wait). Cells priced as '
              'build_bipartite.py: L1 = dur; L2 = S-share x dur; L3/PASTE = min(slack, dur); '
              'T1 = dur if slack >= dur else slack. loop-guard K = identical results shown before a '
              'warning (turns beyond the K-th charged: unclaimed tool s + LLM gap s). read-par = '
              'sum-max over adjacent independent reads minus overlap already realised by parallel '
              'calls. prefill = re-reads of an unchanged (resource, version), count only. '
              'e2e = span first tool start -> last tool end (omits the leading/trailing LLM turns). '
              '"window" = each opportunity additionally capped by the LLM gap that precedes the call '
              '(what a single-turn pre-run can overlap with; L1 is served instantly and is not capped) -- '
              'TAG_ARGUMENT s4 splits first-sight time into within-window (today T1) and beyond-window (no mechanism).\n')
    md.append('## Opportunity census\n')
    md.append('| dataset | mode | fw | tasks | calls | tool s | span s | tool share | cp ratio | slow | hidable | '
              'L1 n/s | L2 n/s | L3 n/s | PASTE n/s | T1 n/s | loop K3 s | loop K8 s | reread | read-par s | wait s |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (ds, cell), tasks in groups.items():
        mode, fw, arm = cellparts(cell)
        fw = fw if arm == '-' else fw + '/' + arm
        if not tasks:
            continue
        agg = Counter(); cells = {c: [0, 0.0] for c in SPINE}
        for x in tasks:
            m, rc = x['m'], x['rc']
            agg['calls'] += x['ncalls']; agg['tool'] += m['serial']; agg['span'] += m['span']
            agg['cp'] += m['cp']; agg['slow'] += m['nslow']; agg['hid'] += m['nhid']
            agg['reread'] += m['reread']; agg['par'] += rc['par']; agg['wait'] += m['wait']
            for c in SPINE:
                cells[c][0] += m['census'][c]['n']; cells[c][1] += m['census'][c]['s']
                agg[c + 'w'] += rc[c + 'w']
            for K in K_TABLE:
                agg['loop%d' % K] += reclaim(rc, {'loop'}, K)
        sp = {name: stats([(x['m']['span'], reclaim(x['rc'], srcs, K)) for x in tasks])
              for name, (srcs, K) in PRESETS.items()}
        row = {'tasks': len(tasks), 'calls': agg['calls'], 'tool_s': round(agg['tool']), 'span_s': round(agg['span']),
               'tool_share': round(agg['tool'] / agg['span'], 3) if agg['span'] else None,
               'cp_ratio': round(agg['cp'] / agg['tool'], 3) if agg['tool'] else None,
               'slow': agg['slow'], 'hidable': agg['hid'],
               'cells': {c: {'n': v[0], 's': round(v[1])} for c, v in cells.items()},
               'loop_K3_s': round(agg['loop3']), 'loop_K8_s': round(agg['loop8']), 'reread': agg['reread'],
               'par_s': round(agg['par']), 'wait_s': round(agg['wait']), 'speedup': sp,
               'window': {c: round(agg[c + 'w']) for c in SPINE},
               'decomp': {'repeated_s': round(sum(cells[c][1] for c in SPINE[:4])),
                          'firstsight_in_window_s': round(agg['T1w']),
                          'firstsight_beyond_window_s': round(cells['T1'][1] - agg['T1w'])}}
        cen['%s/%s' % (ds, cell)] = row
        md.append('| %s | %s | %s | %d | %d | %d | %d | %.3f | %.3f | %d | %d | %s | %d | %d | %d | %d | %d |' % (
            ds, mode, fw, row['tasks'], row['calls'], row['tool_s'], row['span_s'], row['tool_share'] or 0,
            row['cp_ratio'] or 0, row['slow'], row['hidable'],
            ' | '.join('%d/%d' % (row['cells'][c]['n'], row['cells'][c]['s']) for c in SPINE),
            row['loop_K3_s'], row['loop_K8_s'], row['reread'], row['par_s'], row['wait_s']))
    md.append('\n## Where the priced seconds sit (TAG_ARGUMENT s4 decomposition of the spine total)\n')
    md.append('| dataset | mode | fw | spine total s | repeated (L1+L2+L3+PASTE) | first-sight within one LLM window (T1w) | first-sight beyond the window (no mechanism) | window-capped spine s |')
    md.append('|---|---|---|---|---|---|---|---|')
    for key, row in cen.items():
        ds, cell = key.split('/', 1); mode, fw, arm = cellparts(cell)
        fw = fw if arm == '-' else fw + '/' + arm
        tot = sum(row['cells'][c]['s'] for c in SPINE); d = row['decomp']
        pct = lambda v: '%d (%d%%)' % (v, round(100 * v / tot)) if tot else '0'
        md.append('| %s | %s | %s | %d | %s | %s | %s | %d |' % (ds, mode, fw, tot, pct(d['repeated_s']), pct(d['firstsight_in_window_s']),
                  pct(d['firstsight_beyond_window_s']), sum(row['window'][c] for c in SPINE)))
    md.append('\n## Predicted e2e speedup per setting: aggregate (time-weighted, sum e2e / sum(e2e - reclaim)) / mean of per-task ceilings / geomean / median / p90 / max / share of tasks >= 1.10x; single-cell columns = aggregate\n')
    md.append('| dataset | mode | fw | spine | spine + Track-C K=3 | spine + Track-C K=8 | spine, window-capped | window-capped + Track-C K=3 | L1 only | L2 only | L3 only | PASTE only | T1 only | loop K3 only | loop K8 only | read-par only |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')

    def fmt(s):
        return '**%.3f** / %.3f / %.3f / %.3f / %.3f / %.2f / %d%%' % (s['agg'], s['mean'], s['geomean'], s['median'], s['p90'], s['max'], round(100 * s['ge110'])) if s.get('n') else '-'

    def fmt1(s):
        return '%.3f' % s['agg'] if s.get('n') else '-'
    for key, row in cen.items():
        ds, cell = key.split('/', 1); mode, fw, arm = cellparts(cell)
        fw = fw if arm == '-' else fw + '/' + arm
        sp = row['speedup']
        md.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ds, mode, fw, fmt(sp['spine']), fmt(sp['spine+tc3']), fmt(sp['spine+tc8']), fmt(sp['spine_win']), fmt(sp['spine_win+tc3']),
            ' | '.join(fmt1(sp[k]) for k in ('only_L1', 'only_L2', 'only_L3', 'only_PASTE', 'only_T1', 'only_loop3', 'only_loop8', 'only_par'))))
    # top tasks
    allt = [(ds, cellparts(cell)[0], cellparts(cell)[1] + ('' if cellparts(cell)[2] == '-' else '/' + cellparts(cell)[2]), x)
            for (ds, cell), ts in groups.items() for x in ts]
    md.append('\n## Top 20 tasks by priced spine opportunity\n')
    md.append('| dataset | mode | fw | task | span s | tool s | priced s | L1 | L2 | L3 | PASTE | T1 | ceiling spine | ceiling +TC K3 |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    top = {'by_priced': [], 'by_ceiling': []}
    for ds, mode, fw, x in sorted(allt, key=lambda z: -reclaim(z[3]['rc'], set(SPINE), 3))[:20]:
        rc, m = x['rc'], x['m']; pr = reclaim(rc, set(SPINE), 3)
        rec = {'ds': ds, 'mode': mode, 'fw': fw, 'iid': x['iid'], 'span': m['span'], 'tool': m['serial'], 'priced': round(pr, 1),
               'cells': {c: rc[c] for c in SPINE}, 'ceil_spine': round(ceiling(m['span'], pr), 3),
               'ceil_tc3': round(ceiling(m['span'], reclaim(rc, set(SPINE) | {'loop', 'par'}, 3)), 3)}
        top['by_priced'].append(rec)
        md.append('| %s | %s | %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %.2f | %.2f |' % (
            ds, mode, fw, x['iid'][:60], m['span'], m['serial'], pr, rc['L1'], rc['L2'], rc['L3'], rc['PASTE'], rc['T1'],
            rec['ceil_spine'], rec['ceil_tc3']))
    md.append('\n## Top 20 tasks by predicted ceiling (window-capped spine + Track-C, K=3)\n')
    md.append('| dataset | mode | fw | task | span s | tool s | ceiling | window-capped spine s | uncapped spine s | loop K3 s | read-par s |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|')
    WTC = set(SPINE) | {'win', 'loop', 'par'}
    for ds, mode, fw, x in sorted(allt, key=lambda z: -ceiling(z[3]['m']['span'], reclaim(z[3]['rc'], WTC, 3)))[:20]:
        rc, m = x['rc'], x['m']
        c = ceiling(m['span'], reclaim(rc, WTC, 3))
        rec = {'ds': ds, 'mode': mode, 'fw': fw, 'iid': x['iid'], 'span': m['span'], 'tool': m['serial'], 'ceiling': round(c, 3),
               'spine_win': round(reclaim(rc, set(SPINE) | {'win'}, 3), 1), 'spine': round(reclaim(rc, set(SPINE), 3), 1),
               'loop3': round(reclaim(rc, {'loop'}, 3), 1), 'par': rc['par']}
        top['by_ceiling'].append(rec)
        md.append('| %s | %s | %s | %s | %d | %d | %.2f | %d | %d | %d | %d |' % (
            ds, mode, fw, x['iid'][:60], m['span'], m['serial'], c, rec['spine_win'], rec['spine'], rec['loop3'], rec['par']))
    json.dump({'groups': cen, 'top': top}, open(guard_out(os.path.join(a.out, 'census.json')), 'w'), indent=1)
    open(guard_out(os.path.join(a.out, 'TAG_CENSUS.md')), 'w').write('\n'.join(md) + '\n')
    print('wrote', a.out, ': index.json, census.json, TAG_CENSUS.md,', sum(len(v) for g in index.values() for v in g.values()), 'task files')


if __name__ == '__main__':
    main()
