/* run_agg.js -- the pure aggregation behind the Grid-12 run card (#app-run in index.html).
   No DOM, no fetch: index.html loads it with <script src="run_agg.js"> (window.RunAgg) and
   tests/test_pairs.js loads it with require() and checks it against tests/synth_pairs.py's Python
   reference.  Input is data/run/grid12/pairs.json as technique/ops/grid12_compare.py writes it:
     pair = {iid, repo, mode, fw, cell:"<mode>__<fw>", card, order, base:RUN, full:RUN, speedup_e2e,
             tool_reduction, slow_tool_reduction, flip, hidden:{l1,l2,l3,paste,t1}, cpu_burned_s,
             cpu_wasted_s, cpu_unmeasured_jobs, jobs, claimed, probe_ms_mean, probe_n,
             t1_depth:{"<K>":{jobs,claimed,discarded,expired,denied,finished,queued,hidden_s}} | {} | null,
             clean_wall_s:{base_ref}, gate, notes,
             speedup_e2e_norm}                                   (2026-09-09, trace-replay runs only; see below)
   t1_depth (09-08, schema-additive) is the T1 K-step probe's per-depth breakdown (jobs whose layer is
   spork_d<K>); those jobs are ALREADY inside jobs / claimed / hidden.t1, so nothing here sums it -- the
   page's pair table renders it under the T1 cell (index.html t1Cell) and this module ignores it.
     RUN  = {wall_s, llm_s, tool_s, slow_tool_s, n_tool, n_req, in_tok, out_tok, resolved, rc, dur_s,
             patch_bytes, ts, layers_line, hw|null, replay|absent}
   replay (2026-09-09, trace-replay runs only) = {mode:"live"|"record"|"replay-fwd", n_fwd, divergent, divergent_fresh,
   live_llm_s, llm_s_norm, norm_coverage, live_completion_tokens, ...}; the comparator's pair.speedup_e2e_norm is
   base.wall_norm / full.wall_norm with wall_norm = wall_s - replay.live_llm_s + replay.llm_s_norm (the wall after
   scaling each replayed request's live LLM latency to the recorded response length), null when either arm lacks a
   number and ABSENT on an ordinary run -- a run without the field aggregates exactly as before (speedup_norm null).
   Rules that every function here follows (the project's standing failure mode is a missing number
   rendered as 0): a null stays null and is COUNTED, never summed as 0; a time-weighted ratio uses
   only the pairs where both arms carry the number; "unmeasured" CPU is carried as a job count. */
(function (root) {
  'use strict';
  const LAYERS = ['l1', 'l2', 'l3', 'paste', 't1'];
  const LAYER_LABEL = { l1: 'L1', l2: 'L2', l3: 'L3', paste: 'PASTE', t1: 'T1' };
  const SETTINGS = ['thinking__hermes', 'thinking__langgraph', 'instruct__hermes', 'instruct__langgraph', 'coder__hermes', 'coder__langgraph'];
  // final_test/grid12_pro.jsonl: 11 repos x 2 tasks.  Kept as the row order so the matrix has the
  // same shape from the first pair to the last; a repo the file names that is not here is appended.
  const REPOS = ['ansible', 'element-web', 'flipt', 'navidrome', 'NodeBB', 'openlibrary', 'qutebrowser', 'teleport', 'tutanota', 'vuls', 'webclients'];
  const FLIPS = ['same_resolved', 'same_unresolved', 'base_only', 'full_only', 'unknown'];
  const GATES = ['PASS', 'WARN', 'FAIL'];
  const isNum = v => typeof v === 'number' && isFinite(v);
  const repoShort = r => String(r == null ? '' : r).split('/').pop();
  const settingOf = p => p.cell || ((p.mode || '') + '__' + (p.fw || ''));
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  /* the normalised wall of one arm, re-derived from its replay block the way ops/grid12_compare.py does it; null unless
     wall_s, replay.live_llm_s and replay.llm_s_norm are all numbers.  The comparator's own ratio (speedup_e2e_norm) wins
     when it is there; the walls only re-derive it for a file that carries the replay blocks without the field. */
  const replayOf = r => (r && r.replay && typeof r.replay === 'object') ? r.replay : null;
  const wallNorm = r => { const rp = replayOf(r); return (rp && isNum(r.wall_s) && isNum(rp.live_llm_s) && isNum(rp.llm_s_norm)) ? r.wall_s - rp.live_llm_s + rp.llm_s_norm : null; };
  const normOf = p => { if (!p) return null; if (isNum(p.speedup_e2e_norm)) return p.speedup_e2e_norm; const bn = wallNorm(p.base), fn = wallNorm(p.full); return (bn !== null && fn !== null && fn > 0) ? bn / fn : null; };

  /* one aggregate over a list of pairs (a matrix cell, a row total, the grand total) */
  function aggCell(pairs) {
    const A = { n: pairs.length,
      n_wall: 0, base_wall: 0, full_wall: 0, speedup: null,
      n_norm: 0, base_wall_norm: 0, full_wall_norm: 0, speedup_norm: null, n_replay: 0,
      n_tool: 0, base_tool: 0, full_tool: 0, tool_red: null,
      n_slow: 0, base_slow: 0, full_slow: 0, slow_red: null,
      flips: {}, flip_n: 0, n_resolved_base: 0, n_resolved_full: 0, n_graded: 0,
      hidden: {}, hidden_n: {}, hidden_total: null, hidden_n_any: 0,
      cpu_burned: null, cpu_burned_n: 0, cpu_wasted: null, cpu_wasted_n: 0, cpu_unmeasured_jobs: 0, cpu_unmeasured_pairs: 0,
      jobs: 0, claimed: 0, gate: {}, gate_other: 0, iids: [] };
    for (const f of FLIPS) A.flips[f] = 0;
    for (const g of GATES) A.gate[g] = 0;
    for (const l of LAYERS) { A.hidden[l] = null; A.hidden_n[l] = 0; }
    for (const p of pairs) {
      const b = p.base || {}, f = p.full || {};
      A.iids.push(p.iid);
      if (isNum(b.wall_s) && isNum(f.wall_s) && f.wall_s > 0) { A.n_wall++; A.base_wall += b.wall_s; A.full_wall += f.wall_s; }
      // the normalised speedup, over the pairs where it is not null: weighted by the normalised walls when both replay
      // blocks re-derive them, else (a numeric speedup_e2e_norm whose replay block is incomplete) by the base wall so the
      // pair keeps its own ratio; a null is skipped, never summed as 1.00x
      const nv = normOf(p);
      if (nv !== null) { let bn = wallNorm(b), fn = wallNorm(f);
        if (!(bn !== null && fn !== null && fn > 0) && isNum(b.wall_s) && b.wall_s > 0) { bn = b.wall_s; fn = b.wall_s / nv; }
        if (bn !== null && fn !== null && fn > 0) { A.n_norm++; A.base_wall_norm += bn; A.full_wall_norm += fn; } }
      if (replayOf(f)) A.n_replay++;
      if (isNum(b.tool_s) && isNum(f.tool_s) && b.tool_s > 0) { A.n_tool++; A.base_tool += b.tool_s; A.full_tool += f.tool_s; }
      if (isNum(b.slow_tool_s) && isNum(f.slow_tool_s) && b.slow_tool_s > 0) { A.n_slow++; A.base_slow += b.slow_tool_s; A.full_slow += f.slow_tool_s; }
      A.flips[FLIPS.indexOf(p.flip) >= 0 ? p.flip : 'unknown']++;
      if (b.resolved === true) A.n_resolved_base++;
      if (f.resolved === true) A.n_resolved_full++;
      if (typeof b.resolved === 'boolean' && typeof f.resolved === 'boolean') A.n_graded++;
      const h = p.hidden || {};
      let any = false;
      for (const l of LAYERS) if (isNum(h[l])) { A.hidden[l] = (A.hidden[l] || 0) + h[l]; A.hidden_n[l]++; A.hidden_total = (A.hidden_total || 0) + h[l]; any = true; }
      if (any) A.hidden_n_any++;
      if (isNum(p.cpu_burned_s)) { A.cpu_burned = (A.cpu_burned || 0) + p.cpu_burned_s; A.cpu_burned_n++; }
      if (isNum(p.cpu_wasted_s)) { A.cpu_wasted = (A.cpu_wasted || 0) + p.cpu_wasted_s; A.cpu_wasted_n++; }
      if (isNum(p.cpu_unmeasured_jobs) && p.cpu_unmeasured_jobs > 0) { A.cpu_unmeasured_jobs += p.cpu_unmeasured_jobs; A.cpu_unmeasured_pairs++; }
      if (isNum(p.jobs)) A.jobs += p.jobs;
      if (isNum(p.claimed)) A.claimed += p.claimed;
      if (GATES.indexOf(p.gate) >= 0) A.gate[p.gate]++; else A.gate_other++;
    }
    A.speedup = (A.n_wall && A.full_wall > 0) ? A.base_wall / A.full_wall : null;     // time-weighted: sum(base.wall) / sum(full.wall)
    A.speedup_norm = (A.n_norm && A.full_wall_norm > 0) ? A.base_wall_norm / A.full_wall_norm : null;   // the same, over the normalised walls
    A.tool_red = (A.n_tool && A.base_tool > 0) ? 1 - A.full_tool / A.base_tool : null;  // 1 - sum(full.tool) / sum(base.tool)
    A.slow_red = (A.n_slow && A.base_slow > 0) ? 1 - A.full_slow / A.base_slow : null;
    A.flip_n = A.flips.base_only + A.flips.full_only;                                  // "resolved flips": the outcome changed
    return A;
  }

  /* repos x settings; rows = REPOS (+ any repo the pairs add), columns = SETTINGS (+ any cell the pairs add) */
  function matrix(pairs, repos, settings) {
    const R = (repos || REPOS).slice(), S = (settings || SETTINGS).slice();
    for (const p of pairs) { const r = repoShort(p.repo), s = settingOf(p); if (R.indexOf(r) < 0) R.push(r); if (S.indexOf(s) < 0) S.push(s); }
    const cells = {}, rowAll = {}, colAll = {};
    for (const r of R) { cells[r] = {}; const pr = pairs.filter(p => repoShort(p.repo) === r); rowAll[r] = aggCell(pr);
      for (const s of S) cells[r][s] = aggCell(pr.filter(p => settingOf(p) === s)); }
    for (const s of S) colAll[s] = aggCell(pairs.filter(p => settingOf(p) === s));
    return { repos: R, settings: S, cells, rowAll, colAll, all: aggCell(pairs) };
  }

  /* what a pair's delta-wall is made of: the hidden seconds by layer (stacked) + the unexplained rest
     (negative when full took LONGER than the hidden seconds account for), against the CLEAN_0905
     ceiling for the same task (window-capped L1w+L2w+L3w+PASTEw+T1w, from data/tag/index.json). */
  function contribution(pair, ceiling) {
    const b = pair.base || {}, f = pair.full || {}, h = pair.hidden || {};
    const dwall = (isNum(b.wall_s) && isNum(f.wall_s)) ? b.wall_s - f.wall_s : null;
    const segs = LAYERS.map(l => ({ l, label: LAYER_LABEL[l], s: isNum(h[l]) ? h[l] : null }));
    const known = segs.filter(x => x.s !== null);
    const hidden_sum = known.length ? sum(known.map(x => x.s)) : null;
    return { dwall, segs, hidden_sum, n_hidden_null: LAYERS.length - known.length,
      unexplained: (dwall !== null) ? dwall - (hidden_sum || 0) : null,
      ceiling: isNum(ceiling) ? ceiling : null };
  }

  /* the CLEAN_0905 ceiling of the same task in the same setting: index = data/tag/index.json */
  function ceilingOf(index, cell, iid) {
    const rows = index && index.pro_250_nonet && index.pro_250_nonet[cell];
    if (!rows) return null;
    const r = rows.find(x => x.iid === iid);
    if (!r || !r.rc) return null;
    let s = 0;
    for (const k of ['L1w', 'L2w', 'L3w', 'PASTEw', 'T1w']) { if (!isNum(r.rc[k])) return null; s += r.rc[k]; }
    return s;
  }

  /* sortable pair table: the value of one column for one pair (null = not available) */
  function pairValue(p, key) {
    const b = p.base || {}, f = p.full || {}, h = p.hidden || {};
    switch (key) {
      case 'iid': return p.iid || '';
      case 'repo': return repoShort(p.repo);
      case 'setting': return settingOf(p);
      case 'arm': return p.arm_label || p.arm || 'full';   // 2026-09-09: six-arm pairs (schema 2); a legacy pair is the full arm
      case 'card': return isNum(p.card) ? p.card : (p.card == null ? null : String(p.card));
      case 'base_wall': return isNum(b.wall_s) ? b.wall_s : null;
      case 'full_wall': return isNum(f.wall_s) ? f.wall_s : null;
      case 'speedup': return isNum(p.speedup_e2e) ? p.speedup_e2e : ((isNum(b.wall_s) && isNum(f.wall_s) && f.wall_s > 0) ? b.wall_s / f.wall_s : null);
      case 'speedup_norm': return normOf(p);   // trace-replay runs only: null (not 1.00x) everywhere else
      case 'tool_red': return isNum(p.tool_reduction) ? p.tool_reduction : ((isNum(b.tool_s) && isNum(f.tool_s) && b.tool_s > 0) ? 1 - f.tool_s / b.tool_s : null);
      case 'slow_red': return isNum(p.slow_tool_reduction) ? p.slow_tool_reduction : null;
      case 'flip': return FLIPS.indexOf(p.flip) >= 0 ? FLIPS.indexOf(p.flip) : FLIPS.length;
      case 'hidden': { const v = LAYERS.filter(l => isNum(h[l])).map(l => h[l]); return v.length ? sum(v) : null; }
      case 'l1': case 'l2': case 'l3': case 'paste': case 't1': return isNum(h[key]) ? h[key] : null;
      case 'cpu': return isNum(p.cpu_burned_s) ? p.cpu_burned_s : null;
      case 'cpu_wasted': return isNum(p.cpu_wasted_s) ? p.cpu_wasted_s : null;
      case 'probe': return isNum(p.probe_ms_mean) ? p.probe_ms_mean : null;
      case 'gate': return GATES.indexOf(p.gate) >= 0 ? GATES.indexOf(p.gate) : GATES.length;
      default: return null;
    }
  }
  /* stable sort; nulls always last whatever the direction */
  function sortPairs(pairs, key, dir) {
    const d = dir < 0 ? -1 : 1;
    return pairs.map((p, i) => [p, i]).sort((x, y) => {
      const a = pairValue(x[0], key), b = pairValue(y[0], key);
      if (a === null && b === null) return x[1] - y[1];
      if (a === null) return 1;
      if (b === null) return -1;
      const c = (a < b) ? -1 : (a > b) ? 1 : 0;
      return c ? c * d : x[1] - y[1];
    }).map(x => x[0]);
  }

  const api = { LAYERS, LAYER_LABEL, SETTINGS, REPOS, FLIPS, GATES, isNum, repoShort, settingOf, replayOf, wallNorm, normOf, aggCell, matrix, contribution, ceilingOf, pairValue, sortPairs };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.RunAgg = api;
})(typeof window !== 'undefined' ? window : globalThis);
