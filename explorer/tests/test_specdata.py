#!/usr/bin/env python3
"""Unit test for build_specdata.py -- no GPU, no campaign, no network.

Synthesises a spec_events.jsonl covering every state in the machine (QUEUED / RUNNING /
PREEMPTED+resume / FINISHED / CLAIMED / DISCARDED / EXPIRED / DENIED), plus the three cases
the live pool will actually produce and a naive reader would get wrong:
  * a job truncated mid-run (the log stops while it is on the CPU),
  * a pair of jobs that overlap (so the row packer has to stack them),
  * a claim issued while the job was still running (the foreground then waits).
Then runs the fold and asserts the JSON the page consumes.
Usage: python tests/test_specdata.py
"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'build'))
import build_specdata as B   # noqa: E402

T0 = 1000.0


def E(t, ev, jid, **kw):
    return json.dumps(dict({'t': T0 + t, 'ev': ev, 'jid': jid}, **kw))


EVENTS = [
    # the per-process header: both clocks + the task id (spec_events.py:395)
    json.dumps({'t': T0 + 0.5, 'ev': 'SESSION', 'wall': 1.785e9, 'pid': 4242, 'iid': 'task-A',
                'slot': 3, 'clock': 'perf_counter/CLOCK_MONOTONIC'}),
    json.dumps({'t': T0 + 0.6, 'ev': 'NOTE', 'why': 'pool started, budget 4'}),
    # j1: the completion-gated happy path -- ran, exited, then claimed
    E(1, 'QUEUED', 'j1', layer='l2', key='go build ./...', epoch=3, call=1, cmd='cd /app && go build ./...'),
    E(2, 'RUNNING', 'j1', layer='l2', key='go build ./...', epoch=3, cpid=101, cpu_ms=0),
    E(10, 'FINISHED', 'j1', layer='l2', key='go build ./...', epoch=3, cpu_ms=7800, rc=0, lat_ms=8010),
    E(15, 'CLAIMED', 'j1', layer='l2', key='go build ./...', epoch=3, cpu_ms=7800,
      claimed_by='go build ./...', hidden_ms=8000, waited_ms=0, why='identity+epoch'),
    # j2: overlaps j1 (row packing), ran to completion, TTL expired unclaimed
    E(2, 'QUEUED', 'j2', layer='L3', key='go test ./internal/...', epoch=3),
    E(3, 'RUNNING', 'j2', layer='L3', key='go test ./internal/...', epoch=3, cpid=102, cpu_ms=0),
    E(9, 'FINISHED', 'j2', layer='L3', key='go test ./internal/...', epoch=3, cpu_ms=5100),
    E(40, 'EXPIRED', 'j2', layer='L3', key='go test ./internal/...', epoch=3, cpu_ms=5100, why='ttl 30s'),
    # j3: SIGSTOP for a foreground command, SIGCONT, then killed by an epoch bump
    E(5, 'QUEUED', 'j3', layer='L2', key='npx tsc -p .', epoch=3),
    E(6, 'RUNNING', 'j3', layer='L2', key='npx tsc -p .', epoch=3, cpid=103, cpu_ms=0),
    E(8, 'PREEMPTED', 'j3', layer='L2', key='npx tsc -p .', epoch=3, cpu_ms=1800, why='foreground call 2'),
    E(12, 'RESUMED', 'j3', why='foreground done'),   # RESUMED, not a second RUNNING
    E(14, 'DISCARDED', 'j3', layer='L2', key='npx tsc -p .', epoch=3, cpu_ms=3300, why='epoch 3 -> 4'),
    # j4: the per-slot budget refused it; it never ran
    E(20, 'QUEUED', 'j4', layer='L1', key='cat go.mod', epoch=4),
    E(20.1, 'DENIED', 'j4', layer='L1', key='cat go.mod', epoch=4, why='budget 4/4 busy'),
    # j5 truncated (no terminal event) and j7 overlapping it, finished unclaimed
    E(30, 'QUEUED', 'j5', layer='L3', key='pytest -q tests/', epoch=4),
    E(31, 'RUNNING', 'j5', layer='L3', key='pytest -q tests/', epoch=4, cpid=105, cpu_ms=0),
    E(33, 'QUEUED', 'j7', layer='L2', key='make -j4', epoch=4),
    E(34, 'RUNNING', 'j7', layer='L2', key='make -j4', epoch=4, cpid=107, cpu_ms=0),
    E(44, 'FINISHED', 'j7', layer='L2', key='make -j4', epoch=4, cpu_ms=9900),
    # j6: claimed while still running -- the foreground waits 5 s for it
    E(50, 'QUEUED', 'j6', layer='L3', key='go test ./internal/...', epoch=4),
    E(50.5, 'RUNNING', 'j6', layer='L3', key='go test ./internal/...', epoch=4, cpid=106, cpu_ms=0),
    E(52, 'CLAIMED', 'j6', layer='l3', key='go test ./internal/...', epoch=4, cpu_ms=1400,
      claimed_by='go test ./internal/...', why='joined a running job'),
    E(57, 'FINISHED', 'j6', layer='L3', key='go test ./internal/...', epoch=4, cpu_ms=6200),
    # an unknown verb, a blank line, and a half-written final line (killed mid-flush)
    E(58, 'HEARTBEAT', 'j5', layer='L3', key='pytest -q tests/', epoch=4, cpu_ms=22000),
    '',
    '{"t": 1059.0, "ev": "FINI',
]

CHECKS = []


def ck(cond, what):
    CHECKS.append((bool(cond), what))


def main():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join(EVENTS) + '\n')
        g = B.build(p, T0, 60.0, 'task-A')   # tool span 0..60 s, same t0
    by = {j['jid']: j for j in g['j']}
    m = g['m']
    ck(len(g['j']) == 7, 'seven jobs folded, got %d' % len(g['j']))
    ck(g['warn'] == '', 'no clock warning on a perf_counter stream')
    ck(by['j1']['o'] == 'claimed' and by['j1']['seg'] == [[2.0, 10.0]], 'j1 claimed after exit')
    ck(by['j1']['call'] == 1 and by['j1']['cby'] == 'go build ./...' and by['j1']['rc'] == 0,
       'QUEUED-time call index, claimed_by command and rc are kept (call is NOT the claimer)')
    ck(m['sessions'] == 1 and g['ses'][0]['iid'] == 'task-A', 'the SESSION header is parsed, not folded into a job')
    ck(len(g['j']) == 7, 'SESSION/NOTE did not become jobs')
    ck(by['j2']['o'] == 'expired' and by['j2']['ct'] == 40.0, 'j2 expired at its TTL')
    ck(by['j3']['o'] == 'discarded' and by['j3']['seg'] == [[6.0, 8.0], [12.0, 14.0]]
       and by['j3']['pre'] == [[8.0, 12.0]], 'j3 PREEMPTED then RESUMED then discarded')
    ck(by['j4']['o'] == 'denied' and by['j4']['seg'] == [], 'j4 denied, never ran')
    ck(by['j5']['o'] == 'running' and by['j5']['e'] == 60.0, 'j5 truncated -> drawn to the end of the run')
    ck(by['j5']['seg'] == [[31.0, 58.0]] and by['j5']['tl'] == [58.0, 60.0],
       'j5: the MEASURED segment stops at its last event; the rest is the assumed tail')
    ck(abs(m['assumed_s'] - 2.0) < 1e-6, 'the assumed tail is summed separately: %s' % m['assumed_s'])
    ck(by['j7']['o'] == 'finished', 'j7 ran to completion, unclaimed')
    ck(by['j6']['o'] == 'claimed' and by['j6']['e'] == 57.0 and by['j6']['ct'] == 52.0,
       'j6 claimed while running, its segment runs on to 57')
    ck(by['j5']['row'] != by['j7']['row'], 'the overlapping pair j5/j7 is packed onto different rows')
    # rows are packed WITHIN a technique and the bands stacked: l1 {j4} = 1 row, l2 {j1,j3,j7} = 2
    # (j1 and j3 overlap, j7 reuses j1's row), l3 {j2,j5,j6} = 2 (j2 and j5 overlap).
    ck(m['peak'] == 3 and m['rows'] == 5, 'peak concurrency 3, five rows in three technique bands: %s' % m['rows'])
    ck([g0[:4] for g0 in m['grp']] == [['l1', 0, 1, 1], ['l2', 1, 2, 3], ['l3', 3, 2, 3]],
       'one band per technique, in roster order, each with its own row block: %s' % m['grp'])
    ck(all(m['grp'][k][1] <= by[j]['row'] < m['grp'][k][1] + m['grp'][k][2]
           for k, j in ((0, 'j4'), (1, 'j1'), (1, 'j3'), (1, 'j7'), (2, 'j2'), (2, 'j5'), (2, 'j6'))),
       'every job sits inside its own technique band')
    ck(m['claimed'] == 2 and m['discarded'] == 1 and m['expired'] == 1 and m['denied'] == 1
       and m['finished'] == 1 and m['running'] == 1, 'outcome census')
    ck(m['started'] == 6 and m['ran'] == 5,
       'started counts the truncated job too (6), ran counts only terminal records (5)')
    ck(abs(m['claim_rate'] - 2 / 6.0) < 1e-4, 'claim rate is over everything that STARTED (2/6)')
    ck(m['claim_rate_terminal'] == 0.4, 'the terminal-record-only rate is kept as the secondary (2/5)')
    ck(abs(m['cpu_s'] - 54.3) < 1e-6, 'cpu high-water marks summed: %s' % m['cpu_s'])
    ck(abs(m['cpu_wasted_s'] - 40.3) < 1e-6, 'CPU on jobs nobody claimed: %s' % m['cpu_wasted_s'])
    ck(abs(m['waited_s'] - 5.0) < 1e-6, 'the 5 s the foreground waited on j6 (derived; j1 logged 0)')
    ck(abs(m['hidden_s'] - (8.0 + 6.5)) < 1e-6, "j1's measured hidden_ms wins, j6's is derived: %s" % m['hidden_s'])
    ck(m['measured_hidden'] == 1, 'one of the two claims carried a measured hidden_ms')
    ck(m['bad'] == 2 and m['trunc'] == 1, 'one truncated line + one unknown verb; one open job')
    # ---- per-technique attribution: the same ledger restricted to one mechanism's jobs --------
    L = {x['id']: x for x in m['lay']}
    ck([x['id'] for x in m['lay']] == ['l1', 'l2', 'l3'],
       'one row per technique, in roster order: %s' % [x['id'] for x in m['lay']])
    ck(L['l2']['n'] == 3 and L['l3']['n'] == 3 and L['l1']['n'] == 1, 'launched per technique')
    ck(L['l2']['claimed'] == 1 and L['l3']['claimed'] == 1 and L['l1']['claimed'] == 0,
       'claims are attributed to the technique that launched the job')
    ck(L['l1']['denied'] == 1 and L['l1']['started'] == 0 and L['l1']['claim_rate'] is None,
       'a technique whose only job was denied has no claim rate at all (not 0%)')
    ck(abs(L['l2']['hidden_s'] - 8.0) < 1e-6 and abs(L['l3']['hidden_s'] - 6.5) < 1e-6,
       'hidden seconds split by technique: %s / %s' % (L['l2']['hidden_s'], L['l3']['hidden_s']))
    ck(abs(L['l2']['cpu_s'] - (7.8 + 3.3 + 9.9)) < 1e-6 and abs(L['l3']['cpu_s'] - (5.1 + 22.0 + 6.2)) < 1e-6,
       'CPU seconds split by technique: %s / %s' % (L['l2']['cpu_s'], L['l3']['cpu_s']))
    ck(abs(sum(x['cpu_s'] for x in m['lay']) - m['cpu_s']) < 1e-6
       and sum(x['n'] for x in m['lay']) == m['n']
       and sum(x['claimed'] for x in m['lay']) == m['claimed'],
       'the technique rows add up to the run total (launched, claimed, CPU)')
    ck(abs(L['l2']['cpu_wasted_s'] - (3.3 + 9.9)) < 1e-6,
       'BOTH halves per technique: the CPU it burned on jobs nobody claimed: %s' % L['l2']['cpu_wasted_s'])
    ck(L['l2']['busy_s'] <= m['busy_s'] and L['l3']['busy_s'] <= m['busy_s'],
       "a technique's own busy wall time cannot exceed the run's")
    # ---- the claim ACTIONS, as events -------------------------------------------------------
    ck([c['i'] for c in g['cl']] == [by['j1']['i'], by['j6']['i']] and [c['t'] for c in g['cl']] == [15.0, 52.0],
       'one record per claim, in claim order, at the instant of the claim: %s' % g['cl'])
    ck(g['cl'][0]['l'] == 'l2' and g['cl'][0]['hid'] == 8.0 and g['cl'][0]['wait'] == 0.0
       and g['cl'][1]['l'] == 'l3' and g['cl'][1]['hid'] is None,
       "a claim carries its technique and the pool's own hidden/waited when it logged them, None when not")
    ck(g['cl'][0]['by'] == 'go build ./...' and g['cl'][0]['row'] == by['j1']['row'],
       'a claim record carries the command that claimed it and the row its bar is on')
    ck(g['conc'][0] == [2.0, 1] and max(c[1] for c in g['conc']) == 3, 'concurrency series starts at j1 and peaks at 3')
    ck(g['conc'][-1][0] == 58.0 and m['peak'] == m['peak_unknown'],
       'the measured series ends at the last real event, not at the end of the task')
    ck(all(j['cm'] == 1 for j in g['j'] if j['o'] != 'denied') and m['cpu_unmeasured'] == 0,
       'every job that ran here carried a cpu_ms')
    ck(abs(m['busy_s'] - sum(c[1] and (g['conc'][i + 1][0] - c[0]) or 0
                             for i, c in enumerate(g['conc'][:-1]))) < 1e-6,
       'busy_s is wall time with >=1 process on the CPU (a time, unlike net_cpu_s)')
    ck(all(c[1] != g['conc'][i - 1][1] for i, c in enumerate(g['conc']) if i), 'concurrency series is compressed')
    ck(g['conc'][-1][1] == 0, 'concurrency returns to zero')
    ck(json.loads(json.dumps(g)) == g, 'the record is JSON-serialisable')
    # the wall-clock stream the old pool wrote (time.time()): must be detected, not drawn as if aligned
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join(
            json.dumps({'t': 1.785e9 + t, 'ev': ev, 'jid': 'w1', 'layer': 'L2', 'key': 'go build', 'cpu_ms': 500})
            for t, ev in ((1, 'QUEUED'), (2, 'RUNNING'), (8, 'FINISHED'), (9, 'EXPIRED'))) + '\n')
        w = B.build(p, T0, 60.0, 'task-A')
    ck(w['warn'] == 'clock', 'a time.time() stream is flagged, not silently misplaced')
    ck(w['j'][0]['s'] == 1.0 and w['j'][0]['e'] == 7.0, 'the flagged lane is anchored on its own first event')

    # a file shared by two tasks (SPEC_EVENTS_DIR pointed at a slot, not a task)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0 + 0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 'a1', layer='l2', key='mine'), E(2, 'RUNNING', 'a1'), E(3, 'EXPIRED', 'a1'),
            json.dumps({'t': T0 + 10, 'ev': 'SESSION', 'iid': 'task-B', 'pid': 2}),
            E(11, 'QUEUED', 'b1', layer='l2', key='theirs'), E(12, 'RUNNING', 'b1'), E(13, 'EXPIRED', 'b1'),
        ]) + '\n')
        x = B.build(p, T0, 60.0, 'task-A')
    ck(x['warn'] == 'mixed', 'a file mixing two tasks is flagged')
    ck([j['jid'] for j in x['j']] == ['a1'], "only this task's jobs survive the session window")

    # a job that never reported cpu_ms: missing must not look like "burned nothing"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 'n1', layer='L2', key='make'), E(2, 'RUNNING', 'n1'),
            E(8, 'FINISHED', 'n1'), E(9, 'EXPIRED', 'n1', why='ttl'),
            E(1, 'QUEUED', 'z1', layer='L2', key='true'), E(2, 'RUNNING', 'z1', cpu_ms=0),
            E(3, 'FINISHED', 'z1', cpu_ms=0), E(4, 'EXPIRED', 'z1', cpu_ms=0, why='ttl'),
        ]) + '\n')
        u = B.build(p, T0, 60.0, 'task-A')
    ubj = {j['jid']: j for j in u['j']}
    ck(ubj['n1']['cm'] == 0 and ubj['z1']['cm'] == 1,
       'a job with no cpu_ms at all is marked unmeasured; a measured 0 is not')
    ck(u['m']['cpu_unmeasured'] == 1, 'unmeasured CPU is counted, so the CPU tile can say it is a floor')
    ck(u['m']['cpu_unexplained'] == 1 and u['m']['cpu_short'] == 0 and u['m']['cpu_noted'] == 0,
       'a 6 s job with no cpu_ms is UNEXPLAINED: long enough to sample, no note saying why')

    # WHY a cpu_ms is missing.  Three disjoint buckets, because 'no measurement' is not one thing:
    #   s1  ran 0.2 s -- shorter than one sampling period, the sampler could not have caught it
    #   s2  ran 6 s, and the sampler itself logged that it never resolved the job's host pid
    #   s3  ran 6 s, no note: unexplained
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 's1', layer='l3', key='git status'), E(1.1, 'RUNNING', 's1'),
            E(1.3, 'FINISHED', 's1'), E(2, 'EXPIRED', 's1', why='ttl'),
            E(1, 'QUEUED', 's2', layer='l3', key='go build'), E(2, 'RUNNING', 's2'),
            json.dumps({'t': T0 + 5, 'ev': 'NOTE', 'jid': 's2', 'why': 'cpu-unresolved', 'tries': 40}),
            E(8, 'FINISHED', 's2'), E(9, 'EXPIRED', 's2', why='ttl'),
            E(1, 'QUEUED', 's3', layer='l3', key='make'), E(2, 'RUNNING', 's3'),
            E(8, 'FINISHED', 's3'), E(9, 'EXPIRED', 's3', why='ttl'),
        ]) + '\n')
        cw = B.build(p, T0, 60.0, 'task-A')
    cm2, cj = cw['m'], {j['jid']: j for j in cw['j']}
    ck(cm2['cpu_unmeasured'] == 3 and cm2['cpu_short'] == 1 and cm2['cpu_noted'] == 1
       and cm2['cpu_unexplained'] == 1,
       'the three missing-CPU buckets are disjoint and add up: %s' % cm2)
    ck(cj['s2']['cw'] == 'cpu-unresolved' and cj['s1']['cw'] == '' ,
       "the sampler's own give-up note is carried on the job, so the page can say WHY")
    ck(cm2['sample_s'] == B.CPU_SAMPLE_S and B.build(p, T0, 60.0, 'task-A', 0.05)['m']['cpu_short'] == 0,
       'the short-job bucket is the SAMPLING PERIOD, not a magic number: a 0.05 s sampler '
       'would have caught the 0.2 s job, and then it is unexplained instead')
    ck(cm2['cpu_s'] == 0.0 and cm2['lay'][0]['cpu_unmeasured'] == 3,
       'cpu_s is 0 only because there is nothing to add -- the unmeasured count is what says so')

    # the layer vocabulary: three producers, one roster; an unknown label is kept, not binned
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 'p1', layer='copy_path', key='pytest a'), E(2, 'RUNNING', 'p1', cpu_ms=0),
            E(6, 'FINISHED', 'p1', cpu_ms=2000), E(7, 'EXPIRED', 'p1'),
            E(1, 'QUEUED', 'p2', layer='const', key='pytest b'), E(2, 'RUNNING', 'p2', cpu_ms=0),
            E(6, 'FINISHED', 'p2', cpu_ms=1000), E(7, 'EXPIRED', 'p2'),
            E(10, 'QUEUED', 't1a', layer='spork', key='go test'), E(11, 'RUNNING', 't1a', cpu_ms=0),
            E(20, 'FINISHED', 't1a', cpu_ms=9000),
            E(21, 'CLAIMED', 't1a', claimed_by='go test', hidden_ms=9000, waited_ms=0, cpu_ms=9000),
            E(30, 'QUEUED', 'x1', layer='mystery9', key='???'), E(31, 'RUNNING', 'x1', cpu_ms=0),
            E(35, 'FINISHED', 'x1', cpu_ms=400), E(36, 'EXPIRED', 'x1'),
            # loop-guard: a technique that acts WITHOUT launching a process (no jid, has a layer)
            json.dumps({'t': T0 + 12, 'ev': 'NOTE', 'layer': 'lg', 'why': 'loop_guard',
                        'key': 'bash ls -la', 'run': 3, 'k': 3, 'call': 7, 'fired': 1}),
            json.dumps({'t': T0 + 40, 'ev': 'NOTE', 'layer': 'lg', 'why': 'loop_guard_summary',
                        'calls': 40, 'armed': 2, 'notes': 1, 'k': 3, 'keys': 1}),
        ]) + '\n')
        v = B.build(p, T0, 60.0, 'task-A')
    VL = {x['id']: x for x in v['m']['lay']}
    ck(sorted(VL) == ['loopguard', 'mystery9', 'paste', 't1'],
       'spork -> t1, copy_path/const -> paste, lg -> loopguard, and an unknown label keeps its '
       'own row instead of joining someone else: %s' % sorted(VL))
    ck(VL['paste']['n'] == 2 and VL['paste']['raw'] == ['const', 'copy_path'],
       "the binning keeps PASTE's mapper families visible in `raw`: %s" % VL['paste']['raw'])
    ck(VL['mystery9']['known'] == 0 and VL['t1']['known'] == 1,
       'an unrecognised label is FLAGGED, so a producer emitting a new name is visible')
    ck(VL['t1']['claimed'] == 1 and abs(VL['t1']['hidden_s'] - 9.0) < 1e-6
       and VL['paste']['claimed'] == 0 and abs(VL['paste']['cpu_wasted_s'] - 3.0) < 1e-6,
       'T1 hid 9 s; PASTE burned 3 CPU-s and hid nothing -- both halves, per technique')
    ck(VL['loopguard']['n'] == 0 and VL['loopguard']['acts'] == 1,
       'loop-guard launches no process: it gets a row with 0 jobs and 1 action, never a blank')
    ck([x['why'] for x in v['at']] == ['loop_guard'] and v['at'][0]['t'] == 12.0
       and v['at'][0]['l'] == 'loopguard' and v['at'][0]['call'] == 7 and v['at'][0]['run'] == 3,
       'a technique action is an EVENT with a time and the call it fired on: %s' % v['at'])
    ck([x['why'] for x in v['sm']] == ['loop_guard_summary'] and v['sm'][0]['calls'] == 40,
       'the per-task summary note is kept APART from the timed actions')
    ck(v['m']['acts'] == 1 and v['m']['sums'] == 1 and VL['loopguard']['sums'] == 1,
       'the per-task summary note is counted apart from the timed actions')
    ck(json.loads(json.dumps(v)) == v, 'the record with technique rows and actions is serialisable')

    # the worker's one retry: two attempts appended to the SAME file (mode 'a')
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        a1 = [json.dumps({'t': T0 + 0.5, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 111}),
              E(1, 'QUEUED', 'j111.1', layer='L2', key='go build'),
              E(2, 'RUNNING', 'j111.1', cpu_ms=0),
              E(8, 'FINISHED', 'j111.1', cpu_ms=6000),
              E(9, 'CLAIMED', 'j111.1', cpu_ms=6000, claimed_by='go build', hidden_ms=6000, waited_ms=0)]
        a2 = [json.dumps({'t': T0 + 200.0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 222}),
              E(201, 'QUEUED', 'j222.1', layer='L2', key='go build'),
              E(202, 'RUNNING', 'j222.1', cpu_ms=0),
              E(208, 'FINISHED', 'j222.1', cpu_ms=6000),
              E(209, 'CLAIMED', 'j222.1', cpu_ms=6000, claimed_by='go build', hidden_ms=6000, waited_ms=0)]
        open(p, 'w').write('\n'.join(a1 + a2) + '\n')
        rr = B.build(p, T0, 300.0, 'task-A')
    ck(rr['warn'] == 'rerun' and rr['m']['attempts'] == 2, 'two attempts in one file are detected')
    ck([j['jid'] for j in rr['j']] == ['j222.1'], 'only the last attempt is kept')
    ck(rr['m']['n'] == 1 and abs(rr['m']['cpu_s'] - 6.0) < 1e-6 and abs(rr['m']['hidden_s'] - 6.0) < 1e-6,
       'launched / CPU / hidden are NOT doubled by the retry: %s' % rr['m'])
    # ...and a single attempt whose pool opened the log from two processes is NOT a rerun
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0 + 0.5, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 111}),
            E(1, 'QUEUED', 'j111.1', layer='L2', key='go build'), E(2, 'RUNNING', 'j111.1'),
            json.dumps({'t': T0 + 3.0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 112}),
            E(4, 'QUEUED', 'j112.1', layer='L2', key='go test'), E(5, 'RUNNING', 'j112.1'),
            E(9, 'FINISHED', 'j111.1'), E(10, 'FINISHED', 'j112.1'),
        ]) + '\n')
        nr = B.build(p, T0, 60.0, 'task-A')
    ck(nr['warn'] == '' and nr['m']['n'] == 2,
       'two processes of ONE attempt are not mistaken for a retry: %r %d' % (nr['warn'], nr['m']['n']))

    # a stale file from an earlier run of the same task: same clock, no overlap with this run
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0 + 27594.0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 9}),
            E(27595, 'QUEUED', 'o1', layer='L2', key='go build'),
            E(27596, 'RUNNING', 'o1'), E(27600, 'EXPIRED', 'o1'),
        ]) + '\n')
        off = B.build(p, T0, 60.0, 'task-A')
    ck(off['warn'] == 'offaxis' and 'offaxis' in off['warns'],
       'events 7.7 h off the run are flagged, not drawn as if they belonged to it: %r' % off['warn'])

    # ---- MEMO serves: NOTE why=serve (paste_hook._PasteState._serve_note) -> cl[] + ledger ----
    # The L1 auth-cache answers a repeat from the cached authoritative run: no job, no CLAIMED, so
    # before 2026-09-08 the lane and the table showed L1 = 0 while the stats json showed 61.9 s.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 'k1', layer='l3', key='yarn check-types'), E(2, 'RUNNING', 'k1', cpu_ms=0),
            E(9, 'FINISHED', 'k1', cpu_ms=7000),
            E(10, 'CLAIMED', 'k1', claimed_by='yarn check-types', why='l3_filter', hidden_ms=7000,
              waited_ms=200, cpu_ms=7000),
            json.dumps({'t': T0 + 20, 'ev': 'NOTE', 'why': 'serve', 'layer': 'l1', 'kind': 'auth_filter',
                        'hidden_ms': 10601.8, 'waited_ms': 202.3, 'key': 'cd /app && yarn test a.test.ts 2>&1',
                        'call': 47, 'epoch': 3, 'memo': 1}),
            json.dumps({'t': T0 + 30, 'ev': 'NOTE', 'why': 'serve', 'layer': 'l1', 'kind': 'auth_exact',
                        'hidden_ms': 10186.0, 'waited_ms': 0.0, 'key': 'cd /app && yarn test a.test.ts 2>&1',
                        'call': 52, 'epoch': 3, 'memo': 1, 'via': 'l3'}),
            # a job result re-served after its job closed: the engine records the job's own layer
            json.dumps({'t': T0 + 40, 'ev': 'NOTE', 'why': 'serve', 'layer': 'l3', 'kind': 'l3_exact',
                        'hidden_ms': 5000.0, 'waited_ms': 10.0, 'key': 'yarn check-types', 'call': 60, 'memo': 1}),
        ]) + '\n')
        ms = B.build(p, T0, 60.0, 'task-A')
    ML = {x['id']: x for x in ms['m']['lay']}
    ck(sorted(ML) == ['l1', 'l3'], 'a technique that only served from memory gets its own row: %s' % sorted(ML))
    ck(ML['l1']['n'] == 0 and ML['l1']['claimed'] == 0 and ML['l1']['acts'] == 0 and ML['l1']['memo'] == 2,
       'L1: 0 jobs, 0 claims, 0 actions -- and 2 memo serves (a serve NOTE is neither a job nor an action)')
    ck(abs(ML['l1']['memo_hidden_s'] - 20.79) < 0.011 and abs(ML['l1']['hidden_s'] - 20.79) < 0.011,
       "L1's hidden seconds are its memo serves: %s / %s" % (ML['l1']['memo_hidden_s'], ML['l1']['hidden_s']))
    ck(ML['l3']['claimed'] == 1 and ML['l3']['memo'] == 1 and abs(ML['l3']['hidden_s'] - 12.0) < 1e-6
       and abs(ML['l3']['memo_hidden_s'] - 5.0) < 1e-6,
       'L3: hidden_s = the 7 s job claim + the 5 s re-served result; claimed stays 1: %s' % ML['l3'])
    ck(ms['m']['claimed'] == 1 and ms['m']['memo'] == 3 and abs(ms['m']['hidden_s'] - 32.79) < 0.011
       and abs(ms['m']['memo_hidden_s'] - 25.79) < 0.011,
       'run totals: claimed = job claims only (1); hidden_s = claims + memo (32.79); memo = 3: %s' % (
           {k: ms['m'][k] for k in ('claimed', 'memo', 'hidden_s', 'memo_hidden_s')}))
    ck(ms['m']['measured_hidden'] == 1, 'measured_hidden still counts job claims only')
    ck(ms['m']['memo_src'] == 'events' and ms['m']['memo_unaligned'] == 0,
       'the source of the memo records is named (events), nothing unaligned')
    ck(abs(ms['m']['waited_s'] - 0.41) < 0.02 and abs(ms['m']['net_wall_s'] - (32.79 - 0.41 - 7.0)) < 0.05,
       'waited_s and net_wall_s follow hidden_s (busy 7 s = the one job): %s / %s' % (ms['m']['waited_s'], ms['m']['net_wall_s']))
    memo = [c for c in ms['cl'] if c.get('memo')]
    ck(len(ms['cl']) == 4 and len(memo) == 3 and [c['t'] for c in ms['cl']] == [10.0, 20.0, 30.0, 40.0],
       'claims and memo serves share cl[] in time order: %s' % [(c['t'], c.get('memo')) for c in ms['cl']])
    ck(memo[0]['i'] is None and memo[0]['l'] == 'l1' and memo[0]['kind'] == 'auth_filter' and memo[0]['hid'] == 10.6
       and memo[0]['wait'] == 0.2 and memo[0]['call'] == 47 and memo[0]['src'] == 'events'
       and memo[0]['by'] == 'cd /app && yarn test a.test.ts 2>&1',
       'a memo tick carries no job index, its technique, kind, hidden/waited, the call and the command: %s' % memo[0])
    ck(memo[1]['via'] == 'l3', 'the auth entry seeded by an L2/L3 job keeps its `via`')
    ck(memo[0]['row'] is None and memo[2]['row'] == ML['l3'] and False or (memo[0]['row'] is None
       and memo[2]['row'] == [g0 for g0 in ms['m']['grp'] if g0[0] == 'l3'][0][1]),
       "a memo tick sits on its technique's band when it has one (L3), on none for L1 (no jobs): %s / %s" % (memo[0]['row'], memo[2]['row']))
    ck(ms['at'] == [] and ms['m']['acts'] == 0, 'no action is synthesised from a serve NOTE')
    ck(json.loads(json.dumps(ms)) == ms, 'the record with memo serves is serialisable')

    # ---- the STATS fallback: a stream recorded before NOTE why=serve existed -------------------
    # per_call rows are aligned to tool_calls by COMMAND TEXT, monotone, over ALL rows: the engine
    # keeps bookkeeping rows this builder strips ('true', 'git -C ... config'), so indices differ;
    # and the served command is a REPEAT, so matching only served rows would land on its first
    # occurrence -- the authoritative run the memo replayed -- not on the replay.
    LONG = 'cd /app && node -e ' + ('x' * 440)                # per_call keeps command[:400]
    tcs = [{'command': 'true', 'start_s': T0 - 5, 'end_s': T0 - 4.9},
           {'command': 'git -C /app config core.fileMode false', 'start_s': T0 - 4, 'end_s': T0 - 3.9},
           {'command': 'yarn check-types', 'start_s': T0 + 8, 'end_s': T0 + 10},
           {'command': 'cd /app && yarn test a.test.ts 2>&1', 'start_s': T0 + 15, 'end_s': T0 + 25},
           {'command': 'cd /app && yarn test a.test.ts 2>&1', 'start_s': T0 + 30, 'end_s': T0 + 30.2},
           {'command': 'cd /app && yarn test a.test.ts 2>&1', 'start_s': T0 + 35, 'end_s': T0 + 35.2},
           {'command': LONG, 'start_s': T0 + 38, 'end_s': T0 + 38.1},
           {'command': 'ls -la', 'start_s': T0 + 40, 'end_s': T0 + 40.1}]
    stats = {'per_call': [
        {'cmd': 'true', 'hit': None, 'lat_ms': 100},
        {'cmd': 'git -C /app config core.fileMode false', 'hit': None, 'lat_ms': 100},
        {'cmd': 'yarn check-types', 'hit': 'l3_filter', 'hidden_ms': 7000, 'waited_ms': 200},   # job-backed
        {'cmd': 'cd /app && yarn test a.test.ts 2>&1', 'hit': None, 'lat_ms': 10000},
        {'cmd': 'cd /app && yarn test a.test.ts 2>&1', 'hit': 'auth_filter', 'hidden_ms': 10601.8, 'waited_ms': 202.3},
        {'cmd': 'cd /app && yarn test a.test.ts 2>&1', 'hit': 'auth_exact', 'hidden_ms': 10186.0, 'waited_ms': 0.0},
        {'cmd': LONG[:400], 'hit': 'auth_exact', 'hidden_ms': 3000.0, 'waited_ms': 0.0},        # truncated cmd
        {'cmd': 'ls -la', 'hit': None, 'lat_ms': 50},
        {'cmd': 'never issued', 'hit': 'auth_exact', 'hidden_ms': 999.0, 'waited_ms': 0.0},     # unaligned
    ]}
    OLD = [json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
           E(1, 'QUEUED', 'k1', layer='l3', key='yarn check-types'), E(2, 'RUNNING', 'k1', cpu_ms=0),
           E(9, 'FINISHED', 'k1', cpu_ms=7000),
           E(10, 'CLAIMED', 'k1', claimed_by='yarn check-types', why='l3_filter', hidden_ms=7000,
             waited_ms=200, cpu_ms=7000)]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join(OLD) + '\n')
        fb = B.build(p, T0, 60.0, 'task-A', B.CPU_SAMPLE_S, stats, tcs)
        nostats = B.build(p, T0, 60.0, 'task-A')
    FL = {x['id']: x for x in fb['m']['lay']}
    ck('l1' not in {x['id'] for x in nostats['m']['lay']} and nostats['m']['memo'] == 0,
       'without the stats json an old stream still shows no L1 (nothing is invented)')
    ck(fb['m']['memo'] == 3 and fb['m']['memo_src'] == 'stats' and fb['m']['memo_unaligned'] == 1,
       'fallback: 3 memo serves from the stats json, 1 served row matched no tool call and is reported: %s' % (
           {k: fb['m'].get(k) for k in ('memo', 'memo_src', 'memo_unaligned')}))
    fm = [c for c in fb['cl'] if c.get('memo')]
    ck([c['t'] for c in fm] == [30.0, 35.0, 38.0],
       'memo ticks sit on the REPLAYS (30, 35, 38), not on the first occurrence (15) or the bookkeeping rows: %s' % [c['t'] for c in fm])
    ck(all(c['src'] == 'stats' for c in fm) and [c['call'] for c in fm] == [5, 6, 7],
       "synthesised records are marked src=stats and carry the engine's 1-based call index: %s" % [(c['src'], c['call']) for c in fm])
    ck(fm[2]['by'] == LONG[:200] and fm[2]['kind'] == 'auth_exact',
       'a per_call cmd truncated to 400 chars still aligns to its (longer) tool call')
    ck(FL['l1']['memo'] == 3 and abs(FL['l1']['hidden_s'] - 23.79) < 0.011 and FL['l1']['claimed'] == 0,
       'L1 from stats: 3 serves, 23.79 s hidden, 0 claims: %s' % {k: FL['l1'][k] for k in ('memo', 'hidden_s', 'claimed')})
    ck(FL['l3']['memo'] == 0 and FL['l3']['claimed'] == 1 and abs(FL['l3']['hidden_s'] - 7.0) < 1e-6,
       'the l3_filter row is JOB-BACKED (a CLAIMED with that why inside its call window): not counted twice')
    ck(len(fb['cl']) == 4 and abs(fb['m']['hidden_s'] - 30.79) < 0.011,
       'cl[] = 1 claim + 3 memo serves; run hidden_s = 7 + 23.79')
    # the stream's own NOTEs win: with even one why=serve line the stats json is not consulted
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join(OLD + [
            json.dumps({'t': T0 + 30, 'ev': 'NOTE', 'why': 'serve', 'layer': 'l1', 'kind': 'auth_filter',
                        'hidden_ms': 10601.8, 'waited_ms': 202.3, 'key': 'cd /app && yarn test a.test.ts 2>&1',
                        'call': 5, 'memo': 1})]) + '\n')
        pref = B.build(p, T0, 60.0, 'task-A', B.CPU_SAMPLE_S, stats, tcs)
    ck(pref['m']['memo'] == 1 and pref['m']['memo_src'] == 'events' and pref['m']['memo_unaligned'] == 0,
       'a stream that carries its own serve NOTEs is never topped up from the stats json: %s' % (
           {k: pref['m'].get(k) for k in ('memo', 'memo_src', 'memo_unaligned')}))
    # a CLAIMED with no `why` (older emitter) inside the call window still counts as job-backed
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 'k1', layer='l3', key='yarn check-types'), E(2, 'RUNNING', 'k1', cpu_ms=0),
            E(9, 'FINISHED', 'k1', cpu_ms=7000),
            E(10, 'CLAIMED', 'k1', claimed_by='yarn check-types', hidden_ms=7000, waited_ms=200, cpu_ms=7000)]) + '\n')
        nw = B.build(p, T0, 60.0, 'task-A', B.CPU_SAMPLE_S, stats, tcs)
    ck({x['id']: x['memo'] for x in nw['m']['lay']}.get('l3', 0) == 0 and nw['m']['memo'] == 3,
       'a why-less CLAIMED in the window is still recognised as the job behind the served row')
    # ---- K-step probe jobs (09-08 grid v2): spork_d<K> is T1 on the page, the depth stays on the job ----
    ck(B.tech_of('spork_d2') == ('t1', True) and B.tech_of('spork_d3') == ('t1', True)
       and B.tech_of('spork_d4') == ('t1', True) and B.tech_of('SPORK_D2') == ('t1', True)
       and B.tech_of('spork_d9') == ('spork_d9', False),
       'spork_d2/d3/d4 bin to t1 as known labels; a depth the roster does not know stays its own flagged row')
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'spec_events.jsonl')
        open(p, 'w').write('\n'.join([
            json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
            E(1, 'QUEUED', 'd1', layer='spork', key='go test'), E(2, 'RUNNING', 'd1', cpu_ms=0),
            E(5, 'FINISHED', 'd1', cpu_ms=3000), E(6, 'EXPIRED', 'd1'),
            E(1, 'QUEUED', 'd2', layer='spork_d2', key='go vet'), E(2, 'RUNNING', 'd2', cpu_ms=0),
            E(4, 'FINISHED', 'd2', cpu_ms=2000),
            E(7, 'CLAIMED', 'd2', claimed_by='go vet', hidden_ms=2000, waited_ms=0, cpu_ms=2000),
            E(1, 'QUEUED', 'd3', layer='spork_d3', key='go build'), E(2, 'RUNNING', 'd3', cpu_ms=0),
            E(3, 'FINISHED', 'd3', cpu_ms=1000), E(9, 'DISCARDED', 'd3'),
        ]) + '\n')
        v = B.build(p, T0, 60.0, 'task-A')
    VL = {x['id']: x for x in v['m']['lay']}
    ck(list(VL) == ['t1'] and VL['t1']['n'] == 3 and VL['t1']['claimed'] == 1 and VL['t1']['known'] == 1
       and VL['t1']['raw'] == ['spork', 'spork_d2', 'spork_d3'] and abs(VL['t1']['hidden_s'] - 2.0) < 1e-6,
       'K-step probe jobs land in the ONE T1 row (3 jobs, 1 claimed, 2.0 s) and the raw labels keep the depth: %s'
       % (VL['t1'].get('raw'),))
    ck(sorted((x['lr'], x['l']) for x in v['j']) == [('spork', 't1'), ('spork_d2', 't1'), ('spork_d3', 't1')]
       and [g[0] for g in v['m']['grp']] == ['t1'],
       'each job carries the technique id (l) AND its raw depth label (lr), one T1 band on the lane -- a per-depth split needs no re-fold')

    ck(B.serve_layer('exact', [{'k': 'go test ./...', 'l': 'spork'}], 'go test ./... -run X') == 'spork'
       and B.serve_layer('exact', [], 'go test') == '?' and B.serve_layer('auth_stage') == 'l1'
       and B.serve_layer('l2_filter') == 'l2',
       "a bare exact/prefix/promoted kind takes the layer of the job whose key matches, else '?' (flagged, never guessed)")

    # ---- main(): the arm dimension survives all the way to disk -----------------------------
    # Two arms of the SAME task must be reachable as a pair, or a base/full comparison needs the
    # reader to know which cells to fetch.  A tiny synthetic campaign tree, no manifest on disk.
    with tempfile.TemporaryDirectory() as td:
        ev = '\n'.join([json.dumps({'t': T0, 'ev': 'SESSION', 'iid': 'task-A', 'pid': 1}),
                         E(1, 'QUEUED', 'q1', layer='spork', key='go test'),
                         E(2, 'RUNNING', 'q1', cpu_ms=0), E(9, 'FINISHED', 'q1', cpu_ms=7000),
                         E(10, 'CLAIMED', 'q1', claimed_by='go test', hidden_ms=7000, waited_ms=0,
                           cpu_ms=7000)]) + '\n'
        tc = json.dumps({'tool': 'bash', 'command': 'go test ./...', 'start_s': T0,
                         'end_s': T0 + 20, 'latency_s': 20}) + '\n' + json.dumps(
                             {'tool': 'bash', 'command': 'go test ./...', 'start_s': T0 + 30,
                              'end_s': T0 + 30.1, 'latency_s': 0.1}) + '\n'
        for arm in ('base', 'full'):
            d = os.path.join(td, 'camp', arm, 'pro', 'coder', 'hermes', 'traj', 'task-A')
            os.makedirs(d)
            open(os.path.join(d, 'spec_events.jsonl'), 'w').write(ev)
            open(os.path.join(d, B.T.TC), 'w').write(tc)
        # the full arm's engine stats: the CLAIMED job (why-less here) + one memo replay of it
        sd = os.path.join(td, 'camp', 'full', 'pro', 'coder', 'hermes', 'stats')
        os.makedirs(sd)
        json.dump({'per_call': [{'cmd': 'go test ./...', 'hit': 'exact', 'hidden_ms': 7000, 'waited_ms': 0},
                                {'cmd': 'go test ./...', 'hit': 'auth_exact', 'hidden_ms': 20000, 'waited_ms': 0}]},
                  open(os.path.join(sd, 'task-A.json'), 'w'))
        B.T._MAN['final_pro'] = {'task-A': ('/app', 'repo')}      # no manifest file needed
        out = os.path.join(td, 'out')
        argv = sys.argv
        sys.argv = ['build_specdata.py', '--datasets', 'final_pro', '--camp-root',
                    os.path.join(td, 'camp'), '--out', out]
        try:
            B.main()
        finally:
            sys.argv = argv
        idx = json.load(open(os.path.join(out, 'index.json')))
        arms = json.load(open(os.path.join(out, 'arms.json')))
        one = json.load(open(os.path.join(out, 'final_pro', 'coder__hermes__full', 'task-A.json')))
    ck(sorted(idx['final_pro']) == ['coder__hermes__base', 'coder__hermes__full'],
       'both arms are built into their own cells: %s' % sorted(idx['final_pro']))
    ck(idx['final_pro']['coder__hermes__base'][0][:5] == ['task-A', 1, 1, 7.0, 7.0],
       'the first five index columns are unchanged (older readers keep working)')
    ck(idx['final_pro']['coder__hermes__full'][0][:5] == ['task-A', 1, 1, 7.0, 27.0]
       and idx['final_pro']['coder__hermes__full'][0][8:10] == [1, 20.0],
       'hidden (column 4) includes the memo serve; columns 8-9 say how many and how much: %s' % idx['final_pro']['coder__hermes__full'][0])
    ck(idx['final_pro']['coder__hermes__full'][0][5] == 'full'
       and idx['final_pro']['coder__hermes__full'][0][7] == {'l1': [0, 0, 0], 't1': [1, 1, 0]},
       'the arm and the per-technique launched/claimed map are appended to the index row: %s' % idx['final_pro']['coder__hermes__full'][0][7])
    ck(sorted(arms['final_pro']['task-A']['coder__hermes']) == ['base', 'full'],
       'arms.json is keyed task -> mode__fw -> arm, so a pair is one lookup')
    ck(arms['final_pro']['task-A']['coder__hermes']['base']['lay']['t1']['hidden_s'] == 7.0,
       'the per-technique split is carried per arm, so the two can be compared technique by technique')
    fa, ba = arms['final_pro']['task-A']['coder__hermes']['full'], arms['final_pro']['task-A']['coder__hermes']['base']
    ck(fa['memo'] == 1 and fa['memo_hidden_s'] == 20.0 and fa['memo_src'] == 'stats' and fa['hidden_s'] == 27.0
       and fa['lay']['l1'] == {'n': 0, 'claimed': 0, 'hidden_s': 20.0, 'cpu_s': 0.0, 'acts': 0, 'cpu_unmeasured': 0,
                               'memo': 1, 'memo_hidden_s': 20.0}
       and ba['memo'] == 0 and ba['hidden_s'] == 7.0 and 'l1' not in ba['lay'],
       'arms.json carries the memo serves per arm and per technique (full: 1 from stats; base: none): %s' % fa)
    ck(one['m']['memo'] == 1 and [c for c in one['cl'] if c.get('memo')][0]['t'] == 30.0,
       'the per-task file has the memo tick on the replay (t=30), not on the claimed first run')
    ck(one['arm'] == 'full' and one['mode'] == 'coder' and one['fw'] == 'hermes',
       'the per-task file carries its own arm')

    # ---- main(): index.json / arms.json are PER-DATASET merges ------------------------------
    # The driver rebuilds ONE dataset per finished pair (ops/grid12_dash.sh); until 09-08 21:00 both
    # files were rewritten from that run alone, so grid12_pro vanished from the page's selector
    # while data/spec/grid12_pro/ stayed intact.  Build A, then B, then A again (changed), then A
    # against an empty root: B must survive every step, A must be replaced, never duplicated, and
    # a rebuild that finds nothing must keep what it had.
    with tempfile.TemporaryDirectory() as td:
        def camp(name, iid):
            d = os.path.join(td, name, 'full', 'pro', 'coder', 'hermes', 'traj', iid)
            os.makedirs(d)
            open(os.path.join(d, 'spec_events.jsonl'), 'w').write(ev)
            open(os.path.join(d, B.T.TC), 'w').write(tc)
            return os.path.join(td, name)
        out = os.path.join(td, 'out')

        def run(ds, root):
            argv = sys.argv
            sys.argv = ['build_specdata.py', '--datasets', ds, '--camp-root', root, '--out', out, '--arms', 'full']
            try:
                B.main()
            finally:
                sys.argv = argv
            return (json.load(open(os.path.join(out, 'index.json'))),
                    json.load(open(os.path.join(out, 'arms.json'))))
        first = lambda idx, ds: idx[ds]['coder__hermes__full'][0][0]          # noqa: E731
        cA, cB = camp('campA', 'task-A'), camp('campB', 'task-B')
        B.T._MAN['final_pro'] = {'task-A': ('/app', 'repo')}
        B.T._MAN['grid12_pro'] = {'task-B': ('/app', 'repo')}
        run('final_pro', cA)
        idx, arms = run('grid12_pro', cB)
        ck(sorted(idx) == ['final_pro', 'grid12_pro'] and sorted(arms) == ['final_pro', 'grid12_pro']
           and first(idx, 'final_pro') == 'task-A' and first(idx, 'grid12_pro') == 'task-B'
           and 'task-A' in arms['final_pro'] and 'task-B' in arms['grid12_pro'],
           'building dataset B after dataset A keeps A in index.json AND arms.json: %s' % sorted(idx))
        cA2 = camp('campA2', 'task-A2')
        B.T._MAN['final_pro'] = {'task-A2': ('/app', 'repo')}
        idx, arms = run('final_pro', cA2)
        ck(first(idx, 'final_pro') == 'task-A2' and len(idx['final_pro']['coder__hermes__full']) == 1
           and sorted(arms['final_pro']) == ['task-A2'] and first(idx, 'grid12_pro') == 'task-B'
           and sorted(arms['grid12_pro']) == ['task-B'],
           'rebuilding A REPLACES A\'s key (no stale task-A, no duplicate) and leaves B byte-for-byte: %s'
           % sorted(arms['final_pro']))
        idx, arms = run('final_pro', os.path.join(td, 'nowhere'))
        ck(sorted(idx) == ['final_pro', 'grid12_pro'] and first(idx, 'final_pro') == 'task-A2'
           and sorted(arms['final_pro']) == ['task-A2'] and first(idx, 'grid12_pro') == 'task-B',
           'a rebuild that finds NOTHING (wrong --camp-root) keeps the previous entries of both files instead of dropping the dataset')
        ck(os.path.exists(os.path.join(out, '.index.lock')) and not os.path.exists(os.path.join(out, 'index.json.tmp')),
           'the merge took its lock file and left no temp file behind')
        B.T._MAN.pop('grid12_pro', None)

    # ---- 2026-09-09: the trace-replay pass (tag v4r) is registered as a SIBLING of the six-arm dataset: same manifest
    # shape, layout, root and arm table (ops/arms.tsv); only the name differs (and with it the results root the dash
    # passes).  build_tldata / build_specdata / build_hwdata read these tables through build_tagdata, so one check
    # covers every builder; a dataset missing from any table would silently build nothing.
    T = B.T
    ck(T.DATASETS.get('grid12v4r_pro') == 'final_test/grid12v4r_pro.jsonl' and T.DATASETS.get('grid12v4_pro') == 'final_test/grid12v4_pro.jsonl',
       'grid12v4r_pro is registered next to grid12v4_pro with its own manifest')
    ck(all('grid12v4r_pro' in tbl and tbl['grid12v4r_pro'] == tbl['grid12v4_pro'] for tbl in (T.DSDIR, T.DSROOT, T.DSTPL)),
       "the replay dataset shares grid12v4_pro's directory, root and {arm} layout")
    ck(T.DSARMS.get('grid12v4r_pro') is T.DSARMS.get('grid12v4_pro') is T.arms_tsv
       and T.arms_of('grid12v4r_pro') == T.arms_of('grid12v4_pro') == list(T.arms_tsv()) and len(T.arms_of('grid12v4r_pro')) == 6,
       'its arms are the six ops/arms.tsv rows through the same callable as grid12v4_pro: %s' % ','.join(T.arms_of('grid12v4r_pro')))
    ck(T.arms_of('grid12v4r_pro', 'base,full') == ['base', 'full'] and T.arms_of('grid12v3_pro') == ['base', 'full'],
       '--arms still overrides it and the legacy datasets keep base/full')
    # 2026-09-09 (later): the v6 pair is registered the same way -- grid12v6rec_pro (the recording pass; only the serial arm
    # exists on disk, the six-arm table is registered anyway because the builders tolerate absent arms) and grid12v6_pro
    # (its six-arm replay on the recorded Serial trajectories).  Each has its own manifest in final_test/.
    for sib in ('grid12v6rec_pro', 'grid12v6_pro'):
        ck(T.DATASETS.get(sib) == 'final_test/%s.jsonl' % sib and os.path.exists(T.manifest_path(sib) or ''),
           '%s is registered with its own manifest and the file exists (%s)' % (sib, T.manifest_path(sib)))
        ck(all(sib in tbl and tbl[sib] == tbl['grid12v4_pro'] for tbl in (T.DSDIR, T.DSROOT, T.DSTPL)),
           "%s shares grid12v4_pro's directory, root and {arm} layout" % sib)
        ck(T.DSARMS.get(sib) is T.arms_tsv and T.arms_of(sib) == T.arms_of('grid12v4_pro') and len(T.arms_of(sib)) == 6,
           '%s takes the six ops/arms.tsv rows through the same callable as grid12v4_pro: %s' % (sib, ','.join(T.arms_of(sib))))

    bad = [w for ok, w in CHECKS if not ok]
    for ok, w in CHECKS:
        print('%s %s' % ('ok  ' if ok else 'FAIL', w))
    print('%d/%d checks passed' % (len(CHECKS) - len(bad), len(CHECKS)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
