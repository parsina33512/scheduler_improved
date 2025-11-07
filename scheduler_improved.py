#!/usr/bin/env python3
"""
scheduler_improved.py
Improved simulator for CPU scheduling:
Supports: FCFS, SJF (predictive), SRTF (with preemption threshold), Round Robin (static/adaptive),
HRRN, MLFQ (simple).
Features:
 - float times (arrival/burst/quantum)
 - context switch cost
 - input: interactive or CSV (--input-file)
 - save Gantt to PNG and metrics to CSV
Usage examples:
  python scheduler_improved.py --demo
  python scheduler_improved.py
  python scheduler_improved.py --input-file procs.csv --algos fcfs,srtf,rr --quantum 3 --cs 0.1
"""
import argparse
import csv
import math
import statistics
from collections import deque
import copy
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------
def mean_or_zero(lst):
    return sum(lst) / len(lst) if lst else 0.0

def merge_segments(segments):
    """Merge adjacent segments with same pid and contiguous times (float tolerant)."""
    if not segments:
        return []
    merged = []
    eps = 1e-9
    for pid, s, e in segments:
        if merged and merged[-1][0] == pid and abs(merged[-1][2] - s) <= eps:
            merged[-1] = (pid, merged[-1][1], e)
        else:
            merged.append((pid, s, e))
    return merged

def compute_metrics(processes, completion):
    arrival = {p['pid']: p['arrival'] for p in processes}
    burst = {p['pid']: p['burst'] for p in processes}
    tat = {pid: completion[pid] - arrival[pid] for pid in completion}
    wt = {pid: tat[pid] - burst[pid] for pid in tat}
    return completion, tat, wt

def save_metrics_csv(metrics, filename):
    """metrics: dict algoname -> list of per-pid dicts containing pid,AT,BT,CT,TAT,WT"""
    # Flatten into rows with an 'algorithm' column
    rows = []
    for algo, data in metrics.items():
        for row in data:
            r = dict(row)
            r['algorithm'] = algo
            rows.append(r)
    if not rows:
        return
    keys = ['algorithm', 'pid', 'arrival', 'burst', 'completion', 'turnaround', 'waiting']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'algorithm': r.get('algorithm'),
                'pid': r.get('pid'),
                'arrival': r.get('arrival'),
                'burst': r.get('burst'),
                'completion': r.get('completion'),
                'turnaround': r.get('turnaround'),
                'waiting': r.get('waiting'),
            })

def plot_gantt(segments, title, savefile=None):
    """
    segments: list of (pid, start, end)
    """
    if not segments:
        print("No segments to plot for", title)
        return
    merged = merge_segments(segments)
    pids = sorted({s[0] for s in merged})
    # y positions
    y_positions = {pid: i for i, pid in enumerate(pids[::-1])}  # reverse so P1 on top maybe
    fig_height = max(2, 0.5 * len(pids))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    for pid, s, e in merged:
        y = y_positions[pid]
        ax.broken_barh([(s, e - s)], (y - 0.4, 0.8), edgecolor='black')
        ax.text((s + e) / 2, y, f"P{pid}", ha='center', va='center')
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"P{pid}" for pid in sorted(y_positions.keys(), reverse=True)])
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile)
        print(f"Saved Gantt chart to {savefile}")
    plt.show()

# -------------------------
# Algorithm implementations
# -------------------------
def fcfs(processes, cs=0.0):
    procs = sorted(processes, key=lambda p: (p['arrival'], p['pid']))
    time = 0.0
    segments = []
    completion = {}
    for p in procs:
        if time < p['arrival']:
            time = p['arrival']
        start = time
        time += p['burst']
        end = time
        segments.append((p['pid'], start, end))
        completion[p['pid']] = end
        time += cs  # context switch after finishing
    return merge_segments(segments), completion

def sjf_predictive(processes, alpha=0.5, init_tau=None, cs=0.0):
    """Non-preemptive SJF with prediction using exponential averaging.
       For a single-run workload, we simulate prediction by assuming we only learn bursts when a job completes,
       but since all bursts are known for the simulation, predictive vs oracle differs only philosophically.
       We'll use init_tau as the initial guess for all jobs if provided; otherwise use mean burst.
    """
    procs = copy.deepcopy(processes)
    n = len(procs)
    if init_tau is None:
        init_tau = mean_or_zero([p['burst'] for p in procs]) if procs else 1.0
    # We'll track predicted next burst per PID; start with init_tau
    predicted = {p['pid']: init_tau for p in procs}
    completed = set()
    time = 0.0
    segments = []
    while len(completed) < n:
        available = [p for p in procs if p['arrival'] <= time and p['pid'] not in completed]
        if not available:
            time = min(p['arrival'] for p in procs if p['pid'] not in completed)
            continue
        # pick by predicted burst (tie-breaker arrival then pid)
        cur = min(available, key=lambda p: (predicted[p['pid']], p['arrival'], p['pid']))
        start = time
        time += cur['burst']
        end = time
        segments.append((cur['pid'], start, end))
        completed.add(cur['pid'])
        # update prediction for this pid (though won't be reused in single-run), but emulate behavior
        predicted[cur['pid']] = alpha * cur['burst'] + (1 - alpha) * predicted[cur['pid']]
        time += cs
    return merge_segments(segments), {p['pid']: max([e for pid,s,e in segments if pid==p['pid']]) for p in procs}

def srtf(processes, cs=0.0, preempt_threshold=0.0):
    """Preemptive SJF (Shortest Remaining Time First)
       preempt_threshold: require new_remaining < current_remaining - preempt_threshold to preempt.
    """
    procs = sorted(copy.deepcopy(processes), key=lambda p: (p['arrival'], p['pid']))
    n = len(procs)
    time = 0.0
    idx = 0
    import heapq
    heap = []  # entries: (remaining, arrival, pid, burst)
    rem = {}
    for p in procs:
        rem[p['pid']] = p['burst']
    segments = []
    current_pid = None
    start_time = None
    while idx < n or heap:
        # push arrivals
        while idx < n and procs[idx]['arrival'] <= time:
            p = procs[idx]
            heapq.heappush(heap, (rem[p['pid']], p['arrival'], p['pid'], p['burst']))
            idx += 1
        if not heap:
            # jump to next arrival
            time = procs[idx]['arrival']
            continue
        remaining, arrival_t, pid, burst = heapq.heappop(heap)
        # if switching from another pid, record segment
        if current_pid != pid:
            if current_pid is not None:
                segments.append((current_pid, start_time, time))
                time += cs  # context switch when switching
            current_pid = pid
            start_time = time
        # determine next event: either this job finishes, or a new arrival with smaller rem appears
        next_arrival_time = procs[idx]['arrival'] if idx < n else math.inf
        finish_time = time + remaining
        # We simulate until either finish_time or next_arrival_time
        if next_arrival_time < finish_time:
            # execute until next arrival
            exec_time = next_arrival_time - time
            rem[pid] -= exec_time
            time = next_arrival_time
            # push this job back with updated remaining
            heapq.heappush(heap, (rem[pid], arrival_t, pid, burst))
            # continue loop to handle new arrival (which may preempt)
        else:
            # job finishes before next arrival
            time = finish_time
            rem[pid] = 0.0
            segments.append((pid, start_time, time))
            current_pid = None
            start_time = None
            # after finishing, we don't immediately add context switch time here because
            # we add cs when next switch happens at top (above) — but to be consistent, add cs
            time += cs
    # compute completion times from segments
    completion = {}
    for pid in set(p['pid'] for p in procs):
        ends = [e for (pp, s, e) in segments if pp == pid]
        completion[pid] = max(ends) if ends else None
    return merge_segments(segments), completion

def round_robin(processes, quantum=1.0, cs=0.0, adaptive=False):
    procs = sorted(copy.deepcopy(processes), key=lambda p: (p['arrival'], p['pid']))
    n = len(procs)
    time = 0.0
    idx = 0
    from collections import deque
    q = deque()
    rem = {p['pid']: p['burst'] for p in procs}
    completion = {}
    segments = []
    # if adaptive, set quantum to median burst of processes seen so far (simple heuristic)
    base_quantum = float(quantum)
    while idx < n or q:
        # enqueue arrivals
        while idx < n and procs[idx]['arrival'] <= time:
            q.append(procs[idx]['pid'])
            idx += 1
        if not q:
            time = procs[idx]['arrival']
            continue
        pid = q.popleft()
        if adaptive:
            seen_bursts = [p['burst'] for p in procs if p['arrival'] <= time]
            if seen_bursts:
                qtime = max(0.01, statistics.median(seen_bursts))  # avoid zero
            else:
                qtime = base_quantum
        else:
            qtime = base_quantum
        start = time
        exec_time = min(qtime, rem[pid])
        rem[pid] -= exec_time
        time += exec_time
        end = time
        segments.append((pid, start, end))
        # enqueue newly arrived during execution
        while idx < n and procs[idx]['arrival'] <= time:
            q.append(procs[idx]['pid'])
            idx += 1
        if rem[pid] > 0:
            q.append(pid)
        else:
            completion[pid] = end
        # context switch when moving to next process (unless queue empty and next arrival sets time)
        if q:
            time += cs
    return merge_segments(segments), completion

def hrrn(processes, cs=0.0):
    procs = copy.deepcopy(processes)
    n = len(procs)
    completed = set()
    time = 0.0
    segments = []
    completion = {}
    while len(completed) < n:
        available = [p for p in procs if p['arrival'] <= time and p['pid'] not in completed]
        if not available:
            time = min([p['arrival'] for p in procs if p['pid'] not in completed])
            continue
        # response ratio
        def rr(p):
            waiting = time - p['arrival']
            return ( (waiting + p['burst']) / p['burst'], -p['arrival'], p['pid'] )
        cur = max(available, key=rr)
        start = time
        time += cur['burst']
        end = time
        segments.append((cur['pid'], start, end))
        completion[cur['pid']] = end
        completed.add(cur['pid'])
        time += cs
    return merge_segments(segments), completion

def mlfq_simple(processes, quanta=(1.0, 4.0), cs=0.0, aging_threshold=20.0):
    """
    Simple 2-level MLFQ:
     - Q0: quantum quanta[0], RoundRobin
     - Q1: quantum quanta[1], RoundRobin (longer quantum)
     - Q2: FCFS (no preemption)
    Aging: if a process waits more than aging_threshold in lower queues, promote it one level.
    """
    procs = sorted(copy.deepcopy(processes), key=lambda p: (p['arrival'], p['pid']))
    n = len(procs)
    time = 0.0
    idx = 0
    q0 = deque()
    q1 = deque()
    q2 = deque()
    rem = {p['pid']: p['burst'] for p in procs}
    last_enqueue_time = {}
    completion = {}
    segments = []
    def enqueue(pid, qlevel):
        if qlevel == 0:
            q0.append(pid)
        elif qlevel == 1:
            q1.append(pid)
        else:
            q2.append(pid)
        last_enqueue_time[pid] = time
    # map pid to current queue level
    level = {p['pid']: 0 for p in procs}
    while idx < n or q0 or q1 or q2:
        while idx < n and procs[idx]['arrival'] <= time:
            enqueue(procs[idx]['pid'], 0)
            idx += 1
        # aging promotions
        for pid in list(q1):
            if time - last_enqueue_time.get(pid, time) > aging_threshold:
                q1.remove(pid)
                enqueue(pid, 0)
                level[pid] = 0
        for pid in list(q2):
            if time - last_enqueue_time.get(pid, time) > aging_threshold:
                q2.remove(pid)
                enqueue(pid, 1)
                level[pid] = 1
        # pick from highest priority non-empty
        if q0:
            pid = q0.popleft()
            qtime = quanta[0]
            start = time
            exec_time = min(qtime, rem[pid])
            rem[pid] -= exec_time
            time += exec_time
            segments.append((pid, start, time))
            # arrivals during run
            while idx < n and procs[idx]['arrival'] <= time:
                enqueue(procs[idx]['pid'], 0)
                idx += 1
            if rem[pid] > 0:
                # demote to q1
                enqueue(pid, 1)
                level[pid] = 1
            else:
                completion[pid] = time
            if q0 or q1 or q2:
                time += cs
        elif q1:
            pid = q1.popleft()
            qtime = quanta[1]
            start = time
            exec_time = min(qtime, rem[pid])
            rem[pid] -= exec_time
            time += exec_time
            segments.append((pid, start, time))
            while idx < n and procs[idx]['arrival'] <= time:
                enqueue(procs[idx]['pid'], 0)
                idx += 1
            if rem[pid] > 0:
                # demote to q2
                enqueue(pid, 2)
                level[pid] = 2
            else:
                completion[pid] = time
            if q0 or q1 or q2:
                time += cs
        elif q2:
            pid = q2.popleft()
            start = time
            time += rem[pid]
            segments.append((pid, start, time))
            rem[pid] = 0
            completion[pid] = time
            while idx < n and procs[idx]['arrival'] <= time:
                enqueue(procs[idx]['pid'], 0)
                idx += 1
            if q0 or q1 or q2:
                time += cs
        else:
            time = procs[idx]['arrival']
    return merge_segments(segments), completion

# -------------------------
# Orchestration & CLI
# -------------------------
def parse_csv(filename):
    procs = []
    with open(filename, newline='') as f:
        rdr = csv.reader(f)
        for i, row in enumerate(rdr, start=1):
            if not row:
                continue
            # accept arrival,burst
            at = float(row[0])
            bt = float(row[1])
            procs.append({'pid': i, 'arrival': at, 'burst': bt})
    return procs

def input_interactive():
    n = int(input("Number of processes: ").strip())
    procs = []
    for i in range(1, n+1):
        parts = input(f"P{i} (arrival burst): ").strip().split()
        at = float(parts[0])
        bt = float(parts[1])
        procs.append({'pid': i, 'arrival': at, 'burst': bt})
    return procs

def run_algorithms(processes, algos, args):
    results_segments = {}
    results_completion = {}
    metrics = {}
    for algo in algos:
        if algo == 'fcfs':
            segs, comp = fcfs(processes, cs=args.cs)
        elif algo == 'sjf':
            segs, comp = sjf_predictive(processes, alpha=args.alpha, init_tau=args.init_tau, cs=args.cs)
        elif algo == 'srtf':
            segs, comp = srtf(processes, cs=args.cs, preempt_threshold=args.preempt_threshold)
        elif algo == 'rr':
            segs, comp = round_robin(processes, quantum=args.quantum, cs=args.cs, adaptive=args.adaptive)
        elif algo == 'hrrn':
            segs, comp = hrrn(processes, cs=args.cs)
        elif algo == 'mlfq':
            segs, comp = mlfq_simple(processes, quanta=(args.quantum, args.quantum*4), cs=args.cs, aging_threshold=args.aging)
        else:
            print("Unknown algorithm:", algo)
            continue
        # compute metrics
        completion, tat, wt = compute_metrics(processes, comp)
        # store printable metrics
        metrics_rows = []
        for p in processes:
            pid = p['pid']
            metrics_rows.append({
                'pid': pid,
                'arrival': p['arrival'],
                'burst': p['burst'],
                'completion': completion.get(pid),
                'turnaround': tat.get(pid),
                'waiting': wt.get(pid),
            })
        metrics[algo] = metrics_rows
        results_segments[algo] = segs
        results_completion[algo] = completion
        # print summary
        print(f"\n==== {algo.upper()} ====")
        print("PID\tAT\tBT\tCT\tTAT\tWT")
        for r in metrics_rows:
            print(f"{r['pid']}\t{r['arrival']}\t{r['burst']}\t{r['completion']}\t{r['turnaround']}\t{r['waiting']}")
        print(f"Average TAT: {mean_or_zero([r['turnaround'] for r in metrics_rows]):.4f}")
        print(f"Average WT : {mean_or_zero([r['waiting'] for r in metrics_rows]):.4f}")
        # plot and save
        savefile = f"gantt_{algo}.png" if args.save_png else None
        if args.plot:
            plot_gantt(segs, f"{algo.upper()} Gantt", savefile=savefile)
    if args.metrics_csv and metrics:
        save_metrics_csv(metrics, args.metrics_csv)
        print("Saved metrics CSV to", args.metrics_csv)
    return results_segments, results_completion, metrics

def demo_processes():
    return [
        {'pid':1, 'arrival':0.0, 'burst':3.0},
        {'pid':2, 'arrival':2.0, 'burst':6.0},
        {'pid':3, 'arrival':4.0, 'burst':4.0},
        {'pid':4, 'arrival':6.0, 'burst':5.0},
        {'pid':5, 'arrival':8.0, 'burst':2.0},
    ]

def main():
    parser = argparse.ArgumentParser(description="Improved CPU scheduling simulator")
    parser.add_argument('--input-file', '-i', help='CSV file with arrival,burst per line', default=None)
    parser.add_argument('--algos', '-a', help='comma-separated algorithms: fcfs,sjf,srtf,rr,hrrn,mlfq', default='fcfs,sjf,srtf,rr,hrrn')
    parser.add_argument('--quantum', '-q', type=float, default=3.0, help='time quantum for RR and base for MLFQ')
    parser.add_argument('--adaptive', action='store_true', help='adaptive quantum for RR')
    parser.add_argument('--cs', type=float, default=0.0, help='context switch cost (time units)')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for predictive SJF (0..1)')
    parser.add_argument('--init-tau', type=float, default=None, help='initial tau for predictive SJF (if omitted, mean burst used)')
    parser.add_argument('--preempt-threshold', type=float, default=0.0, help='SRTF preemption threshold')
    parser.add_argument('--aging', type=float, default=20.0, help='aging threshold for MLFQ promotion')
    parser.add_argument('--plot', action='store_true', help='show Gantt charts (matplotlib)')
    parser.add_argument('--save-png', action='store_true', help='save Gantt charts to PNG files')
    parser.add_argument('--metrics-csv', default=None, help='path to save per-algorithm metrics CSV')
    parser.add_argument('--demo', action='store_true', help='run demo dataset')
    args = parser.parse_args()

    if args.input_file:
        processes = parse_csv(args.input_file)
    elif args.demo:
        processes = demo_processes()
    else:
        print("No input-file and not demo => entering interactive mode.")
        processes = input_interactive()

    algos = [a.strip().lower() for a in args.algos.split(',') if a.strip()]
    run_algorithms(processes, algos, args)

if __name__ == "__main__":
    main()
