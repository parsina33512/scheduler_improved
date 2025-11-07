#!/usr/bin/env python3
"""
scheduler_final.py
Interactive, user-friendly CPU scheduling simulator for class/demo.

Features:
 - Interactive menu (choose single algorithm, run-all, or batch from CSV)
 - Clear prompts that explain what to enter
 - Supports: FCFS, SJF(predictive), SRTF, Round Robin, HRRN, MLFQ(simple)
 - Float times supported, context-switch cost, predictive alpha, SRTF threshold, RR adaptive
 - Plots Gantt (matplotlib), optional save PNG, optional metrics CSV
Usage: python scheduler_final.py
(then follow the on-screen menu)
"""
import copy
import math
import csv
import statistics
from collections import deque
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------
def mean_or_zero(lst):
    return sum(lst) / len(lst) if lst else 0.0

def merge_segments(segments):
    if not segments:
        return []
    merged = []
    eps = 1e-9
    for pid, s, e in segments:
        if merged and merged[-1][0] == pid and abs(merged[-1][2] - s) <= eps:
            merged[-1] = (merged[-1][0], merged[-1][1], e)
        else:
            merged.append((pid, s, e))
    return merged

def compute_metrics(processes, completion):
    arrival = {p['pid']: p['arrival'] for p in processes}
    burst = {p['pid']: p['burst'] for p in processes}
    tat = {pid: completion[pid] - arrival[pid] for pid in completion}
    wt = {pid: tat[pid] - burst[pid] for pid in tat}
    return completion, tat, wt

def save_metrics_csv(metrics_rows, filename):
    keys = ['algorithm', 'pid', 'arrival', 'burst', 'completion', 'turnaround', 'waiting']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in metrics_rows:
            writer.writerow(r)

def plot_gantt(segments, title="Gantt Chart", savefile=None, figsize=(12,3), dpi=200):
    if not segments:
        print(f"[plot] Nothing to plot for: {title}")
        return
    merged = merge_segments(segments)
    pids = sorted({s[0] for s in merged})
    y_positions = {pid: i for i, pid in enumerate(pids[::-1])}
    fig_height = max(2, 0.5 * len(pids))
    fig, ax = plt.subplots(figsize=figsize if figsize else (10, fig_height), dpi=dpi)
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
        plt.savefig(savefile, bbox_inches='tight', dpi=dpi)
        print(f"[plot] Saved: {savefile}")
    plt.show()

# ---------------------------
# Scheduling algorithm implementations
# ---------------------------
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
        time += cs
    return merge_segments(segments), completion

def sjf_predictive(processes, alpha=0.5, init_tau=None, cs=0.0):
    procs = copy.deepcopy(processes)
    n = len(procs)
    if init_tau is None:
        init_tau = mean_or_zero([p['burst'] for p in procs]) if procs else 1.0
    predicted = {p['pid']: init_tau for p in procs}
    completed = set()
    time = 0.0
    segments = []
    while len(completed) < n:
        available = [p for p in procs if p['arrival'] <= time and p['pid'] not in completed]
        if not available:
            time = min(p['arrival'] for p in procs if p['pid'] not in completed)
            continue
        cur = min(available, key=lambda p: (predicted[p['pid']], p['arrival'], p['pid']))
        start = time
        time += cur['burst']
        end = time
        segments.append((cur['pid'], start, end))
        completed.add(cur['pid'])
        predicted[cur['pid']] = alpha * cur['burst'] + (1 - alpha) * predicted[cur['pid']]
        time += cs
    completion = {p['pid']: max([e for pid,s,e in segments if pid==p['pid']]) for p in procs}
    return merge_segments(segments), completion

def srtf(processes, cs=0.0, preempt_threshold=0.0):
    procs = sorted(copy.deepcopy(processes), key=lambda p: (p['arrival'], p['pid']))
    n = len(procs)
    import heapq
    rem = {p['pid']: p['burst'] for p in procs}
    time = 0.0
    idx = 0
    heap = []
    segments = []
    current_pid = None
    start_time = None
    while idx < n or heap:
        while idx < n and procs[idx]['arrival'] <= time:
            p = procs[idx]
            heapq.heappush(heap, (rem[p['pid']], p['arrival'], p['pid'], p['burst']))
            idx += 1
        if not heap:
            time = procs[idx]['arrival']
            continue
        rem_now, arr_t, pid, burst = heapq.heappop(heap)
        if current_pid != pid:
            if current_pid is not None:
                segments.append((current_pid, start_time, time))
                time += cs
            current_pid = pid
            start_time = time
        # decide next event time
        next_arrival = procs[idx]['arrival'] if idx < n else math.inf
        finish_time = time + rem[pid]
        if next_arrival < finish_time:
            exec_time = next_arrival - time
            rem[pid] -= exec_time
            time = next_arrival
            heapq.heappush(heap, (rem[pid], arr_t, pid, burst))
        else:
            time = finish_time
            rem[pid] = 0.0
            segments.append((pid, start_time, time))
            current_pid = None
            start_time = None
            time += cs
    completion = {}
    for p in procs:
        ends = [e for (pp,s,e) in segments if pp == p['pid']]
        completion[p['pid']] = max(ends) if ends else None
    return merge_segments(segments), completion

def round_robin(processes, quantum=1.0, cs=0.0, adaptive=False):
    procs = sorted(copy.deepcopy(processes), key=lambda p: (p['arrival'], p['pid']))
    n = len(procs)
    time = 0.0
    idx = 0
    q = deque()
    rem = {p['pid']: p['burst'] for p in procs}
    completion = {}
    segments = []
    base_q = float(quantum)
    while idx < n or q:
        while idx < n and procs[idx]['arrival'] <= time:
            q.append(procs[idx]['pid'])
            idx += 1
        if not q:
            time = procs[idx]['arrival']
            continue
        pid = q.popleft()
        if adaptive:
            seen = [p['burst'] for p in procs if p['arrival'] <= time]
            qtime = max(0.01, statistics.median(seen)) if seen else base_q
        else:
            qtime = base_q
        start = time
        exec_time = min(qtime, rem[pid])
        rem[pid] -= exec_time
        time += exec_time
        end = time
        segments.append((pid, start, end))
        while idx < n and procs[idx]['arrival'] <= time:
            q.append(procs[idx]['pid'])
            idx += 1
        if rem[pid] > 0:
            q.append(pid)
        else:
            completion[pid] = end
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
        def rr(p):
            waiting = time - p['arrival']
            return ((waiting + p['burst']) / p['burst'], -p['arrival'], p['pid'])
        cur = max(available, key=rr)
        start = time
        time += cur['burst']
        end = time
        segments.append((cur['pid'], start, end))
        completion[cur['pid']] = end
        completed.add(cur['pid'])
        time += cs
    return merge_segments(segments), completion

def mlfq_simple(processes, quanta=(1.0,4.0), cs=0.0, aging_threshold=20.0):
    procs = sorted(copy.deepcopy(processes), key=lambda p: (p['arrival'], p['pid']))
    n = len(procs)
    time = 0.0
    idx = 0
    q0 = deque(); q1 = deque(); q2 = deque()
    rem = {p['pid']: p['burst'] for p in procs}
    last_enqueue_time = {}
    completion = {}
    segments = []
    def enqueue(pid, qlevel):
        if qlevel == 0: q0.append(pid)
        elif qlevel == 1: q1.append(pid)
        else: q2.append(pid)
        last_enqueue_time[pid] = time
    level = {p['pid']: 0 for p in procs}
    while idx < n or q0 or q1 or q2:
        while idx < n and procs[idx]['arrival'] <= time:
            enqueue(procs[idx]['pid'], 0); idx += 1
        for pid in list(q1):
            if time - last_enqueue_time.get(pid, time) > aging_threshold:
                q1.remove(pid); enqueue(pid,0); level[pid]=0
        for pid in list(q2):
            if time - last_enqueue_time.get(pid, time) > aging_threshold:
                q2.remove(pid); enqueue(pid,1); level[pid]=1
        if q0:
            pid = q0.popleft(); qtime = quanta[0]
            start = time; exec_time = min(qtime, rem[pid]); rem[pid]-=exec_time; time+=exec_time
            segments.append((pid,start,time))
            while idx<n and procs[idx]['arrival']<=time: enqueue(procs[idx]['pid'],0); idx+=1
            if rem[pid]>0: enqueue(pid,1); level[pid]=1
            else: completion[pid]=time
            if q0 or q1 or q2: time+=cs
        elif q1:
            pid = q1.popleft(); qtime = quanta[1]
            start = time; exec_time = min(qtime, rem[pid]); rem[pid]-=exec_time; time+=exec_time
            segments.append((pid,start,time))
            while idx<n and procs[idx]['arrival']<=time: enqueue(procs[idx]['pid'],0); idx+=1
            if rem[pid]>0: enqueue(pid,2); level[pid]=2
            else: completion[pid]=time
            if q0 or q1 or q2: time+=cs
        elif q2:
            pid = q2.popleft()
            start = time; time += rem[pid]; segments.append((pid,start,time)); rem[pid]=0; completion[pid]=time
            while idx<n and procs[idx]['arrival']<=time: enqueue(procs[idx]['pid'],0); idx+=1
            if q0 or q1 or q2: time+=cs
        else:
            time = procs[idx]['arrival']
    return merge_segments(segments), completion

# ---------------------------
# Input helpers and interactive menu
# ---------------------------
def read_csv(fname):
    procs=[]
    with open(fname, newline='') as f:
        rdr = csv.reader(f)
        for i,row in enumerate(rdr, start=1):
            if not row: continue
            at=float(row[0]); bt=float(row[1])
            procs.append({'pid':i,'arrival':at,'burst':bt})
    return procs

def interactive_input():
    while True:
        try:
            n = int(input("Enter number of processes (e.g. 4): ").strip())
            if n<=0:
                print("Number must be positive.")
                continue
            break
        except Exception:
            print("Please enter a valid integer.")
    procs=[]
    for i in range(1,n+1):
        while True:
            raw = input(f"P{i} — enter Arrival and Burst (e.g. 0 3): ").strip().split()
            if len(raw) < 2:
                print("Two numbers required; try again.")
                continue
            try:
                at = float(raw[0]); bt = float(raw[1])
                procs.append({'pid':i,'arrival':at,'burst':bt})
                break
            except Exception:
                print("Enter valid numbers (e.g. 0.0 3.5).")
    return procs

def show_menu():
    print("\n=== CPU Scheduling Simulator ===")
    print("1) Run a single algorithm")
    print("2) Run all algorithms (benchmark)")
    print("3) Read from CSV file (each line: arrival,burst)")
    print("4) Exit")
    choice = input("Choose (1/2/3/4): ").strip()
    return choice

def choose_algo_prompt():
    print("\nWhich algorithm?")
    print("1) FCFS (First-Come First-Served)")
    print("2) SJF (predictive)")
    print("3) SRTF (preemptive SJF)")
    print("4) Round Robin")
    print("5) HRRN")
    print("6) MLFQ (simple)")
    print("7) Return to menu")
    return input("Choose (1..7): ").strip()

def run_single_algo(procs, algo_key, params):
    name_map = {'1':'fcfs','2':'sjf','3':'srtf','4':'rr','5':'hrrn','6':'mlfq'}
    algo = name_map.get(algo_key)
    if not algo:
        return None
    print(f"\n[info] Running: {algo.upper()} ...")
    if algo == 'fcfs':
        segs, comp = fcfs(procs, cs=params['cs'])
    elif algo == 'sjf':
        segs, comp = sjf_predictive(procs, alpha=params['alpha'], init_tau=params['init_tau'], cs=params['cs'])
    elif algo == 'srtf':
        segs, comp = srtf(procs, cs=params['cs'], preempt_threshold=params['preempt_threshold'])
    elif algo == 'rr':
        segs, comp = round_robin(procs, quantum=params['quantum'], cs=params['cs'], adaptive=params['adaptive'])
    elif algo == 'hrrn':
        segs, comp = hrrn(procs, cs=params['cs'])
    elif algo == 'mlfq':
        segs, comp = mlfq_simple(procs, quanta=(params['quantum'], params['quantum']*4), cs=params['cs'], aging_threshold=params['aging'])
    else:
        return None
    completion, tat, wt = compute_metrics(procs, comp)
    rows=[]
    print("\nPID\tAT\tBT\tCT\tTAT\tWT")
    for p in procs:
        pid=p['pid']
        row={'algorithm': algo, 'pid': pid, 'arrival': p['arrival'], 'burst': p['burst'],
             'completion': completion.get(pid), 'turnaround': tat.get(pid), 'waiting': wt.get(pid)}
        rows.append(row)
        print(f"{pid}\t{p['arrival']}\t{p['burst']}\t{row['completion']}\t{row['turnaround']}\t{row['waiting']}")
    print(f"\nAverage TAT: {mean_or_zero([r['turnaround'] for r in rows]):.4f}")
    print(f"Average WT : {mean_or_zero([r['waiting'] for r in rows]):.4f}")
    if params['plot']:
        plot_gantt(segs, title=f"{algo.upper()} Gantt")
    if params['save_png']:
        plot_gantt(segs, title=f"{algo.upper()} Gantt", savefile=f"gantt_{algo}.png")
    if params['metrics_csv']:
        save_metrics_csv(rows, params['metrics_csv'])
        print(f"[io] metrics saved to {params['metrics_csv']}")
    return rows

# ---------------------------
# Main interactive loop
# ---------------------------
def interactive_run():
    print("Starting interactive mode. If you want to read processes from a CSV file, choose that option in the menu.")
    params = {
        'quantum': 3.0, 'cs': 0.0, 'alpha': 0.5, 'init_tau': None,
        'preempt_threshold': 0.0, 'adaptive': False, 'aging':20.0,
        'plot': True, 'save_png': False, 'metrics_csv': None
    }
    while True:
        choice = show_menu()
        if choice == '4':
            print("Exit. Good luck.")
            break
        if choice == '3':
            fname = input("Enter path to CSV file (each line: arrival,burst): ").strip()
            try:
                procs = read_csv(fname)
            except Exception as e:
                print("Error reading file:", e)
                continue
        elif choice in ('1','2'):
            # get processes from interactive input
            procs = interactive_input()
        else:
            print("Invalid choice, try again.")
            continue

        # gather global params
        try:
            cs_in = input(f"Context switch cost (time units) (default {params['cs']}): ").strip()
            if cs_in != '':
                params['cs'] = float(cs_in)
        except:
            print("Invalid value; using default.")

        try:
            q_in = input(f"Base time quantum for RR/MLFQ (default {params['quantum']}): ").strip()
            if q_in != '':
                params['quantum'] = float(q_in)
        except:
            pass

        alpha_in = input(f"Alpha for predictive-SJF (0..1) (default {params['alpha']}): ").strip()
        if alpha_in != '':
            try: params['alpha'] = float(alpha_in)
            except: pass

        th_in = input(f"SRTF preempt threshold (default {params['preempt_threshold']}): ").strip()
        if th_in != '':
            try: params['preempt_threshold'] = float(th_in)
            except: pass

        adapt_in = input(f"Use adaptive RR? (y/n) (default n): ").strip().lower()
        params['adaptive'] = (adapt_in == 'y')

        plot_in = input(f"Show Gantt charts? (y/n) (default y): ").strip().lower()
        params['plot'] = (plot_in != 'n')
        savepng_in = input(f"Save Gantt charts to PNG? (y/n) (default n): ").strip().lower()
        params['save_png'] = (savepng_in == 'y')
        metrics_in = input(f"Save metrics to CSV? Enter filepath or press Enter to skip: ").strip()
        params['metrics_csv'] = metrics_in if metrics_in != '' else None

        if choice == '2':
            # run all algorithms
            all_algos = ['1','2','3','4','5','6']
            all_rows = []
            for ak in all_algos:
                rows = run_single_algo(procs, ak, params)
                if rows:
                    all_rows.extend(rows)
            if params['metrics_csv']:
                # append algorithm names are already in rows
                save_metrics_csv(all_rows, params['metrics_csv'])
                print(f"[io] All metrics saved to {params['metrics_csv']}")
            continue

        # single algorithm execution
        while True:
            ak = choose_algo_prompt()
            if ak == '7':
                break
            if ak not in ('1','2','3','4','5','6'):
                print("Enter a valid number.")
                continue
            # for RR ask quantum if user wants override
            if ak == '4':
                q_ask = input(f"Do you want to override quantum? (current {params['quantum']}) y/n: ").strip().lower()
                if q_ask == 'y':
                    try:
                        params['quantum'] = float(input("Enter quantum value: ").strip())
                    except:
                        print("Invalid value; using previous quantum.")
            rows = run_single_algo(procs, ak, params)
            # after running, ask if run another algorithm on same input
            again = input("Run another algorithm with the same input? (y/n): ").strip().lower()
            if again != 'y':
                break

def main():
    print("scheduler_final — interactive CPU scheduling simulator\n")
    interactive_run()

if __name__ == "__main__":
    main()
