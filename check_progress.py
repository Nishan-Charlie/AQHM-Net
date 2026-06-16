"""
check_progress.py — print current experiment progress.
Run anytime while experiments are executing:
    python check_progress.py
"""
import json, os, glob
from datetime import datetime

RESULTS = os.path.join(os.path.dirname(__file__), "results")
DS_ORDER = ["mnist_0123", "mnist", "pneumoniamnist", "breastmnist", "dermamnist", "pathmnist"]
DS_NAMES = {
    "mnist_0123":    "MNIST-4",
    "mnist":         "MNIST-10",
    "pneumoniamnist":"PneumoniaMNIST",
    "breastmnist":   "BreastMNIST",
    "dermamnist":    "DermaMNIST",
    "pathmnist":     "PathMNIST",
}

print(f"\nAQHM-Net progress  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
print("=" * 78)
print(f"{'Dataset':<20}  {'Seeds done':>11}  {'Best W-F1':>10}  {'Best W-Acc':>11}  {'Status'}")
print("-" * 78)

for tag in DS_ORDER:
    dname = DS_NAMES[tag]
    ds_dir = os.path.join(RESULTS, tag)
    if not os.path.isdir(ds_dir):
        print(f"  {dname:<18}  {'pending':>11}  {'—':>10}  {'—':>11}")
        continue

    # Count completed seeds from HISTORIES (not summary, which is only written at end)
    hist_dir = os.path.join(ds_dir, "histories")
    hfiles   = sorted(glob.glob(os.path.join(hist_dir, "*.json"))) if os.path.isdir(hist_dir) else []

    # Count in-progress seeds from CHECKPOINTS that have no matching history yet
    ckpt_dir  = os.path.join(ds_dir, "checkpoints")
    ckpt_pts  = glob.glob(os.path.join(ckpt_dir, "*.pt")) if os.path.isdir(ckpt_dir) else []
    ckpt_hist = glob.glob(os.path.join(ckpt_dir, "*.json")) if os.path.isdir(ckpt_dir) else []

    n_hist_done = len(hfiles)
    n_in_prog   = len(ckpt_pts) - n_hist_done   # seeds with checkpoint but no history yet

    # Best epoch from latest history file
    latest_ep = "—"
    if hfiles:
        with open(hfiles[-1]) as f:
            h = json.load(f)
        stopped = h.get("stopped_epoch", len(h.get("train_loss", [])))
        latest_ep = f"ep {stopped}"

    # Current in-progress epoch from checkpoint history file
    elif ckpt_hist:
        with open(sorted(ckpt_hist)[-1]) as f:
            h = json.load(f)
        ep = len(h.get("train_loss", []))
        latest_ep = f"ep {ep} (live)"

    # Final summary if available
    s_path = os.path.join(ds_dir, "summary.json")
    if os.path.isfile(s_path):
        with open(s_path) as f:
            s = json.load(f)
        n_runs = s.get("n_runs", "?")
        if s.get("n_runs", 0) > 1:  # real paper run summary, not 1-seed test
            wf1  = s.get("weighted_f1",      {}).get("mean", 0)
            wacc = s.get("weighted_accuracy", {}).get("mean", 0)
            status = "DONE" if n_hist_done >= n_runs else f"{n_hist_done}/{n_runs} seeds"
            print(f"  {dname:<18}  {n_hist_done:>4}/{n_runs:<6}  {wf1:>10.4f}  {wacc:>11.4f}  {status}")
            continue

    # No final summary yet — show live progress
    in_prog_str = f" (+{n_in_prog} live)" if n_in_prog > 0 else ""
    status = f"{n_hist_done} done{in_prog_str}  ({latest_ep})"
    print(f"  {dname:<18}  {n_hist_done:>11}  {'—':>10}  {'—':>11}  {status}")

print("=" * 78)

# Show tail of experiment log
log = os.path.join(RESULTS, "experiment_log.txt")
if os.path.isfile(log):
    lines = [l.rstrip() for l in open(log).readlines() if l.strip()]
    print(f"\nLast log entries ({os.path.basename(log)}):")
    for l in lines[-5:]:
        print(" ", l)
print()
