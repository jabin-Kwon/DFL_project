# run_dfl.py
# ------------------------------------------------------------
# Runner for Two-Stage Communication-Aware DFL on UCI HAR
# Plots accuracy, bytes (control+model), and energy proxy.
# Policies: Full / Random / Proposed / Hybrid
# ------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dfl_har_hybird_V2 import run_experiment

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=False,
                   help="Path to unzipped 'UCI HAR Dataset' directory")
    p.add_argument("--selection_mode", type=str, default="threshold",
                   choices=["topk", "threshold"])
    p.add_argument("--tau", type=int, default=70,
                   help="percentile τ (threshold mode)")
    p.add_argument("--rho", type=float, default=0.3,
                   help="top-ρ selection ratio")
    p.add_argument("--rounds", type=int, default=300)
    # hybrid 관련 옵션(필요하면 조정)
    p.add_argument("--t_switch", type=int, default=220)
    p.add_argument("--mix_rounds", type=int, default=20)
    p.add_argument("--ramp_rounds", type=int, default=20)
    return p.parse_args()

def plot_histories(hists, labels, title_suffix=""):
    plt.figure()
    for h, lb in zip(hists, labels):
        plt.plot(h["round"], h["acc_mean"], label=f"{lb}")
    plt.xlabel("Round"); plt.ylabel("Mean Test Accuracy")
    plt.title(f"Accuracy vs Round {title_suffix}")
    plt.legend(); plt.tight_layout()

    plt.figure()
    for h, lb in zip(hists, labels):
        total_bytes = (np.array(h["bytes_ctrl"]) + np.array(h["bytes_model"])) / (1024*1024)
        plt.plot(h["round"], total_bytes, label=f"{lb}")
    plt.xlabel("Round"); plt.ylabel("Bytes per Round (MB)")
    plt.title(f"Communication Bytes per Round {title_suffix}")
    plt.legend(); plt.tight_layout()

    plt.figure()
    for h, lb in zip(hists, labels):
        plt.plot(h["round"], np.array(h["E_proxy"]) / (1024*1024), label=f"{lb}")
    plt.xlabel("Round"); plt.ylabel("Energy Proxy (Σ C×Bytes, MB-scaled)")
    plt.title(f"Energy Proxy per Round {title_suffix}")
    plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    DEFAULT_DATA_ROOT = Path(r"C:\Users\kckwu\source\data\UCI HAR Dataset")
    DATA_ROOT = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT

    xtrain = DATA_ROOT / "train" / "X_train.txt"
    xtest  = DATA_ROOT / "test"  / "X_test.txt"
    if not xtrain.exists() or not xtest.exists():
        raise FileNotFoundError(
            f"[DATA_ROOT 점검] '{DATA_ROOT}' 아래에 'train/X_train.txt' 또는 "
            f"'test/X_test.txt'가 없습니다.\n"
            f"현재 확인된 경로:\n  - {xtrain}\n  - {xtest}"
        )

    common = dict(
        data_root=str(DATA_ROOT),
        n_nodes=20, alpha_dir=0.5,
        topology="erdos", p_edge=0.2,
        T=args.rounds, local_steps=1, batch_size=128,
        alpha=0.5, beta=0.3, gamma=0.1, delta=0.1,
        rho=args.rho,
        selection_mode=args.selection_mode, tau_percentile=args.tau,
        topk=1024, lr=0.01, device="cpu", seed=0,
        t_switch=args.t_switch, mix_rounds=args.mix_rounds,
        ramp_rounds=args.ramp_rounds,
        eta_min=0.05, eta_max=0.20
    )

    # Full / Random / Proposed / Hybrid 네 가지 모두 실행
    hist_full    = run_experiment(**common, policy="full")
    hist_rand    = run_experiment(**common, policy="random")
    hist_prop    = run_experiment(**common, policy="proposed")
    hist_hybrid  = run_experiment(**common, policy="hybrid")

    plot_histories(
        [hist_full, hist_rand, hist_prop, hist_hybrid],
        labels=["Full", "Random", "Proposed", "Hybrid(P→R, η-ramp+mix)"],
        title_suffix="(HAR / 20 nodes)"
    )
