# dfl_har.py
# ------------------------------------------------------------
# Two-Stage Communication-Aware DFL (HAR dataset, PyTorch)
# - Stage 1 (Control): ping(RTT/Loss/BW) + Top-k grad signature exchange
# - Stage 2 (Model): only selected neighbors exchange 8-bit quantized model
# Policies: "proposed" | "full" | "random" | "hybrid"(proposed→random)
# Hybrid switching:
#   * mode="round": at t >= t_switch
#   * mode="plateau": if acc[t] - acc[t-win] < eps
# ------------------------------------------------------------

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx

# -----------------------
# 1) HAR loader
# -----------------------
def load_har(data_root: str):
    def _load_split(split):
        X = np.loadtxt(os.path.join(data_root, split, f"X_{split}.txt"))
        y = np.loadtxt(os.path.join(data_root, split, f"y_{split}.txt")).astype(int) - 1
        return X, y

    Xtr, ytr = _load_split("train")
    Xte, yte = _load_split("test")

    mean = Xtr.mean(axis=0, keepdims=True)
    std  = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)
    return (Xtr, ytr), (Xte, yte)

# -----------------------
# 2) Model
# -----------------------
class HARMLP(nn.Module):
    def __init__(self, in_dim=561, n_cls=6, hid=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, n_cls)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------
# 3) Dirichlet split (robust)
# -----------------------
def dirichlet_partition(X, y, n_nodes=20, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    K = int(y.max().item() + 1)
    cls_idx = [np.where(y.cpu().numpy() == k)[0] for k in range(K)]
    frac = rng.dirichlet([alpha]*n_nodes, size=K)  # (K, n_nodes)

    node_indices = [[] for _ in range(n_nodes)]
    for k in range(K):
        idx = cls_idx[k].copy()
        rng.shuffle(idx)
        cnts = np.floor(frac[k] * len(idx)).astype(int)
        rem  = len(idx) - cnts.sum()
        if rem > 0:
            add = rng.choice(len(cnts), size=rem, replace=False)
            for r in add: cnts[r] += 1
        start = 0
        for n in range(n_nodes):
            end = start + cnts[n]
            if end > start:
                node_indices[n].extend(idx[start:end].tolist())
            start = end

    # ensure no empty shard
    sizes = [len(v) for v in node_indices]
    while any(s == 0 for s in sizes):
        donor = int(np.argmax(sizes)); recv = int(np.argmin(sizes))
        if sizes[donor] <= 1: break
        node_indices[recv].append(node_indices[donor].pop())
        sizes = [len(v) for v in node_indices]

    data = []
    for n in range(n_nodes):
        idx = node_indices[n]
        Xn, yn = (X[:0], y[:0]) if len(idx)==0 else (X[idx], y[idx])
        data.append(TensorDataset(Xn, yn))
    return data

# -----------------------
# 4) Quantization helpers (8-bit)
# -----------------------
@torch.no_grad()
def quantize_model_state(state_dict):
    q_state, scales, total_bytes = {}, {}, 0
    for k, w in state_dict.items():
        wfp = w.detach().cpu().float()
        s = max(wfp.abs().max().item(), 1e-8) / 127.0
        q = torch.clamp((wfp / s).round(), -127, 127).to(torch.int8)
        q_state[k] = q; scales[k] = s
        total_bytes += q.numel() * 1 + 4  # int8 + scale
    return q_state, scales, total_bytes

@torch.no_grad()
def dequantize_to_model(model, q_state, scales):
    sd = model.state_dict()
    for k in sd.keys():
        sd[k] = (q_state[k].to(torch.float32) * scales[k]).to(sd[k].dtype)
    model.load_state_dict(sd)
    return model

# -----------------------
# 5) Top-k gradient signature
# -----------------------
def topk_grad_signature(model, k=1024):
    vec = []
    for p in model.parameters():
        if p.grad is not None:
            vec.append(p.grad.detach().flatten().cpu())
    if len(vec) == 0:  # no grads
        return np.array([], np.int32), np.array([], np.int8), 0
    g = torch.cat(vec)
    k = min(k, g.numel())
    if k <= 0:
        return np.array([], np.int32), np.array([], np.int8), 0
    _, idx = torch.topk(g.abs(), k)
    signs = torch.sign(g[idx]).to(torch.int8).numpy()
    idx = idx.to(torch.int32).numpy()
    bytes_est = k*4 + k*1
    return idx, signs, bytes_est

def cosine_sim_from_topk(idx_i, sign_i, idx_j, sign_j):
    if idx_i.size == 0 or idx_j.size == 0: return 0.0
    set_i, set_j = set(idx_i.tolist()), set(idx_j.tolist())
    inter = set_i.intersection(set_j)
    if not inter: return 0.0
    map_j = {int(idx_j[t]): int(sign_j[t]) for t in range(len(idx_j))}
    dot = 0
    for t in inter:
        pos = np.where(idx_i == t)[0]
        if len(pos)==0: continue
        si = int(sign_i[pos[0]]); sj = map_j.get(int(t), 0)
        dot += (1 if (si*sj)>0 else -1)
    denom = math.sqrt(len(set_i)*len(set_j) + 1e-12)
    return float(dot/denom)

# -----------------------
# 6) Node
# -----------------------
class DFLNode:
    def __init__(self, node_id, dataset, model_fn, lr=1e-2, device="cpu"):
        self.id = node_id
        self.dataset = dataset
        self.model = model_fn().to(device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.device = device
        self.resource_level = float(np.random.uniform(0.5, 1.5))  # R_i

    def local_step(self, batch_size=128, steps=1):
        if len(self.dataset) == 0: return
        loader = DataLoader(self.dataset, batch_size=min(batch_size, len(self.dataset)),
                            shuffle=True, drop_last=False)
        it = iter(loader)
        self.model.train()
        for _ in range(steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader); x, y = next(it)
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            self.opt.step()

    def eval_on(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)

# -----------------------
# 7) Link env & cost
# -----------------------
class LinkEnv:
    def __init__(self, G, seed=0):
        rng = np.random.default_rng(seed)
        self.RTT, self.LOSS, self.BW = {}, {}, {}
        for (i, j) in G.edges():
            rtt = rng.uniform(10, 100); loss = rng.uniform(0.0, 0.2); bw = rng.uniform(5, 50)
            self.RTT[(i, j)] = rtt; self.RTT[(j, i)] = rtt
            self.LOSS[(i, j)] = loss; self.LOSS[(j, i)] = loss
            self.BW[(i, j)]  = bw;   self.BW[(j, i)]  = bw

    def control_ping(self, i, j):
        ctrl_bytes = 128
        rtt  = self.RTT[(i, j)] + np.random.normal(0, 2.0)
        loss = min(max(self.LOSS[(i, j)] + np.random.normal(0, 0.01), 0.0), 1.0)
        bw   = max(self.BW[(i, j)] + np.random.normal(0, 2.0), 1.0)
        return float(rtt), float(loss), float(bw), ctrl_bytes

    def update_bw_after_model(self, i, j, payload_bytes):
        bw_mbps = self.BW[(i, j)]
        sec = payload_bytes / (bw_mbps * 125000.0)  # 1 Mbps = 125000 B/s
        obs_bw = payload_bytes / max(sec, 1e-6) / 125000.0
        new_bw = 0.8*self.BW[(i, j)] + 0.2*obs_bw
        self.BW[(i, j)] = new_bw; self.BW[(j, i)] = new_bw
        drift = np.random.uniform(0.98, 1.02)
        self.RTT[(i, j)] *= drift; self.RTT[(j, i)] = self.RTT[(i, j)]
        self.LOSS[(i, j)] = min(max(self.LOSS[(i, j)]*drift, 0.0), 1.0)
        self.LOSS[(j, i)] = self.LOSS[(i, j)]

def comm_cost(rtt_ms, loss, bw_mbps, lam=(1.0, 0.5, 0.2)):
    return float(lam[0]*rtt_ms + lam[1]*loss - lam[2]*bw_mbps)

def softmax_np(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.max() if x.size>0 else x
    e = np.exp(x); s = e/(e.sum()+1e-12) if e.size>0 else e
    return s

# -----------------------
# 8) Experiment (+ hybrid)
# -----------------------
def run_experiment(
    data_root, n_nodes=20, alpha_dir=0.5,
    topology="erdos", p_edge=0.2,
    T=200, local_steps=1, batch_size=128,
    alpha=0.5, beta=0.3, gamma=0.1, delta=0.1,
    rho=0.3,                   # top-ρ selection (fraction of neighbors)
    selection_mode="topk",     # "topk" | "threshold"
    tau_percentile=60,         # used if selection_mode=="threshold": keep S > percentile(τ)
    topk=1024, lr=1e-2, device="cpu", seed=0,
    policy="proposed",         # "proposed" | "full" | "random" | "hybrid"
    # --- hybrid options ---
    hybrid_mode="round",       # "round" | "plateau"
    t_switch=220,              # used if hybrid_mode=="round"
    plateau_win=20,            # used if hybrid_mode=="plateau"
    plateau_eps=1e-3           # used if hybrid_mode=="plateau"
):
    """
    Returns history dict. If policy=='hybrid', starts as proposed and switches to random.
    'switch_round' key records when the switch actually happened (or -1 if never).
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    (Xtr, ytr), (Xte, yte) = load_har(str(data_root))
    node_data = dirichlet_partition(Xtr, ytr, n_nodes=n_nodes, alpha=alpha_dir, seed=seed)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512, shuffle=False)

    model_fn = lambda: HARMLP()
    nodes = [DFLNode(i, node_data[i], model_fn, lr=lr, device=device) for i in range(n_nodes)]
    if topology == "erdos":
        G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=seed)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=seed+1)
    elif topology == "ring":
        G = nx.cycle_graph(n_nodes)
    else:
        raise ValueError("unknown topology")
    env = LinkEnv(G, seed=seed)

    freshness = {(i, j): 0 for i in range(n_nodes) for j in G.neighbors(i)}
    hist = {"round": [], "acc_mean": [], "bytes_ctrl": [], "bytes_model": [], "E_proxy": [],
            "switch_round": -1}

    # current policy for this run (hybrid이면 시작은 proposed)
    current_policy = "proposed" if policy == "hybrid" else policy

    acc_history = []  # for plateau detection

    for t in range(T):
        # ---------- local updates
        for nd in nodes:
            nd.local_step(batch_size=batch_size, steps=local_steps)

        # --- if hybrid & plateau mode: 체크 후 스위칭
        if policy == "hybrid" and current_policy == "proposed" and hybrid_mode == "plateau":
            if len(acc_history) >= plateau_win:
                if (acc_history[-1] - acc_history[-plateau_win]) < plateau_eps:
                    current_policy = "random"
                    hist["switch_round"] = t  # 스위치 기록

        # --- if hybrid & round mode: t >= t_switch 이면 스위칭
        if policy == "hybrid" and current_policy == "proposed" and hybrid_mode == "round":
            if t >= t_switch:
                current_policy = "random"
                hist["switch_round"] = t

        selected_neighbors = {}
        bytes_ctrl_total = 0
        E_proxy_round = 0.0
        S_cache, C_cache = {}, {}
        topk_idx, topk_sign, topk_size = {}, {}, {}

        # ---------- Stage-1 (Control) if needed
        if current_policy == "proposed":
            # (A) compute top-k signatures
            for i, nd in enumerate(nodes):
                if len(nd.dataset) == 0:
                    topk_idx[i], topk_sign[i], topk_size[i] = np.array([], np.int32), np.array([], np.int8), 0
                    continue
                bs = min(128, len(nd.dataset))
                loader = DataLoader(nd.dataset, batch_size=bs, shuffle=True, drop_last=False)
                x_s, y_s = next(iter(loader))
                x_s, y_s = x_s.to(nd.device), y_s.to(nd.device)
                nd.opt.zero_grad(); logits = nd.model(x_s)
                F.cross_entropy(logits, y_s).backward()
                idx, sgn, b = topk_grad_signature(nd.model, k=topk)
                topk_idx[i], topk_sign[i], topk_size[i] = idx, sgn, b

            # (B) neighbor scoring with undirected edge accounting
            seen_edges = set()
            for i in range(n_nodes):
                scores, neighs = [], []
                for j in G.neighbors(i):
                    rtt, lossr, bw, ctrl_b = env.control_ping(i, j)
                    edge = frozenset({i, j})
                    if edge not in seen_edges:
                        bytes_ctrl_total += ctrl_b
                        bytes_ctrl_total += topk_size.get(i, 0) + topk_size.get(j, 0)
                        seen_edges.add(edge)

                    C_ij = comm_cost(rtt, lossr, bw)
                    dt   = freshness[(i, j)]
                    sim  = cosine_sim_from_topk(
                        topk_idx.get(i, np.array([], np.int32)),
                        topk_sign.get(i, np.array([], np.int8)),
                        topk_idx.get(j, np.array([], np.int32)),
                        topk_sign.get(j, np.array([], np.int8)),
                    )
                    Rj   = nodes[j].resource_level
                    S_ij = alpha*sim - beta*C_ij + gamma*Rj - delta*dt

                    S_cache[(i, j)] = S_ij
                    C_cache[(i, j)] = C_ij
                    scores.append(S_ij); neighs.append(j)

                if len(scores) == 0:
                    selected_neighbors[i] = []
                else:
                    if selection_mode == "threshold":
                        tau = np.percentile(scores, tau_percentile)
                        sel = [neighs[u] for u, s in enumerate(scores) if s > tau]
                        if len(sel) == 0:
                            sel = [neighs[int(np.argmax(scores))]]
                    else:
                        k = max(1, int(math.ceil(len(scores) * rho)))
                        idx_sorted = np.argsort(scores)[::-1][:k]
                        sel = [neighs[u] for u in idx_sorted]
                    selected_neighbors[i] = sel

        elif current_policy == "full":
            for i in range(n_nodes):
                selected_neighbors[i] = list(G.neighbors(i))

        elif current_policy == "random":
            for i in range(n_nodes):
                neighs = list(G.neighbors(i))
                k = max(1, int(math.ceil(len(neighs) * rho))) if len(neighs) > 0 else 0
                selected_neighbors[i] = random.sample(neighs, k) if k > 0 else []
        else:
            raise ValueError("policy must be in {'proposed','full','random'} (hybrid switches to 'random')")

        # ---------- Stage 2: model exchange
        inbox_states = {i: [] for i in range(n_nodes)}
        bytes_model_total = 0

        for i in range(n_nodes):
            q_state_i, scales_i, bytes_i = quantize_model_state(nodes[i].model.state_dict())
            for j in selected_neighbors[i]:
                if np.random.rand() < env.LOSS[(i, j)]:
                    continue
                inbox_states[j].append((i, q_state_i, scales_i, bytes_i))
                bytes_model_total += bytes_i
                env.update_bw_after_model(i, j, bytes_i)

                if (i, j) in C_cache:
                    C_ij = C_cache[(i, j)]
                else:
                    rtt, lossr, bw, _ = env.control_ping(i, j)
                    C_ij = comm_cost(rtt, lossr, bw)
                E_proxy_round += C_ij * bytes_i

        # ---------- aggregation
        for i in range(n_nodes):
            if len(inbox_states[i]) == 0:
                for j in G.neighbors(i):
                    freshness[(i, j)] = freshness.get((i, j), 0) + 1
                continue

            S_list, models_j, src_ids = [], [], []
            for (j, q_state, scales, b) in inbox_states[i]:
                S_list.append(S_cache.get((i, j), 0.0))
                peer = HARMLP().to(nodes[i].device)
                peer = dequantize_to_model(peer, q_state, scales)
                models_j.append(peer); src_ids.append(j)

            W = softmax_np(S_list) if len(S_list) > 1 else np.array([1.0])

            eta_w = 0.2
            base = nodes[i].model.state_dict()
            with torch.no_grad():
                for k in base.keys():
                    ws = [peer.state_dict()[k].to(nodes[i].device) * float(wj)
                          for wj, peer in zip(W, models_j)]
                    avg_k = torch.stack(ws, dim=0).sum(dim=0) if len(ws) > 1 else ws[0]
                    base[k].mul_(1.0 - eta_w).add_(eta_w * avg_k)
            nodes[i].model.load_state_dict(base)

            recv_from = set(src_ids)
            for j in G.neighbors(i):
                freshness[(i, j)] = 0 if (j in recv_from) else (freshness.get((i, j), 0) + 1)

        # ---------- metrics
        accs = [nd.eval_on(test_loader) for nd in nodes]
        acc_mean = float(np.mean(accs))
        acc_history.append(acc_mean)

        hist["round"].append(t)
        hist["acc_mean"].append(acc_mean)
        hist["bytes_ctrl"].append(bytes_ctrl_total if current_policy == "proposed" else 0)
        hist["bytes_model"].append(bytes_model_total)
        hist["E_proxy"].append(E_proxy_round)

    return hist
