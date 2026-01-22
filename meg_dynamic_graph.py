# base/meg_dynamic_graph.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# ---------------- MEG Dynamic Brain Graph ----------------
class SemanticDynamicBrainGraphMEG(nn.Module):
    def __init__(self, c_num=271, t_len=201, d_graph=256, k=12, alpha=0.6, dropout=0.2, n_tcn_layers=2):  # ← d_graph 从 64 → 256
        super().__init__()
        self.c_num = c_num
        self.k = k
        self.alpha = alpha
        self.sem_proj = nn.Linear(2048, c_num * 16, bias=False)  # ← 输出向量而非标量
        self.temporal_proj = nn.Linear(t_len, d_graph)  # 现在是 256
        self.node_attn = nn.Sequential(
            nn.Linear(d_graph, d_graph // 2),
            nn.Tanh(),
            nn.Linear(d_graph // 2, 1)
        )

    def forward(self, x, z_vis2048):
        B, C, T = x.shape
        assert C == self.c_num

        # ---------- 1. semantic adjacency (增强版) ----------
        w = self.sem_proj(z_vis2048).view(B, C, 16)  # [B,271,16]
        A = torch.bmm(w, w.transpose(1, 2))  # [B,271,271]
        A = torch.softmax(A / np.sqrt(16), dim=-1)  # scaled softmax

        # Top-k sparsification (保持不变)
        topk, idx = torch.topk(A, k=self.k, dim=-1)
        A_sparse = torch.zeros_like(A)
        A_sparse.scatter_(-1, idx, topk)
        A = (A_sparse + A_sparse.transpose(1, 2)) / 2

        # ---------- 2. temporal embedding ----------
        h = self.temporal_proj(x)  # [B,271,256]

        # ---------- 3. graph propagation ----------
        h = torch.bmm(A, h)  # [B,271,256]

        # ---------- 4. attention pooling ----------
        score = self.node_attn(h)  # [B,271,1]
        score = torch.softmax(score, dim=1)
        z_graph = torch.sum(score * h, dim=1)  # [B,256]

        return z_graph, A