# dynamic_brain_graph.py  (ST-GAT 修正维度版)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------- 17 通道坐标 ----------------
CH_NAMES = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
COORDS_17 = np.array(
[[-63., -91., -1.], [-42., -95., -1.], [-21., -95., -1.], [-7., -95., -1.],
 [  0., -95., -1.], [  7., -95., -1.], [ 21., -95., -1.], [ 42., -95., -1.],
 [ 63., -91., -1.], [-49., -85., -1.], [-28., -85., -1.], [  0., -85., -1.],
 [ 28., -85., -1.], [ 49., -85., -1.], [-21., -79., -1.], [  0., -79., -1.],
 [ 21., -79., -1.]], dtype=np.float32)


# ---------------- 多头图注意力层 ----------------
class GATLayer(nn.Module):
    def __init__(self, in_ch, out_ch, n_heads=4, dropout=0.2):
        super().__init__()
        assert out_ch % n_heads == 0
        self.n_heads = n_heads
        self.d_k = out_ch // n_heads
        self.W = nn.Linear(in_ch, out_ch, bias=False)
        self.a = nn.Linear(2 * self.d_k, 1, bias=False)
        self.dropout = dropout
        self.leaky = nn.LeakyReLU(0.2)
        self.reset_params()

    def reset_params(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.a.weight, gain=gain)

    def forward(self, x, adj):
        """
        x:   [B*T, N, in_ch]   N=17
        adj: [B*T, N, N]       0/1 mask
        return [B*T, N, out_ch]
        """
        B_T, N, _ = x.shape
        h = self.W(x).reshape(B_T, N, self.n_heads, self.d_k)   # [B*T,N,H,d_k]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)          # [B*T,N,N,H,d_k]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)          # [B*T,N,N,H,d_k]
        a_input = torch.cat([h_i, h_j], dim=-1)                 # [B*T,N,N,H,2*d_k]
        e = self.leaky(self.a(a_input)).squeeze(-1)             # [B*T,N,N,H]
        mask = (adj == 0).float().unsqueeze(-1) * -1e9          # [B*T,N,N,1]
        alpha = F.softmax(e + mask, dim=2)                      # [B*T,N,N,H]
        alpha = F.dropout(alpha, self.dropout, self.training)
        out = torch.einsum('btnh,bnhd->bnhd', alpha, h)         # [B*T,N,H,d_k]
        out = out.reshape(B_T, N, -1)                           # [B*T,N,out_ch]
        return out + x                                          # 残差


# ---------------- 时间 TCN（DepthWise） ----------------
class DepthTCN(nn.Module):
    def __init__(self, channels, k=3, dilation=1, dropout=0.2):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, k, dilation=dilation,
                              groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B*17, channels, T]
        """
        x = F.pad(x, (self.pad, 0))
        return self.drop(self.act(self.bn(self.conv(x))))


# ---------------- ST-Conv Block ----------------
class STConvBlock(nn.Module):
    def __init__(self, n_node=17, in_ch=64, hid_ch=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATLayer(in_ch, hid_ch, n_heads, dropout)
        self.tcn1 = DepthTCN(hid_ch, k=3, dilation=1, dropout=dropout)
        self.tcn2 = DepthTCN(hid_ch, k=3, dilation=2, dropout=dropout)
        self.res = nn.Conv1d(in_ch, hid_ch, 1) if in_ch != hid_ch else nn.Identity()

    def forward(self, x, adj):
        """
        x:   [B, N, T, in_ch]   N=17
        adj: [B, N, N]
        return [B, N, T, hid_ch]
        """
        B, N, T, C = x.shape

        # 1. 空间 GAT：需要 [B*T, N, C]
        h = x.permute(0, 2, 1, 3).reshape(B * T, N, C)        # [B*T,17,64]
        adj_bt = adj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        h = self.gat(h, adj_bt)                               # [B*T,17,64]

        # 2. 时间 TCN：需要 [B*N, C, T]
        h = h.reshape(B, T, N, -1).permute(0, 2, 1, 3)        # [B,17,64,T]
        h = h.reshape(B * N, -1, T)                           # [B*17,64,T]
        h = self.tcn2(self.tcn1(h))                           # [B*17,64,T]

        # 3. 残差：统一 [B,N,T,hid]
        h = h.reshape(B, N, -1, T).permute(0, 1, 3, 2)        # [B,17,T,64]
        res = self.res(x.permute(0, 3, 1, 2).reshape(B * N, C, T)).reshape(B, N, C, T).permute(0, 1, 3, 2)
        return h + res


# ---------------- ST-GAT 主模块 ----------------
class SemanticDynamicBrainGraph17(nn.Module):
    def __init__(self, in_ch=17, d_graph=64, k=6, sigma=50, alpha=0.6,
                 n_heads=4, dropout=0.2, n_blocks=2):
        super().__init__()
        self.k, self.sigma, self.alpha = k, sigma, alpha
        self.n_blocks = n_blocks

        # 1. 语义投影 2048→17
        self.sem_proj = nn.Linear(2048, 17, bias=False)
        self.register_buffer('coords', torch.from_numpy(COORDS_17))

        # 2. 17→64 初始嵌入
        self.emb = nn.Conv1d(17, d_graph, 1)

        # 3. ST-Conv blocks
        self.st_blocks = nn.ModuleList()
        for i in range(n_blocks):
            ic = d_graph if i == 0 else d_graph
            self.st_blocks.append(STConvBlock(17, ic, d_graph, n_heads, dropout))

        # 4. 全局池化
        self.attn_pool = nn.Sequential(
            nn.Linear(d_graph, d_graph // 2),
            nn.Tanh(),
            nn.Linear(d_graph // 2, 1)
        )

    def forward(self, x, z_vis2048):
        """
        x:        [B, 17, T]
        z_vis2048:[B, 2048]
        return:   z_graph [B,d_graph], A [B,17,17]
        """
        B, C, T = x.shape
        assert C == 17

        # ---------- 1. 语义-功能双通道 A ----------
        w_sem = self.sem_proj(z_vis2048)                           # [B,17]
        S = torch.bmm(w_sem.unsqueeze(2), w_sem.unsqueeze(1))      # [B,17,17]
        S = S.clamp(0, 1)
        dist = torch.cdist(self.coords, self.coords, p=2).unsqueeze(0)  # [1,17,17]
        D = torch.exp(-dist.pow(2) / (2 * self.sigma ** 2))
        A = self.alpha * S + (1 - self.alpha) * D
        topk, indices = torch.topk(A, k=self.k, dim=-1)
        A.zero_().scatter_(-1, indices, topk)
        A = (A + A.transpose(1, 2)) / 2                            # [B,17,17]

        # ---------- 2. 17→64 嵌入 ----------
        z = self.emb(x)                                            # [B,64,T]
        z = z.permute(0, 2, 1).unsqueeze(1).expand(-1, 17, -1, -1)  # [B,17,T,64]

        # ---------- 3. 多层 ST-Conv ----------
        for st_block in self.st_blocks:
            z = st_block(z, A)                                    # [B,17,T,64]

        # ---------- 4. 全局池化 ----------
        z_node = z.mean(dim=1)                                    # [B,T,64]
        score = self.attn_pool(z_node)                            # [B,T,1]
        z_graph = torch.sum(score * z_node, dim=1)                # [B,64]

        return z_graph, A