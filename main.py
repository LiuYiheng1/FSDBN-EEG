import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

##import user lib

from base.data_meg import load_meg_data
from base.utils import update_config, ClipLoss, instantiate_from_config, get_device

import open_clip
import torch.nn as nn
from torchvision.models import resnet18

device = get_device('auto')
import os, torch
os.environ["USE_LIBUV"] = "0"
import torch.nn.functional as F
def load_model(config, train_loader, test_loader):
    model = {}
    # 1. 原脑编码器
    model['brain'] = instantiate_from_config(config['models']['brain'])
    # 2. 双流
    # 加载 CLIP 模型
    clip_model, _ = open_clip.create_model_from_pretrained('RN50', pretrained='openai')

    # 初始化双流编码器
    model['what'] = WhatEncoder(clip_model, output_dim=config['z_dim'])
    model['where'] = WhereEncoder(output_dim=config['z_dim'])

    pl_model = TriStreamPLModel(model, config, train_loader, test_loader)
    return pl_model
class WhatEncoder(nn.Module):
    """前景→物体语义，用 CLIP-RN50 的 conv 层提取 mask 区域特征"""
    def __init__(self, clip_rn50, output_dim=1024):
        super().__init__()
        self.backbone = clip_rn50.visual
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # [B,2048,7,7]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, output_dim)

    def forward(self, img):  # ← 去掉 fg_mask
        x = self.backbone(img)  # [B,2048,7,7]
        x = self.pool(x).flatten(1)  # [B,2048]
        return self.proj(x)  # [B,output_dim]


class WhereEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        net = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # 统一命名
        self.proj = nn.Linear(512, output_dim)

    def forward(self, img):
        x = self.backbone(img)
        x = x.flatten(1)
        return self.proj(x)


class GateFusion(nn.Module):
    """
    不确定性门控融合
    u ∈ [0,1]   越不确定 → 越信任 What（前景语义）
    """
    def __init__(self, dim):
        super().__init__()
        # 先拼接 [what; where] → 降维 → sigmoid 输出门控 g
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        # 不确定性修正：g = g * u + (1 - u) * 0.5
        # 0.5 表示中性融合，u 越大越偏向 What
        self.register_buffer('neutral', torch.tensor(0.5))

    def forward(self, z_what, z_where, u):
        """
        z_what, z_where: [B, dim]
        u: [B] 或 [B,1]  0-1 不确定性分数
        return: [B, dim]
        """
        if u.dim() == 1:
            u = u.unsqueeze(-1)                       # [B,1]
        concat = torch.cat([z_what, z_where], dim=1)  # [B,2*dim]
        g = self.gate(concat)                         # [B,dim]  0-1
        g = g * u + self.neutral * (1 - u)            # 不确定性修正
        return g * z_what + (1 - g) * z_where


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader, model_type='RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3

        self.mAP_total = 0
        self.match_similarities = []

    def forward(self, batch, sample_posterior=False):

        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']
        img = batch['img']

        img_z = batch['img_features']

        eeg_z = self.brain(eeg)
        img_z = img_z / img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)

        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label

            loss = total_loss
        else:
            loss = total_loss
        return eeg_z, img_z, loss

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch, sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        k = min(5, similarity.size(-1))
        top_kvalues, top_k_indices = similarity.topk(k, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            counter = Counter(self.match_label)
            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]

        eeg_z, img_z, loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        k = min(5, similarity.size(-1))
        top_kvalues, top_k_indices = similarity.topk(k, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        k = min(5, similarity.size(-1))
        top_kvalues, top_k_indices = similarity.topk(k, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        # label =  batch['label']
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        # compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap

        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),
                'test_top5_acc': top_k_accuracy.item(), 'mAP': self.mAP, 'similarity': self.match_similarities}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=500
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


class TriStreamPLModel(PLModel):
    """
    双流视觉(What/Where) + 脑信号对比学习
    修复了:
    1. logit_scale 未约束导致对比损失失效
    2. 循环依赖的 uncertainty 更新逻辑
    3. caption 损失权重过高问题
    4. u 为 NumPy 数组导致的 AttributeError
    5. 模型缺少 cap_feats 属性
    6. 张量梯度问题导致无法转 numpy
    """

    def __init__(self, model_dict, config, train_loader, test_loader):
        super().__init__(model_dict, config, train_loader, test_loader)
        self.what = model_dict['what']
        self.where = model_dict['where']
        self.fusion = GateFusion(config['z_dim'])
        self.proj_cap = nn.Linear(self.z_dim, 384)

        # 修复: 加载 caption 特征到模型中（用于调试和辅助方法）
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        cap_path = os.path.join(project_root, 'weights', 'meg', 'caption_sbert384.pt')
        self.cap_feats = torch.load(cap_path, map_location='cpu')

    def forward(self, batch, sample_posterior=False):
        """返回 (z_eeg, z_vis, loss)，不操作 dataloader"""
        idx = batch['idx'].cpu().numpy()
        eeg = batch['eeg']  # [B, C, T]
        fg_m = batch['sam_mask_fg'].to(self.device)  # [B,224,224]
        bg_m = batch['sam_mask_bg'].to(self.device)

        # ---- 双流视觉特征 ----
        z_what = self.what(batch['fg_tensor'].to(self.device))
        z_where = self.where(batch['bg_tensor'].to(self.device))

        # ① 额外拿 2048-d 向量供脑图分支
        with torch.no_grad():
            # WhatEncoder 里 backbone 输出 [B,2048,7,7]
            fg_feat = self.what.backbone(batch['fg_tensor'].to(self.device))
            z_vis2048 = F.adaptive_avg_pool2d(fg_feat, 1).flatten(1)  # [B,2048]

        # 不确定性分数 → 移除循环依赖，使用上一轮缓存的 match_label
        batch_size = batch['idx'].shape[0]

        # --- 修复: 验证阶段或首次训练时，u 默认为中性值 ---
        if self.config['data']['uncertainty_aware'] and hasattr(self, 'match_label'):
            # 从内部缓存读取，而非 batch（避免循环依赖）
            batch_labels = self.match_label[idx]  # NumPy array
            # 修复: 转换为 PyTorch tensor
            u = torch.from_numpy(batch_labels).float().to(self.device)  # [B]
            u = (u + 1) / 4.0  # 0.25/0.5/0.75
        else:
            u = torch.ones(batch_size, device=self.device) * 0.5  # 中性融合

        # 修复: 确保 u 是 tensor 后再调用 .dim()
        if u.dim() == 1:
            u = u.unsqueeze(-1)  # [B,1]

        z_vis = self.fusion(z_what, z_where, u.to(self.device))

        # ---- 脑特征 ----
        z_eeg, A, logit_scale = self.brain(eeg, z_vis2048)  # ✅ 接收三个

        # ---- 对比损失 ----
        z_vis = z_vis / z_vis.norm(dim=-1, keepdim=True)

        eeg_loss, img_loss, logits = self.criterion(z_eeg, z_vis, logit_scale)

        eeg_loss, img_loss, logits = self.criterion(z_eeg, z_vis, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        # ---------- caption 对齐 ----------
        cap_vec = batch['cap_feat'].to(self.device)  # [B, 384]
        z_eeg_cap = self.proj_cap(z_eeg)  # [B, 384]
        z_eeg_cap = F.normalize(z_eeg_cap, dim=-1)
        cap_vec = F.normalize(cap_vec, dim=-1)
        loss_cap = F.mse_loss(z_eeg_cap, cap_vec)

        # 修复: 动态权重，前10 epoch 降低 caption 影响
        current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        # 将 λ 从 0.3 降到 0.05
        λ = 0.01 if current_epoch < 10 else 0.05  # ← 关键修改
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2 + λ * loss_cap


        # >>>>>>>  新增图 loss 开始 <<<<<<<<
        if A is not None:  # DBN 已启用
            B, C, _ = A.shape  # 实时取 B 和 C

            # 1. 图对比 InfoNCE
            if self.config['data'].get('use_graph_contrast', True):
                z_g1 = A.mean(dim=1)  # (B, C)
                z_g2 = A.roll(50, dims=-1).mean(dim=1)
                z_g1 = F.normalize(z_g1, dim=1)
                z_g2 = F.normalize(z_g2, dim=1)
                logits_g = torch.einsum('bd,bd->b', z_g1, z_g2) / 0.1  # (B,)
                neg = torch.einsum('bd,nd->bn', z_g1, z_g2)  # (B, B)
                labels_g = torch.zeros(B, dtype=torch.long, device=self.device)
                loss_gcon = F.cross_entropy(torch.cat([logits_g.unsqueeze(1), neg], dim=1), labels_g)
                total_loss = total_loss + 0.2 * loss_gcon

            # 2. 拓扑正则：稀疏 + 平滑
            if self.config['data'].get('use_graph_reg', True):
                A_mean = A.mean(dim=0)  # (C, C)
                sparsity = A_mean.sum() / (C * C)
                smooth = torch.norm(A_mean[:, 1:] - A_mean[:, :-1], p=2)
                loss_reg = 1e-4 * sparsity + 1e-4 * smooth
                total_loss = total_loss + loss_reg
        # >>>>>>>  新增图 loss 结束 <<<<<<<<

        # ---- uncertainty 感知分支（仅计算，不写回 dataset）----
        if self.config['data']['uncertainty_aware']:
            # 修复: 使用 detach() 避免梯度回传
            diagonal_elements = torch.diagonal(logits.detach()).cpu().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label = np.ones_like(batch_sim, dtype=int)
            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            # 仅更新内部缓存，不碰 dataloader（修复循环依赖）
            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label
            self.last_eeg = z_eeg.detach()

        return z_eeg, z_vis, total_loss

    def training_step(self, batch, batch_idx):
        # ========== 调试：验证数据对齐 ==========
        if batch_idx == 0:
            idx = batch['idx'].cpu().numpy()
            img_paths = [self.trainer.train_dataloader.dataset.data['img'][i] for i in idx[:3]]
            print(f"\n[DEBUG] Batch 0 first 3 samples:")
            for i, path in enumerate(img_paths):
                print(f"  Sample {i}: idx={idx[i]}, img_path={path}")
                # 验证 caption 存在
                cap = self._get_caption_feature(path)
                print(f"    Caption feature found: {cap is not None}, norm={cap.norm() if cap is not None else 'N/A'}")
        # =======================================

        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch, sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = eeg_z @ img_z.T
        k = min(5, similarity.size(-1))
        top_kvalues, top_k_indices = similarity.topk(k, dim=-1)

        # 修复: 先 detach 再转 numpy
        self.all_predicted_classes.append(top_k_indices.detach().cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        # --- 修复: 在 epoch 最后 batch 时计算指标（不移除循环依赖的更新）---
        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(all_true_labels)

            self.log('train_top1_acc', top_1_accuracy, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.all_predicted_classes = []
            self.all_true_labels = []

            # 记录 match_label 分布（但不更新 dataset，修复循环依赖）
            counter = Counter(self.match_label)
            key_mapping = {0: 'low_blur', 1: 'medium_blur', 2: 'high_blur'}
            count_dict_mapped = {}
            for k, v in counter.items():
                name = key_mapping.get(k, f"class_{k}")
                count_dict_mapped[f"match_{name}"] = v
            self.log_dict(count_dict_mapped, on_epoch=True, logger=True, sync_dist=True)

            # 移除循环依赖的关键：不再写回 dataloader
            # self.trainer.train_dataloader.dataset.match_label = self.match_label.copy()

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]

        eeg_z, img_z, loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        k = min(5, similarity.size(-1))
        top_kvalues, top_k_indices = similarity.topk(k, dim=-1)
        # 修复: 添加 detach
        self.all_predicted_classes.append(top_k_indices.detach().cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(all_true_labels)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(all_true_labels)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        k = min(5, similarity.size(-1))
        top_kvalues, top_k_indices = similarity.topk(k, dim=-1)
        # 修复: 添加 detach
        self.all_predicted_classes.append(top_k_indices.detach().cpu().numpy())
        # label =  batch['label']
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        # compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap

        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(all_true_labels)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(all_true_labels)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),
                'test_top5_acc': top_k_accuracy.item(), 'mAP': self.mAP, 'similarity': self.match_similarities}

    def _get_caption_feature(self, img_path):
        """鲁棒的 caption 特征获取方法"""
        cap_feat = None

        # 尝试多种可能的键名变体
        key_variants = [
            img_path,  # 原始路径 (可能包含反斜杠)
            img_path.replace("\\", "/"),  # Windows -> Linux 路径分隔符
            os.path.basename(img_path),  # 只用文件名
            os.path.join(os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path)),  # 一级目录+文件名
        ]

        # 如果 img_path 是相对路径且包含子目录，也尝试这种形式
        if '/' in img_path or '\\' in img_path:
            key_variants.insert(0, img_path)  # 优先尝试原始形式

        for key in key_variants:
            if key in self.cap_feats:
                cap_feat = self.cap_feats[key]
                break

        return cap_feat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/meg/ubp.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,

        default="meg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-01',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        default='MEGProjectLayer',
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default='RN50',
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'], exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name=config['name'],
                               version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}")
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))

    train_loader, val_loader, test_loader = load_meg_data(config) if config['dataset'] == 'eeg' else load_meg_data(
        config)

    print(
        f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='max'
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='min'
        )

    trainer = Trainer(log_every_n_steps=10, strategy='auto',
                      callbacks=[early_stop_callback, checkpoint_callback], max_epochs=config['train']['epoch'],
                      devices=[device], accelerator='cuda', logger=logger)
    print(trainer.logger.log_dir)

    ckpt_path = 'last'  # None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()