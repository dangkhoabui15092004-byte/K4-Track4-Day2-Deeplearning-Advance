from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """q'(k) = (1-eps)*1[k==y] + eps/K. Tự cài đặt; eps=0 cho đúng CE."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1 - self.eps) * nll + self.eps * smooth).mean()


class FocalLoss(nn.Module):
    """FL = -alpha_t * (1-p_t)^gamma * log(p_t). gamma=0 (alpha=None) cho đúng CE."""
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Chỉ truyền số ảnh của TRAIN. beta=0: 1/n chuẩn hoá trung bình 1.
    beta>0: class-balanced (1-beta)/(1-beta^n), chuẩn hoá tổng = số lớp."""
    n = np.asarray(counts, dtype=np.float64)
    if beta == 0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)


def build_criterion(kind: str = "ce", **kw):
    """kind: ce | ls | focal | ce_weighted.
    kw: smoothing, gamma, alpha, weight (tensor)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss không hợp lệ: {kind}")


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        H, W = x.shape[2], x.shape[3]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
        x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - float((y2 - y1) * (x2 - x1)) / (H * W)  # diện tích thực sau khi cắt biên
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
