from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predict_logits(model, loader, device, view=None, amp=True):
    """Logit theo đúng thứ tự file của loader. view: hàm biến đổi batch, hoặc None."""
    model.eval()
    fns, ys, lgs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device)
            if view is not None:
                x = view(x)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                lg = model(x)
            fns += list(f)
            ys.append(y.numpy())
            lgs.append(lg.float().cpu().numpy())
    return fns, np.concatenate(ys), np.concatenate(lgs)


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) và bản lật của chúng (10 view)."""
    H, W = x.shape[2], x.shape[3]
    pos = [(0, 0), (0, W - crop), (H - crop, 0), (H - crop, W - crop),
           ((H - crop) // 2, (W - crop) // 2)]
    crops = [x[:, :, i:i + crop, j:j + crop] for i, j in pos]
    return crops + [torch.flip(c, dims=[3]) for c in crops]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước. Chỉ dùng cho CNN/ConvNeXt (ViT/Swin cố định kích thước)."""
    return [F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(space)


def ensemble_probs(list_of_probs):
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Cực tiểu NLL trên VAL theo T (tìm có biên, scipy)."""
    from scipy.optimize import minimize_scalar
    y = np.asarray(val_labels)

    def nll(T):
        p = _softmax(val_logits / T)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    return float(minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x)


def apply_temperature(logits, T: float):
    return _softmax(logits / T)


def fuse_conv_bn(model):
    """Gộp BN vào Conv liền trước (chỉ áp dụng cho Sequential/ResNet-style).
    ConvNeXt (LayerNorm), ViT, Swin không có BatchNorm: hàm trả về model không đổi."""
    model = copy.deepcopy(model).eval()

    def fuse(conv, bn):
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros(w.size(0), device=w.device)
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                        conv.padding, conv.dilation, conv.groups, bias=True).to(w.device)
        new.weight.data = w * s.view(-1, 1, 1, 1)
        new.bias.data = bn.bias + s * (b - bn.running_mean)
        return new

    def walk(m):
        names = list(m._modules.keys())
        for i, n in enumerate(names):
            child = m._modules[n]
            walk(child)
            if isinstance(child, nn.BatchNorm2d) and i > 0 and isinstance(m._modules[names[i - 1]], nn.Conv2d):
                m._modules[names[i - 1]] = fuse(m._modules[names[i - 1]], child)
                m._modules[n] = nn.Identity()

    walk(model)
    return model
