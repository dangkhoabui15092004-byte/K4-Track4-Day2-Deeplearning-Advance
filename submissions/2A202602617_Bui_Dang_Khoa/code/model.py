from __future__ import annotations
import timm, torch, torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50", "resnext50": "resnext50_32x4d", "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224", "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0", "mobilenetv3": "mobilenetv3_large_100",
}


def _head_params(model):
    return list(model.get_classifier().parameters())


def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune"):
    name = SUGGESTED_BACKBONES.get(name, name)
    use_pre = init != "scratch" and pretrained
    model = timm.create_model(name, pretrained=use_pre, num_classes=num_classes, drop_rate=drop_rate)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    model.weights_tag = (cfg.get("hf_hub_id") or cfg.get("url") or "none") if use_pre else "scratch"
    model.arch_name = name
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model):
    head = {id(p) for p in _head_params(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.init_mode = "frozen"


def set_train_mode(model):
    """Gọi thay cho model.train() trong train loop: nếu frozen thì backbone ở eval (BN không đổi)."""
    model.train()
    if getattr(model, "init_mode", "") == "frozen":
        head = {id(p) for p in _head_params(model)}
        for m in model.modules():
            if not any(id(p) in head for p in m.parameters(recurse=False)) and \
               not list(m.children()):
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        # BN nằm trong head (nếu có) vẫn train; backbone BN đã eval ở trên


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = {id(p) for p in _head_params(model)}
    bb_decay, bb_nodecay, hd = [], [], []
    for p in model.parameters():
        if not p.requires_grad: continue
        if id(p) in head: hd.append(p)
        elif p.ndim <= 1: bb_nodecay.append(p)
        else: bb_decay.append(p)
    groups = [{"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay},
              {"params": bb_nodecay, "lr": lr_backbone, "weight_decay": 0.0},
              {"params": hd, "lr": lr_head, "weight_decay": weight_decay}]
    return [g for g in groups if g["params"]]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    # pip install fvcore ; fvcore đếm 1 MAC = 1 "flop" nên chính là GMAC
    from fvcore.nn import FlopCountAnalysis
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        fa = FlopCountAnalysis(model, torch.randn(1, 3, img_size, img_size, device=dev))
        fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
        g = fa.total() / 1e9
    model.train(was_training)
    return g
