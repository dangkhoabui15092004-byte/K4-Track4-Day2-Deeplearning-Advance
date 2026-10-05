from __future__ import annotations
import argparse, copy, dataclasses, json, math, random, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent) if "__file__" in globals() else ".")
import dataset as D, model as M, losses as L
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 10
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    cache: bool = True
    images_dir: str = "/content/data"
    labels_dir: str = "/content/data"
    out_dir: str = "/content/drive/MyDrive/deepweeds/runs"
    pred_dir: str = "/content/drive/MyDrive/deepweeds/predictions"
    curves_dir: str = "/content/drive/MyDrive/deepweeds/curves"
    save_test_predictions: bool = False


def run_dir(cfg): return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"
def pred_path(cfg, split): return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    return torch.optim.AdamW(M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Cập nhật theo bước: warmup tuyến tính rồi cosine về ~0."""
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    def f(s):
        if s < warm: return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        ms, es = model.state_dict(), self.module.state_dict()
        for k, v in es.items():
            if v.dtype.is_floating_point: v.mul_(self.decay).add_(ms[k].detach(), alpha=1 - self.decay)
            else: v.copy_(ms[k])  # num_batches_tracked, BN buffer số nguyên


def build_crit(cfg, train_df, device):
    kw = {}
    if cfg.loss == "ls": kw["smoothing"] = cfg.label_smoothing or 0.1
    if cfg.loss == "focal": kw["gamma"] = cfg.focal_gamma
    if cfg.loss == "ce_weighted":
        cnt = np.bincount(train_df.Label.values, minlength=9)
        kw["weight"] = L.class_weights(cnt, cfg.class_weight_beta or 0.0).to(device)
    return L.build_criterion(cfg.loss, **kw).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    M.set_train_mode(model)
    tot, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            if cfg.mix:
                x, tg = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                loss = L.mixed_loss(criterion, model(x), tg)
            else:
                loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if ema: ema.update(model)
        tot += loss.item() * x.size(0); n += x.size(0)
    return {"train_loss": tot / n, "lr": optimizer.param_groups[-1]["lr"]}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp=True):
    model.eval()
    fns, ys, lgs, tot = [], [], [], 0.0
    for x, y, f in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg = model(x)
        lg = lg.float()
        tot += nn.functional.cross_entropy(lg, y, reduction="sum").item()
        fns += list(f); ys.append(y.cpu().numpy()); lgs.append(lg.cpu().numpy())
    ys, lgs = np.concatenate(ys), np.concatenate(lgs)
    return fns, ys, lgs, tot / len(ys)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def plot_curves(history, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = pd.DataFrame(history); ep = h["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, h.train_loss, label="train"); ax[0].plot(ep, h.val_loss, label="val")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(ep, h.val_macro_f1, label="val macro-F1"); ax[1].plot(ep, h.val_top1, label="val top-1")
    ax[1].set(title="Metric val", xlabel="epoch", ylabel="score"); ax[1].legend()
    ax[2].plot(ep, h.lr); ax[2].set(title="LR (cuối epoch)", xlabel="epoch", ylabel="lr")
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=120); plt.close(fig)


def run(cfg):
    set_seed(cfg.seed)
    rd = run_dir(cfg); rd.mkdir(parents=True, exist_ok=True)
    json.dump(dataclasses.asdict(cfg), open(rd / "config.json", "w"), indent=1)
    dev = torch.device("cuda")
    tr, va, te = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(tr, va, te, cfg.images_dir)
    tt, vt = D.build_transforms(True, cfg.img_size, cfg.aug), D.build_transforms(False, cfg.img_size)
    ltr = D.make_loader(tr, cfg.images_dir, tt, cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.seed, cfg.cache)
    lva = D.make_loader(va, cfg.images_dir, vt, 128, False, None, cfg.num_workers, cfg.seed, cfg.cache)
    model = M.build_model(cfg.backbone, True, 9, cfg.drop_rate, cfg.init).to(dev).to(memory_format=torch.channels_last)
    crit = build_crit(cfg, tr, dev)
    opt = build_optimizer(model, cfg); sch = build_scheduler(opt, cfg, len(ltr))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    ev_model = lambda: ema.module if ema else model

    hist, best, best_ep, t_ep = [], -1.0, -1, []
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, ltr, crit, opt, sch, scaler, cfg, dev, ema)
        torch.cuda.synchronize(); t_ep.append(time.time() - t0)
        fns, y, lg, vl = evaluate(ev_model(), lva, crit if False else nn.CrossEntropyLoss(), dev, cfg.amp)
        p = softmax(lg); m = compute_metrics(y, p.argmax(1), p)
        hist.append({"epoch": ep, **tl, "val_loss": vl, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "time": t_ep[-1]})
        print(f"[{cfg.exp_id} s{cfg.seed}] ep{ep} train_loss {tl['train_loss']:.4f} val_loss {vl:.4f} F1 {m['macro_f1']:.4f} acc {m['top1']:.4f} ({t_ep[-1]:.0f}s)")
        if m["macro_f1"] > best:  # hòa -> giữ epoch sớm hơn
            best, best_ep = m["macro_f1"], ep
            torch.save(ev_model().state_dict(), rd / "best.pt")
        pd.DataFrame(hist).to_csv(rd / "history.csv", index=False)

    final = copy.deepcopy(ev_model()); final.load_state_dict(torch.load(rd / "best.pt"))
    fns, y, lg, _ = evaluate(final, lva, nn.CrossEntropyLoss(), dev, cfg.amp)
    np.save(rd / "val_logits.npy", lg)
    save_predictions(pred_path(cfg, "val"), fns, y, softmax(lg))
    if cfg.save_test_predictions:  # CHỈ Bước 4
        lte = D.make_loader(te, cfg.images_dir, vt, 128, False, None, cfg.num_workers, cfg.seed, cfg.cache)
        fns, yt, lgt, _ = evaluate(final, lte, nn.CrossEntropyLoss(), dev, cfg.amp)
        np.save(rd / "test_logits.npy", lgt)
        save_predictions(pred_path(cfg, "test"), fns, yt, softmax(lgt))
    plot_curves(hist, Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{cfg.backbone}.png",
                f"{cfg.exp_id} {cfg.backbone} seed{cfg.seed}")
    res = {"exp_id": cfg.exp_id, "backbone": cfg.backbone, "seed": cfg.seed, "best_epoch": best_ep,
           "val_macro_f1": best, "sec_per_epoch": float(np.mean(t_ep)),
           "params_M": M.count_params(model), "weights_tag": getattr(model, "weights_tag", "")}
    json.dump(res, open(rd / "result.json", "w"), indent=1)
    return res
