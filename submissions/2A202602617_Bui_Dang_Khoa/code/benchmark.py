from __future__ import annotations
import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    t = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(np.mean(t)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        model = model.half(); x = x.half()
    sync = torch.cuda.synchronize if device == "cuda" else None

    def fn():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(dtype == "amp")):
            model(x)

    r = bench(fn, warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "dtype": dtype,
            "batch": batch_size, "img_size": img_size, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Đo thật K lượt forward liên tiếp (so với K * p50 của 1 lượt)."""
    single = latency_report(model, **kw)
    model = model.eval()
    x = torch.randn(kw["batch_size"], 3, kw["img_size"], kw["img_size"], device=kw.get("device", "cuda"))

    def fn():
        with torch.inference_mode():
            for _ in range(k_views):
                model(x)

    r = bench(fn, kw.get("warmup", 10), kw.get("iters", 100),
              torch.cuda.synchronize if kw.get("device", "cuda") == "cuda" else None)
    return {"k": k_views, **{k: r[k] for k in ("p50", "p95", "p99")}, "single_p50": single["p50"],
            "k_times_single_p50": k_views * single["p50"]}
