"""基线（协议 v2）：persistence / 逐日气候均值 / 异常持续 / pooled HistGradientBoosting。

划分与评估只用 src/data.py 与 src/eval.py，本脚本不重复实现它们。

用法：
  python scripts/run_baselines.py --subset 20 --skip-test   # 20 座水库试运行（不碰 main 测试集）
  python scripts/run_baselines.py --time-full-hgb           # 仅 main、h=10 全量 HGB 训练+验证选迭代耗时
  python scripts/run_baselines.py --final                   # 全量正式运行（评估 main 测试集，只应运行一次）
必须且只能指定 --final 或 --skip-test 之一，否则拒绝运行（--time-full-hgb 永远不评估测试集）。
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import data, eval as ev

METHODS = ("persistence", "climatology", "anomaly_persistence", "pooled_hgb")
SPLITS = tuple(data.SPLIT_YEARS)                     # main, D1, D2；main 必须最先跑（选迭代次数）
STRIDE = 6                                           # 训练起点每 6 天取一个
VAL_SELECT_STRIDE = 3                                # 选迭代次数时验证样本每 3 个取 1 个
CLIM_HALF_WIDTH = 7                                  # 逐日气候均值 ±7 天平滑
CLIP = (0.0, 100.0)                                  # 所有方法的预测在生成时统一截断
HGB_PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, random_state=0, early_stopping=False)

LAGS = (1, 3, 7, 10, 30)
RAIN_WINDOWS = (3, 7, 14, 30, 60)
TEMP_WINDOWS = (7, 30)
HUM_WINDOWS = (7,)
MAX_WINDOW = max(LAGS + RAIN_WINDOWS + TEMP_WINDOWS + HUM_WINDOWS)
assert MAX_WINDOW <= data.MAX_LOOKBACK, "特征窗口超过 MAX_LOOKBACK"

BASE_NAMES = (["rv_t"] + [f"d_lag{k}" for k in LAGS] + [f"rain_{k}" for k in RAIN_WINDOWS]
              + [f"tmp_{k}" for k in TEMP_WINDOWS] + [f"hum_{k}" for k in HUM_WINDOWS])
STATIC_NAMES = ["log_basin", "log_yoohyoy", "log_agr", "agr_ratio", "agr_zero"]
FEATURE_NAMES = BASE_NAMES + ["clim_t", "clim_th", "anom_t", "sin_t", "cos_t", "sin_th", "cos_th"] + STATIC_NAMES

FORMAL_COLUMNS = ["split", "segment", "h", "method", "group", "threshold", "n_samples", "n_reservoirs",
                  "nse_median", "nse_q25", "nse_q75", "rmse", "events_seg", "events_seg_hit",
                  "alerts_seg", "alerts_seg_overlap", "precision_seg", "recall_seg"]
RAW_COLUMNS = ["events_raw", "alerts_raw", "tp_raw", "precision_raw", "recall_raw", "n_eligible"]  # 只进日志


# ----------------------------------------------------------------------------- 计时
class Timer:
    def __init__(self):
        self.rows = []

    @contextlib.contextmanager
    def step(self, name, split=None, h=None, n_sub=None, n_full=None):
        t0 = time.perf_counter()
        yield
        sec = time.perf_counter() - t0
        self.rows.append(dict(step=name, split=split, h=h, seconds=sec, n_sub=n_sub, n_full=n_full))
        print(f"  [{sec:8.1f}s] {name} {split or ''} {'' if h is None else 'h=%d' % h} "
              f"{'' if n_sub is None else 'n=%d' % n_sub}", flush=True)


TIMER = Timer()


# ----------------------------------------------------------------------------- 数据与特征
def load_covariates(ds):
    """气象（rain/tempt/humid，(T,N)）与静态属性（基面积、有效库容、灌溉面积，(N,3)），对齐 ds。"""
    cols = ["fac_code", "date", "rain", "tempt", "humid", "basin_area", "yoohyoy", "agr_area"]
    df = pd.read_parquet(data.PARQUET, columns=cols)
    df = df[df.fac_code.isin(ds.fac_codes)]

    def wide(c):
        w = df.pivot(index="date", columns="fac_code", values=c)
        return w.reindex(index=ds.dates, columns=ds.fac_codes).to_numpy(np.float32)

    cov = dict(rain=wide("rain"), tempt=wide("tempt"), humid=wide("humid"))
    for k, a in cov.items():
        assert np.isfinite(a).all(), f"{k} 含缺失，滚动和会被污染"
    static_cols = ["basin_area", "yoohyoy", "agr_area"]
    g = df.groupby("fac_code")[static_cols]
    assert (g.nunique(dropna=False).to_numpy() == 1).all(), "静态属性在时间上不恒定"
    cov["static"] = g.first().reindex(ds.fac_codes).to_numpy(np.float32)
    return cov


def _rolling_sum(a, k):
    """(T,N) -> 窗口 [t-k+1, t] 的和；t < k-1 处为 NaN。"""
    cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0, dtype=np.float64)])
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[k - 1:] = (cs[k:] - cs[:-k])[: a.shape[0] - k + 1]
    return out


def _lag(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def build_base_features(rvow, rain, tempt, humid):
    """(T,N) 输入 -> {特征名: (T,N) float32}。每个特征在第 t 行只用第 t 行及之前的数据。"""
    f = {"rv_t": rvow.astype(np.float32)}
    for k in LAGS:
        f[f"d_lag{k}"] = rvow - _lag(rvow, k)
    for k in RAIN_WINDOWS:
        f[f"rain_{k}"] = _rolling_sum(rain, k)
    for k in TEMP_WINDOWS:
        f[f"tmp_{k}"] = _rolling_sum(tempt, k) / k
    for k in HUM_WINDOWS:
        f[f"hum_{k}"] = _rolling_sum(humid, k) / k
    assert list(f) == BASE_NAMES
    return f


def static_features(static_raw):
    """(N,3)[基面积, 有效库容, 灌溉面积] -> (N,5)：三者的 log1p、灌溉面积比例、灌溉面积为零的标记。"""
    basin, yoo, agr = static_raw[:, 0], static_raw[:, 1], static_raw[:, 2]
    ratio = np.where(basin > 0, agr / np.maximum(basin, 1e-6), 0.0)
    return np.column_stack([np.log1p(basin), np.log1p(yoo), np.log1p(agr), ratio,
                            (agr == 0).astype(np.float32)]).astype(np.float32)


def climatology(rvow, years, doy, train_years, half_width=CLIM_HALF_WIDTH):
    """(366,N) 逐日气候均值：只用训练年份的观测，按日序 ±half_width 天循环平滑；
    平滑窗口内训练期仍无观测的（水库，日序）用全体水库该日序的均值填充。"""
    N = rvow.shape[1]
    in_train = np.isin(years, train_years)
    obs = ~np.isnan(rvow)
    vals = np.where(obs, rvow, 0.0)
    S, C = np.zeros((366, N)), np.zeros((366, N))
    for d in range(1, 367):
        m = in_train & (doy == d)
        if m.any():
            S[d - 1], C[d - 1] = vals[m].sum(0), obs[m].sum(0)

    def circ(A):
        return sum(np.roll(A, s, axis=0) for s in range(-half_width, half_width + 1))

    Ssm, Csm = circ(S), circ(C)
    tab = np.where(Csm > 0, Ssm / np.maximum(Csm, 1), np.nan)
    pooled = np.nanmean(tab, axis=1, keepdims=True)
    return np.where(np.isnan(tab), pooled, tab).astype(np.float32)


def make_context(ds, cov, stations):
    """把评估要用的数组打包；stations 为全量数据下的站编号（只用于分解表，不进特征）。"""
    return dict(ds=ds, stations=stations, doy=ds.dates.dayofyear.to_numpy(), years=ds.years,
                base=build_base_features(ds.rvow, cov["rain"], cov["tempt"], cov["humid"]),
                statics=static_features(cov["static"]), clim={}, splits={})


def _get_split(ctx, name, h):
    if (name, h) not in ctx["splits"]:
        ctx["splits"][(name, h)] = data.make_split(ctx["ds"], name, h)
    return ctx["splits"][(name, h)]


def _get_clim(ctx, name):
    if name not in ctx["clim"]:
        ctx["clim"][name] = climatology(ctx["ds"].rvow, ctx["years"], ctx["doy"], data.SPLIT_YEARS[name]["train"])
    return ctx["clim"][name]


def make_X(ctx, name, h, t, res):
    """(n, 26) float32 特征矩阵。特征只用起点 t 及之前的观测；clim_th 与 sin/cos 的目标日项由日历决定。"""
    tab, doy = _get_clim(ctx, name), ctx["doy"]
    cl_t, cl_th = tab[doy[t] - 1, res], tab[doy[t + h] - 1, res]
    w = 2 * np.pi / 365.25
    cols = [ctx["base"][n][t, res] for n in BASE_NAMES]
    cols += [cl_t, cl_th, ctx["base"]["rv_t"][t, res] - cl_t,
             np.sin(w * doy[t]), np.cos(w * doy[t]), np.sin(w * doy[t + h]), np.cos(w * doy[t + h])]
    cols += [ctx["statics"][res, j] for j in range(len(STATIC_NAMES))]
    X = np.column_stack(cols).astype(np.float32)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X


# ----------------------------------------------------------------------------- 训练与预测
def _select_iterations(gb, X, y0, y):
    best_it, best = 1, np.inf
    for it, pv in enumerate(gb.staged_predict(X), 1):
        mse = np.mean((np.clip(y0 + pv, *CLIP) - y) ** 2)
        if mse < best:
            best, best_it = mse, it
    return best_it


def _hgb_increment(gb, X, it):
    if gb.n_iter_ == it:
        return gb.predict(X)
    for i, pv in enumerate(gb.staged_predict(X), 1):
        if i == it:
            return pv
    raise ValueError(f"迭代次数 {it} 超过已训练的 {gb.n_iter_}")


def fit_unit(ctx, name, h, hgb_params=HGB_PARAMS, best_it=None):
    """在划分 name 的训练集（stride 取样）上拟合 rho 与 HGB。

    best_it=None（仅 main）：用 400 次迭代训练，在验证集上选迭代次数；
    best_it 给定（D1/D2）：直接以该迭代次数训练，不再调参。返回模型字典。
    """
    ds = ctx["ds"]
    mask = _get_split(ctx, name, h).masks["train"] & (np.arange(len(ds.dates)) % STRIDE == 0)[:, None]
    t, res, y0, y = ev.samples_from_mask(ds.rvow, mask, h)
    with TIMER.step("build_train_X", name, h, n_sub=len(t)):
        X = make_X(ctx, name, h, t, res)
    tab = _get_clim(ctx, name)
    a0, a1 = y0 - tab[ctx["doy"][t] - 1, res], y - tab[ctx["doy"][t + h] - 1, res]
    rho = float((a0 * a1).sum() / (a0 * a0).sum())
    params = dict(hgb_params)
    if best_it is not None:
        params["max_iter"] = best_it
    gb = HistGradientBoostingRegressor(**params)
    with TIMER.step("fit_hgb", name, h, n_sub=len(t)):
        gb.fit(X, y - y0)
    model = dict(gb=gb, rho=rho, n_train=len(t), n_iter_trained=int(gb.n_iter_))
    if best_it is None:
        vt, vr, vy0, vy = ev.samples_from_mask(ds.rvow, _get_split(ctx, name, h).masks["val"], h)
        sub = np.arange(0, len(vt), VAL_SELECT_STRIDE)
        with TIMER.step("select_iter", name, h, n_sub=len(sub)):
            Xv = make_X(ctx, name, h, vt[sub], vr[sub])
            model["best_it"] = _select_iterations(gb, Xv, vy0[sub], vy[sub])
        model["n_val_sub"] = len(sub)
    else:
        model["best_it"] = best_it
    return model


def raw_predictions(ctx, model, name, h, t, res, y0):
    """四种方法的未截断预测（float64）。"""
    tab, doy = _get_clim(ctx, name), ctx["doy"]
    cl_t, cl_th = tab[doy[t] - 1, res], tab[doy[t + h] - 1, res]
    incr = _hgb_increment(model["gb"], make_X(ctx, name, h, t, res), model["best_it"])
    return {"persistence": y0.astype(np.float64),
            "climatology": cl_th.astype(np.float64),
            "anomaly_persistence": cl_th + model["rho"] * (y0 - cl_t),
            "pooled_hgb": y0 + incr}


def predict_segment(ctx, model, name, h, segment):
    """某段样本的 (t, res, y0, y, 预测字典)。预测在此统一截断到 [0,100] 并存为 float32。"""
    ds = ctx["ds"]
    t, res, y0, y = ev.samples_from_mask(ds.rvow, _get_split(ctx, name, h).masks[segment], h)
    preds = {m: np.clip(p, *CLIP).astype(np.float32) for m, p in raw_predictions(ctx, model, name, h, t, res, y0).items()}
    return t, res, y0, y, preds


# ----------------------------------------------------------------------------- 评估与输出
def evaluate_segment(ctx, name, segment, h, t, res, y0, y, preds):
    """-> (汇总表, 分解表, 原始事件计数表)。评估全部经 src/eval.py。"""
    ds = ctx["ds"]
    cold = data.cold_start_flags(ds, name)
    groups = {"all": np.ones(len(cold), dtype=bool), "existing": ~cold, "cold_start": cold}
    years_t, stations_t = ds.years[t], ctx["stations"][res]
    rows, brk = [], []
    for method in METHODS:
        p = preds[method]
        df = ev.summary_frame(ev.evaluate(y, p, y0, res, t, cold))
        for i, (col, val) in enumerate([("split", name), ("segment", segment), ("h", h), ("method", method)]):
            df.insert(i, col, val)
        rows.append(df)
        for g, gm in groups.items():
            s = gm[res]
            if not s.any():
                continue                                    # 空组（D1/D2 的冷启动）不出分解表
            b = ev.breakdown_table(y[s], p[s], res[s], len(cold), years_t[s], stations_t[s])
            for i, (col, val) in enumerate([("split", name), ("segment", segment), ("h", h),
                                            ("method", method), ("group", g)]):
                b.insert(i, col, val)
            brk.append(b)
    rows = pd.concat(rows, ignore_index=True)
    return rows[FORMAL_COLUMNS], pd.concat(brk, ignore_index=True), rows[FORMAL_COLUMNS[:6] + RAW_COLUMNS]


# ----------------------------------------------------------------------------- 元信息
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def git_info():
    status = _git("status", "--short")
    return dict(commit=_git("rev-parse", "HEAD"), dirty=None if status is None else bool(status))


def cpu_model():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def data_md5s():
    """parquet 的 md5 与 data_manifest.md 登记值核对（原始 csv 不再读取，只引用登记值）。"""
    manifest = (ROOT / "data_manifest.md").read_text(encoding="utf-8")
    raw = re.search(r"raw csv:.*?md5=`([0-9a-f]{32})`", manifest).group(1)
    pq_registered = re.search(r"parquet:.*?md5=`([0-9a-f]{32})`", manifest).group(1)
    pq = md5_of(data.PARQUET)
    if pq != pq_registered:
        sys.exit(f"parquet md5 {pq} 与 data_manifest.md 登记值 {pq_registered} 不一致，停止")
    return dict(raw_csv_md5_from_manifest=raw, parquet_md5=pq, parquet_md5_matches_manifest=True)


def environment_info():
    import scipy, sklearn, pyarrow, threadpoolctl
    return dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                scikit_learn=sklearn.__version__, pyarrow=pyarrow.__version__, threadpoolctl=threadpoolctl.__version__,
                cpu_model=cpu_model(), cpu_count=os.cpu_count(), gpu="none (CPU only)",
                threadpools=[dict(user_api=p["user_api"], internal_api=p["internal_api"], num_threads=p["num_threads"])
                             for p in threadpoolctl.threadpool_info()])


def config_dict(args):
    return dict(splits=list(SPLITS), horizons=list(data.HORIZONS), methods=list(METHODS), stride=STRIDE,
                val_select_stride=VAL_SELECT_STRIDE, clim_half_width=CLIM_HALF_WIDTH, clip=list(CLIP),
                hgb_params=HGB_PARAMS, thresholds=list(ev.THRESHOLDS), min_nse_samples=ev.MIN_SAMPLES,
                max_lookback=data.MAX_LOOKBACK, lags=list(LAGS), rain_windows=list(RAIN_WINDOWS),
                temp_windows=list(TEMP_WINDOWS), humid_windows=list(HUM_WINDOWS),
                n_features=len(FEATURE_NAMES), feature_names=FEATURE_NAMES,
                seeds_note="HGB 无随机成分（无子采样、early_stopping=False），random_state=0，只跑一个种子；"
                           "persistence/气候均值/异常持续为确定性方法",
                cli=dict(subset=args.subset, skip_test=args.skip_test, final=args.final,
                         time_full_hgb=args.time_full_hgb))


# ----------------------------------------------------------------------------- 试运行工具
def pick_subset(fac_codes, stations, n):
    """按站编号轮流、每站取 fac_code 最小的（剩余）水库，直到凑够 n 座；返回升序下标。"""
    lists = []
    for s in range(int(stations.max()) + 1):
        idx = np.nonzero(stations == s)[0]
        lists.append(idx[np.argsort(fac_codes[idx])])
    sel, k = [], 0
    while len(sel) < n:
        before = len(sel)
        for lst in lists:
            if k < len(lst) and len(sel) < n:
                sel.append(lst[k])
        if len(sel) == before:
            break
        k += 1
    return np.array(sorted(sel))


def subset_dataset(ds, sel):
    return data.Dataset(dates=ds.dates, fac_codes=ds.fac_codes[sel], rvow=np.ascontiguousarray(ds.rvow[:, sel]),
                        first_valid=ds.first_valid[sel], dropped_fac_codes=ds.dropped_fac_codes)


def full_sample_counts(ds_full):
    """全量数据下每个 (划分, h) 的样本数：训练（stride）、选迭代用验证子样本、各评估段。"""
    stride_rows = (np.arange(len(ds_full.dates)) % STRIDE == 0)[:, None]
    out = {}
    for name in SPLITS:
        for h in data.HORIZONS:
            m = data.make_split(ds_full, name, h).masks
            c = {seg: int(mask.sum()) for seg, mask in m.items()}
            c["train"] = int((m["train"] & stride_rows).sum())
            if "val" in m:
                c["val_sub"] = -(-c["val"] // VAL_SELECT_STRIDE)
            out[(name, h)] = c
    return out


def estimate_full(rows, counts_full, n_res_sub, n_res_full, skip_test):
    """把试运行各步耗时按（全量/试运行）样本数比例外推；水库数相关的步骤按水库数比例；
    固定成本（读全量数据）不缩放。skip_test 时 main 测试段耗时用验证段耗时按样本数比例估计。"""
    est = []
    for r in rows:
        n_full = r["n_full"]
        if r["step"] in ("base_features", "climatology"):
            n_sub, n_full = n_res_sub, n_res_full
        elif r["n_sub"] is not None and r["split"] is not None and r["h"] is not None:
            c = counts_full[(r["split"], r["h"])]
            key = {"build_train_X": "train", "fit_hgb": "train", "select_iter": "val_sub"}.get(r["step"])
            if key is None and r["step"].startswith("predict_eval_"):
                key = r["step"][len("predict_eval_"):]
            n_sub, n_full = r["n_sub"], c[key]
        else:
            n_sub, n_full = None, None
        f = 1.0 if n_full is None else n_full / n_sub
        est.append(dict(step=r["step"], split=r["split"], h=r["h"], seconds_sub=r["seconds"], n_sub=n_sub,
                        n_full=n_full, est_full_seconds=r["seconds"] * f))
    if skip_test:
        for r in list(est):
            if r["step"] == "predict_eval_val" and r["split"] == "main":
                n_test = counts_full[("main", r["h"])]["test"]
                est.append(dict(step="predict_eval_test (估计，未运行)", split="main", h=r["h"],
                                seconds_sub=r["seconds_sub"], n_sub=r["n_sub"], n_full=n_test,
                                est_full_seconds=r["seconds_sub"] * n_test / r["n_sub"]))
    return est


# ----------------------------------------------------------------------------- 主流程
class Outputs:
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        (self.dir / "predictions").mkdir(parents=True, exist_ok=True)
        self.rows, self.brk, self.raw = [], [], []

    def add_segment(self, name, segment, h, t, res, preds, rows, brk, raw):
        self.rows.append(rows), self.brk.append(brk), self.raw.append(raw)
        np.savez_compressed(self.dir / "predictions" / f"{name}_{segment}_h{h}.npz",
                            t=t.astype(np.int32), res=res.astype(np.int32), **preds)

    def flush(self, log):
        pd.concat(self.rows, ignore_index=True).to_csv(self.dir / "baseline_v2.csv", index=False)
        pd.concat(self.brk, ignore_index=True).to_csv(self.dir / "baseline_v2_breakdown.csv", index=False)
        log["raw_event_counts_log_only"] = json.loads(pd.concat(self.raw, ignore_index=True).to_json(orient="records"))
        log["timings"] = TIMER.rows
        (self.dir / "baseline_v2_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def run_time_full_hgb(args):
    """main、h=10、全量 401 座水库：特征 + HGB 训练 + 验证集选迭代的耗时（只用训练与验证段）。"""
    ds = data.load_dataset()
    cov = load_covariates(ds)
    stations = data.station_ids(ds)
    n_res = len(ds.fac_codes)
    with TIMER.step("base_features", n_sub=n_res, n_full=n_res):
        ctx = make_context(ds, cov, stations)
    with TIMER.step("climatology", "main", n_sub=n_res, n_full=n_res):
        _get_clim(ctx, "main")                               # 单独计时，避免混入 build_train_X
    model = fit_unit(ctx, "main", 10)
    rows = {r["step"]: r["seconds"] for r in TIMER.rows}
    out = dict(unit="main h=10, 全量 %d 座水库，仅训练+验证段" % len(ds.fac_codes), n_train=model["n_train"],
               n_val_sub=model["n_val_sub"], best_it=model["best_it"], n_iter_trained=model["n_iter_trained"],
               seconds=rows,
               environment=environment_info(), git=git_info(), config=config_dict(args))
    dry = ROOT / "results" / "dryrun_sub20" / "baseline_v2_log.json"
    if dry.exists():
        est = {(e["step"], e["split"], e["h"]): e for e in json.loads(dry.read_text(encoding="utf-8")).get("estimates", [])}
        cmp = {}
        for step, split, h in (("base_features", None, None), ("climatology", "main", None),
                               ("build_train_X", "main", 10), ("fit_hgb", "main", 10), ("select_iter", "main", 10)):
            e = est.get((step, split, h))
            if e:
                cmp[step] = dict(measured=rows[step], extrapolated=e["est_full_seconds"],
                                 measured_over_extrapolated=rows[step] / e["est_full_seconds"])
        out["vs_dryrun_extrapolation"] = cmp
    path = ROOT / "results" / "timing_full_hgb_main_h10.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_train", "n_val_sub", "best_it", "n_iter_trained", "seconds",
                                          "vs_dryrun_extrapolation")
                      if k in out}, indent=1))


def run(args):
    with TIMER.step("load_dataset"):
        ds_full = data.load_dataset()
    with TIMER.step("station_ids"):
        stations_full = data.station_ids(ds_full)
    with TIMER.step("load_covariates"):
        cov_full = load_covariates(ds_full)
    if args.subset:
        sel = pick_subset(ds_full.fac_codes, stations_full, args.subset)
        ds = subset_dataset(ds_full, sel)
        cov = dict(rain=cov_full["rain"][:, sel], tempt=cov_full["tempt"][:, sel], humid=cov_full["humid"][:, sel],
                   static=cov_full["static"][sel])
        stations = stations_full[sel]
        out = Outputs(ROOT / "results" / f"dryrun_sub{args.subset}")
        print("试运行水库（fac_code, 站）:", list(zip(ds.fac_codes.tolist(), stations.tolist())))
    else:
        ds, cov, stations = ds_full, cov_full, stations_full
        out = Outputs(ROOT / "results")
    log = dict(config=config_dict(args), data=data_md5s(), environment=environment_info(), git_at_start=git_info(),
               n_reservoirs=len(ds.fac_codes), best_iterations={})
    with TIMER.step("base_features", n_sub=len(ds.fac_codes), n_full=len(ds_full.fac_codes)):
        ctx = make_context(ds, cov, stations)
    segments = {"main": ["val"] + (["test"] if args.final else []), "D1": ["holdout"], "D2": ["holdout"]}
    best_its = {}
    for name in SPLITS:
        with TIMER.step("climatology", name, n_sub=len(ds.fac_codes), n_full=len(ds_full.fac_codes)):
            _get_clim(ctx, name)
        for h in data.HORIZONS:
            print(f"== {name} h={h}", flush=True)
            model = fit_unit(ctx, name, h, best_it=best_its.get(h) if name != "main" else None)
            if name == "main":
                best_its[h] = model["best_it"]
                log["best_iterations"][str(h)] = model["best_it"]
            log.setdefault("rho", {})[f"{name}_h{h}"] = model["rho"]
            log.setdefault("n_iter_trained", {})[f"{name}_h{h}"] = model["n_iter_trained"]
            for seg in segments[name]:
                if seg == "test":
                    log["final"] = dict(eval_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                        **{f"git_{k}": v for k, v in git_info().items()})
                n_seg = int(_get_split(ctx, name, h).masks[seg].sum())
                with TIMER.step(f"predict_eval_{seg}", name, h, n_sub=n_seg):
                    t, res, y0, y, preds = predict_segment(ctx, model, name, h, seg)
                    rows, brk, raw = evaluate_segment(ctx, name, seg, h, t, res, y0, y, preds)
                    out.add_segment(name, seg, h, t, res, preds, rows, brk, raw)
            out.flush(log)
    np.savez_compressed(out.dir / "predictions" / "index.npz", dates=ds.dates.values.astype("datetime64[D]"),
                        fac_codes=ds.fac_codes)
    if args.subset:
        counts = full_sample_counts(ds_full)
        est = estimate_full(TIMER.rows, counts, len(ds.fac_codes), len(ds_full.fac_codes), args.skip_test)
        log["estimates"] = est
        log["full_counts"] = {f"{k[0]}_h{k[1]}": v for k, v in counts.items()}
        df = pd.DataFrame(est)
        print("\n各步骤耗时与全量外推（秒）:")
        print(df.groupby("step", dropna=False)[["seconds_sub", "est_full_seconds"]].sum().round(1).to_string())
        print(f"\n试运行合计 {df.seconds_sub.sum():.0f}s；全量外推合计 {df.est_full_seconds.sum():.0f}s "
              f"（{df.est_full_seconds.sum() / 60:.1f} 分钟）")
    log["environment"] = environment_info()                  # HGB 用过之后再记录，线程池信息才完整
    out.flush(log)
    print("输出目录:", out.dir)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=int, help="只用 N 座水库（试运行）")
    p.add_argument("--skip-test", action="store_true", help="不评估 main 测试集")
    p.add_argument("--final", action="store_true", help="评估 main 测试集（只应在最终评估时使用一次）")
    p.add_argument("--time-full-hgb", action="store_true", help="仅计时：main、h=10、全量的 HGB 训练+验证选迭代")
    a = p.parse_args(argv)
    if a.time_full_hgb:
        if a.final or a.subset:
            p.error("--time-full-hgb 不能与 --final / --subset 同用")
    else:
        if a.final == a.skip_test:
            p.error("必须且只能指定 --final 或 --skip-test 之一（评估 main 测试集必须显式 --final）")
        if a.final and a.subset:
            p.error("--final 不能与 --subset 同用：试运行不得触碰测试集")
    return a


if __name__ == "__main__":
    args = parse_args()
    run_time_full_hgb(args) if args.time_full_hgb else run(args)
