"""评估：全项目唯一的评估实现，所有实验共用（see the full implementation in the companion repository）。

约定：一批样本是一组等长一维数组，对应固定的预测步长 h：
    y   目标值 rvow[t+h]          p   预测值          y0  起点值 rvow[t]
    res 水库下标（0..n_res-1）    t   起点日下标（整数，逐日连续）
预测值按原样评估，不做截断或修正。y、p、y0 必须都是有限值（缺失样本应在上游剔除，不做插值）。

new_exceedance_metrics 是协议 v2 的主指标（命中 = 事件段与告警段共享至少一天）。
new_exceedance_metrics_tolerant 是**非主指标、仅用于敏感性分析**（a post hoc sensitivity analysis; see the companion repository），
把命中标准放宽为「起点相差不超过 tol_days 天」；不改变、不替代 new_exceedance_metrics，不进入已冻结的协议 v2 判据。
year_block_distortion_check 检测按年分块的事件段配对差估计是否被跨年告警段切分严重扭曲；发现于 direct_classifier
实验（τ*=0 时告警段跨年持续、按年分块把长段切碎），用于事后诊断已有按年分块结果是否可信，不改变已冻结的判据本身，
只决定该判据该用按年还是按站分块作为主口径（see the full implementation in the companion repository）。
"""
import numpy as np
import pandas as pd

MIN_SAMPLES = 100            # 样本数不足该值的水库不计入 NSE
THRESHOLDS = (20.0, 30.0, 40.0)  # 低库容阈值（%）；30 为主，20/40 为敏感性分析
GROUPS = ("all", "existing", "cold_start")
_SST_MIN = 1e-6              # 目标值方差退化（近似常数序列）时 NSE 无定义，该水库不计


def samples_from_mask(rvow, mask, h):
    """(T, N) 样本掩码 -> (t, res, y0, y) 四个一维数组，y0=rvow[t]，y=rvow[t+h]。"""
    t, res = np.nonzero(mask)
    return t, res, rvow[t, res], rvow[t + h, res]


def _as_float(*arrays):
    out = [np.asarray(a, dtype=np.float64) for a in arrays]
    for a in out:
        if not np.isfinite(a).all():
            raise ValueError("y、p、y0 含 NaN/inf；缺失样本需在上游剔除")
    return out


# ---------------------------------------------------------------- NSE / RMSE
def nse_per_reservoir(y, p, res, n_res, min_samples=MIN_SAMPLES):
    """(n_res,) 每座水库的 NSE = 1 - SSE/SST，SST 以该水库参与评估的样本自身的均值计。

    样本数 < min_samples、或目标值方差退化（SST <= 1e-6，NSE 无定义）的水库记为 NaN，不参与汇总。
    """
    y, p = _as_float(y, p)
    res = np.asarray(res, dtype=np.int64)
    n = np.bincount(res, minlength=n_res)
    mean = np.bincount(res, y, n_res) / np.maximum(n, 1)
    sst = np.bincount(res, (y - mean[res]) ** 2, n_res)
    sse = np.bincount(res, (y - p) ** 2, n_res)
    ok = (n >= min_samples) & (sst > _SST_MIN)
    out = np.full(n_res, np.nan)
    out[ok] = 1.0 - sse[ok] / sst[ok]
    return out


def nse_summary(nse):
    """逐水库 NSE -> 中位数、25/75 分位数（线性插值）、参与的水库数（非 NaN 个数）。"""
    v = np.asarray(nse)[~np.isnan(nse)]
    if len(v) == 0:
        return dict(nse_median=np.nan, nse_q25=np.nan, nse_q75=np.nan, n_reservoirs=0)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return dict(nse_median=float(med), nse_q25=float(q25), nse_q75=float(q75), n_reservoirs=int(len(v)))


def rmse(y, p):
    """汇总 RMSE：对全部样本（不受 100 样本门槛限制）求 sqrt(mean((y-p)^2))。"""
    y, p = _as_float(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def regression_metrics(y, p, res, n_res, min_samples=MIN_SAMPLES):
    """NSE 中位数/分位数/参与水库数 + 汇总 RMSE + 样本数。"""
    out = nse_summary(nse_per_reservoir(y, p, res, n_res, min_samples))
    out.update(rmse=rmse(y, p), n_samples=int(len(y)))
    return out


# ------------------------------------------------------------ 新发越界事件
def _segments(mask, res, t, replicate, flag):
    """把 mask 内的样本按（水库，连续日期）合并成段，返回 (段数, 段内至少有一个 flag 的段数)。

    连续 = 同一水库、起点日下标相差恰为 1；有缺日（该日不是 mask 内样本）就断开。
    replicate 非 None 时（bootstrap 中同一块被重复抽到），不同副本永远不合并。
    """
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return 0, 0
    r, tt, f = res[idx], t[idx].astype(np.int64), flag[idx]
    keys = (tt, r) if replicate is None else (tt, r, replicate[idx])
    order = np.lexsort(keys)  # 最后一个键优先：先按副本、再按水库、再按日期
    r, tt, f = r[order], tt[order], f[order]
    new = np.ones(len(idx), dtype=bool)
    new[1:] = (r[1:] != r[:-1]) | (tt[1:] - tt[:-1] != 1)
    if replicate is not None:
        rep = replicate[idx][order]
        new[1:] |= rep[1:] != rep[:-1]
    seg = np.cumsum(new) - 1
    n_seg = int(seg[-1]) + 1
    n_flagged = int((np.bincount(seg, weights=f, minlength=n_seg) > 0).sum())
    return n_seg, n_flagged


def new_exceedance_metrics(y, p, y0, res, t, threshold, replicate=None):
    """低库容「新发越界」事件的 precision/recall，同时给原始样本口径与事件段口径。

    只在「起点不低于阈值」的样本（y0 >= threshold，起点恰好等于阈值算不低于）里评估：
      真实事件样本：y < threshold（目标低于阈值，严格小于）
      告警样本：    p < threshold（预测低于阈值，严格小于）

    原始样本口径：events_raw / alerts_raw / tp_raw（既是事件又是告警）逐样本计数，
      precision_raw = tp_raw / alerts_raw，recall_raw = tp_raw / events_raw。

    事件段口径（按事件段去重，主指标）：
      - 事件段：同一水库、起点日连续的事件样本合并为一段（起点日差 1 天才连续，缺日即断开）。
      - 告警段：同一水库、起点日连续的告警样本合并为一段，定义同上。
      - 命中：某事件段内只要有一个起点被预测为越界（即该段含告警样本），这一段就算被命中。
      - recall_seg    = 被命中的事件段数 / 事件段总数。
      - precision_seg = 与真实事件段有重叠的告警段数 / 告警段总数；重叠 = 两段共享至少一个起点日，
        即告警段内至少有一个样本本身就是真实事件样本。
    没有事件时 recall 为 NaN；没有告警时 precision 为 NaN（不定义为 0 或 1）。

    replicate：可选，每个样本的副本编号，供 block bootstrap 使用（见 block_bootstrap）。
    """
    y, p, y0 = _as_float(y, p, y0)
    res, t = np.asarray(res), np.asarray(t)
    rep = None if replicate is None else np.asarray(replicate)
    eligible = y0 >= threshold
    event = eligible & (y < threshold)
    alert = eligible & (p < threshold)
    tp = event & alert
    events_raw, alerts_raw, tp_raw = int(event.sum()), int(alert.sum()), int(tp.sum())
    events_seg, events_hit = _segments(event, res, t, rep, alert)
    alerts_seg, alerts_overlap = _segments(alert, res, t, rep, event)
    nan = float("nan")
    return dict(
        threshold=float(threshold),
        n_eligible=int(eligible.sum()),
        events_raw=events_raw, alerts_raw=alerts_raw, tp_raw=tp_raw,
        precision_raw=tp_raw / alerts_raw if alerts_raw else nan,
        recall_raw=tp_raw / events_raw if events_raw else nan,
        events_seg=events_seg, events_seg_hit=events_hit,
        alerts_seg=alerts_seg, alerts_seg_overlap=alerts_overlap,
        precision_seg=alerts_overlap / alerts_seg if alerts_seg else nan,
        recall_seg=events_hit / events_seg if events_seg else nan,
    )


def _segment_starts(mask, res, t):
    """把 mask 内的样本按（水库，连续日期）合并成段（规则与 _segments 相同），返回每段的 (水库, 起点日)。

    不支持 replicate（这个函数只用于 new_exceedance_metrics_tolerant 的敏感性分析，不用于 bootstrap）。
    """
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return np.array([], dtype=res.dtype), np.array([], dtype=np.int64)
    r, tt = res[idx], t[idx].astype(np.int64)
    order = np.lexsort((tt, r))
    r, tt = r[order], tt[order]
    new = np.ones(len(idx), dtype=bool)
    new[1:] = (r[1:] != r[:-1]) | (tt[1:] - tt[:-1] != 1)
    seg = np.cumsum(new) - 1
    n_seg = int(seg[-1]) + 1
    first = np.searchsorted(seg, np.arange(n_seg))     # seg 单调不减，first[i] 是段 i 第一个样本的位置
    return r[first], tt[first]


def _tolerant_hit(seg_res_a, seg_start_a, seg_res_b, seg_start_b, tol_days):
    """对 a 的每一段，判断同一水库内是否存在 b 的某一段，其起点与 a 这段的起点相差 <= tol_days 天
    （存在性判定，非独占匹配：a 的一段可以和 b 的多段都「够近」，只要有一个满足即算命中，与 _segments
    现有的重叠判定同一逻辑，只是把「共享至少一天」换成「起点相差不超过 tol_days 天」）。"""
    hit = np.zeros(len(seg_start_a), dtype=bool)
    if len(seg_start_a) == 0 or len(seg_start_b) == 0:
        return hit
    order_b = np.argsort(seg_res_b, kind="stable")
    res_b, start_b = seg_res_b[order_b], seg_start_b[order_b]
    uniq_res_b, group_starts = np.unique(res_b, return_index=True)
    group_ends = np.append(group_starts[1:], len(res_b))
    lookup = {int(r): np.sort(start_b[lo:hi]) for r, lo, hi in zip(uniq_res_b, group_starts, group_ends)}
    for i, (r, s) in enumerate(zip(seg_res_a, seg_start_a)):
        starts = lookup.get(int(r))
        if starts is None:
            continue
        pos = np.searchsorted(starts, s)
        cand = starts[max(0, pos - 1): pos + 1]
        if len(cand) and np.min(np.abs(cand - s)) <= tol_days:
            hit[i] = True
    return hit


def new_exceedance_metrics_tolerant(y, p, y0, res, t, threshold, tol_days):
    """**非主指标、仅用于敏感性分析**（不是协议 v2 的判据，不进入已冻结结果；见 S4 事后分析）。

    与 new_exceedance_metrics 的事件段/告警段构造完全相同（同样的 eligible/event/alert 定义、同样的
    连续日合并规则），唯一区别是命中标准：不再要求事件段与告警段「共享至少一天」，改成「同一水库内，
    事件段与告警段各自的起点日相差不超过 tol_days 天」，recall 与 precision 两侧对称放宽。存在性判定，
    非独占匹配（与 new_exceedance_metrics 一致）。

    **tol_days=0 时不等价于 new_exceedance_metrics 的 recall_seg/precision_seg**：0 天在这里的意思是
    「两段起点恰好是同一天」，比"两段共享至少一天"更严格（例如事件段 100–105 与告警段 103–110 在原始
    标准下算重叠命中，起点相差 3 天，这里 tol_days=0 不算命中）。因此这个函数不提供、也不应被当作
    tol_days=0 时对主指标的复现；对照「当前 ±0 天结果」请直接使用 new_exceedance_metrics 的输出，这个
    函数只用来报告 tol_days ∈ {3, 7} 等正的容差下 recall/precision 如何变化。

    没有事件时 recall 为 NaN；没有告警时 precision 为 NaN（与 new_exceedance_metrics 一致，不定义为 0 或 1）。
    不支持 replicate，不用于 bootstrap。
    """
    y, p, y0 = _as_float(y, p, y0)
    res, t = np.asarray(res), np.asarray(t)
    eligible = y0 >= threshold
    event = eligible & (y < threshold)
    alert = eligible & (p < threshold)
    ev_res, ev_start = _segment_starts(event, res, t)
    al_res, al_start = _segment_starts(alert, res, t)
    events_seg, alerts_seg = len(ev_start), len(al_start)
    events_hit = int(_tolerant_hit(ev_res, ev_start, al_res, al_start, tol_days).sum())
    alerts_overlap = int(_tolerant_hit(al_res, al_start, ev_res, ev_start, tol_days).sum())
    nan = float("nan")
    return dict(
        threshold=float(threshold), tol_days=int(tol_days),
        events_seg=events_seg, events_seg_hit=events_hit,
        alerts_seg=alerts_seg, alerts_seg_overlap=alerts_overlap,
        precision_seg=alerts_overlap / alerts_seg if alerts_seg else nan,
        recall_seg=events_hit / events_seg if events_seg else nan,
    )


# --------------------------------------------------------------- 汇总（三组）
def evaluate(y, p, y0, res, t, is_cold_start, thresholds=THRESHOLDS, min_samples=MIN_SAMPLES):
    """三组结果：all（全部水库）、existing（训练期已有观测）、cold_start（新增/冷启动）。

    is_cold_start：(n_res,) bool，见 data.cold_start_flags。返回
    {组名: {"regression": {...}, "exceedance": {阈值: {...}}}}。
    """
    cold = np.asarray(is_cold_start, dtype=bool)
    n_res = len(cold)
    y, p, y0 = _as_float(y, p, y0)
    res, t = np.asarray(res), np.asarray(t)
    reservoir_masks = {"all": np.ones(n_res, dtype=bool), "existing": ~cold, "cold_start": cold}
    out = {}
    for g in GROUPS:
        s = reservoir_masks[g][res]
        out[g] = dict(
            regression=regression_metrics(y[s], p[s], res[s], n_res, min_samples),
            exceedance={thr: new_exceedance_metrics(y[s], p[s], y0[s], res[s], t[s], thr) for thr in thresholds},
        )
    return out


def summary_frame(results):
    """evaluate 的返回值 -> 每个（组, 阈值）一行的 DataFrame（回归指标在各阈值行重复）。"""
    rows = []
    for g, r in results.items():
        for thr, e in r["exceedance"].items():
            rows.append({"group": g, **r["regression"], **e})
    return pd.DataFrame(rows)


def breakdown_table(y, p, res, n_res, years, stations, min_samples=MIN_SAMPLES):
    """每一年、每个气象站的单独结果：样本数、参与 NSE 的水库数、NSE 中位数、RMSE。

    years：每个样本的起点年份；stations：每个样本所属气象站编号（如 data.station_ids(ds)[res]）。
    返回长表：by ∈ {"year", "station"}，key 为年份或站编号。
    """
    y, p = _as_float(y, p)
    res = np.asarray(res)
    rows = []
    for by, keys in (("year", np.asarray(years)), ("station", np.asarray(stations))):
        for k in np.unique(keys):
            m = keys == k
            r = regression_metrics(y[m], p[m], res[m], n_res, min_samples)
            rows.append(dict(by=by, key=int(k), n_samples=r["n_samples"], n_reservoirs=r["n_reservoirs"],
                             nse_median=r["nse_median"], rmse=r["rmse"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- block bootstrap
def make_blocks(*keys):
    """把若干等长的分组键组合成整数块编号（每个不同的键组合一个块）。"""
    stacked = np.stack([np.asarray(k) for k in keys], axis=1)
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    return ids.reshape(-1)


def block_bootstrap(stat_fn, block_ids, n_boot=1000, seed=0, ci=0.95):
    """块自助法：以块为单位有放回地重抽，重复计算统计量。

    stat_fn(idx, replicate) -> float：idx 是重抽后的样本下标（数组，可含重复），replicate 是每个样本
    来自第几次抽取（同一块被抽到两次时副本编号不同；按事件段统计的指标需要它以免跨副本合并段）。
    只算 NSE/RMSE 的统计量可忽略 replicate。返回点估计、bootstrap 均值、百分位区间与全部重抽结果。
    比较两个模型时，让 stat_fn 直接返回二者指标之差（配对差），用同一批重抽下标。

    通用局限：块内样本仍是相依的；重抽只反映块间波动，且百分位区间在块数少时很粗糙。
    请优先用 bootstrap_by_year（主）和 bootstrap_by_station（敏感性），并读它们各自的局限说明。
    """
    block_ids = np.asarray(block_ids)
    n = len(block_ids)
    order = np.argsort(block_ids, kind="stable")
    _, starts, counts = np.unique(block_ids[order], return_index=True, return_counts=True)
    k = len(starts)
    rng = np.random.default_rng(seed)
    estimate = float(stat_fn(np.arange(n), np.zeros(n, dtype=np.int64)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, k, size=k)
        idx = np.concatenate([order[starts[j]: starts[j] + counts[j]] for j in draw])
        rep = np.repeat(np.arange(k), counts[draw])
        boots[b] = stat_fn(idx, rep)
    a = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)])
    return dict(estimate=estimate, boot_mean=float(np.nanmean(boots)), ci_low=float(lo), ci_high=float(hi),
                n_blocks=int(k), n_boot=int(n_boot), seed=int(seed), ci=float(ci), boots=boots)


def bootstrap_by_year(stat_fn, years, **kw):
    """按年分块的 block bootstrap（主要方案）：一个起点年份 = 一个块，年内所有水库、所有站一起重抽。

    这样保留了同一年内各站、各水库共享的天气冲击，是本项目的主要不确定性度量。
    局限：main 测试期（2016–2020）只有 5 个年份块，重抽只有 5^5 种（不计顺序仅 126 种），
    百分位区间很粗、偏窄且不稳定，不应把它当作精确的置信区间；只能说明年际波动的量级。
    年份块内部的时间自相关和水库间相依性都没有被建模；年份块也不能回答「换一批水库会怎样」。
    """
    return block_bootstrap(stat_fn, years, **kw)


def bootstrap_by_station(stat_fn, stations, **kw):
    """按气象站分块的 block bootstrap（敏感性分析）：一个站 = 一个块，站内所有水库、所有年份一起重抽。

    这样把同站水库共用同一条气象序列这一依赖考虑进来，不把 401 座当独立样本。
    局限：只有 7 个聚类，且规模极不均（每站 33–111 座水库）；每次重抽的有效样本量随抽到大站还是小站
    剧烈变化，区间宽而不稳。每个站的全部年份被整体重抽，同一年各站共享的天气冲击没有被重抽，
    因此对年际天气波动的不确定性会被低估，只宜作为对按年分块结果的补充。
    """
    return block_bootstrap(stat_fn, stations, **kw)


# ------------------------------------------------------------- 按年分块失真诊断
YEAR_BLOCK_REL_THRESHOLD = 0.20   # 相对差门槛：在看到分类器失真之后选定，依据与不敏感性范围见 year_block_distortion_check 文档字符串
YEAR_BLOCK_ABS_FLOOR = 0.03       # 真实值接近 0 时相对差没有意义，改用绝对差门槛；与本项目其它地方的
                                   # MIN_PRACTICAL_DIFF 同一量级、同一定位（预先设定的实践阈值，不是统计推导值）


def year_block_distortion_check(year_estimate, true_pooled_estimate, rel_threshold=YEAR_BLOCK_REL_THRESHOLD,
                                 abs_floor=YEAR_BLOCK_ABS_FLOOR):
    """检测「按年分块（先按年切样本、块内单独算事件段、再求和）算出的点估计」是否偏离「全量样本不分块直接算
    出的真实点估计」。

    背景：事件段/告警段是按连续日期合并的；按年分块把样本先按起点年份切开、块内单独重新计算事件段，如果
    一段真实的事件段或告警段跨越年份边界，会被这一步切成两段——`block_counts` 的文档字符串已经提到这一点
    （"按年分块会在年份边界处切开跨年的事件段"），但原先只被当作一个小的边界效应：本项目里大多数告警段
    生命周期很短（几天到几十天），跨年的比例很低，切开的影响可以忽略。当告警策略产生长期持续的告警（比如
    「资格起点就告警」这种简并策略，见 direct_classifier 实验）时，告警段经常横跨多个年份，按年分块会把
    这些长段反复切碎，告警段数被人为放大数倍，precision/recall/F1 因此被严重扭曲——这不是随机噪声，是
    确定性的：`year_estimate` 与 `true_pooled_estimate` 是对同一批数据的两次确定性重新计算（不是两次独立
    抽样），差异只可能来自「块内重算事件段」这一步造成的分段方式改变，不会有统计噪声。因此没有切分（或只有
    极少数、贡献很小的段被切分）的正常情况下，两者应该完全相等或只有可忽略的差异；只要出现了有意义的差异，
    说明确实发生了非平凡的段切分。

    **门槛依据与出处（如实记录）**：`rel_threshold=0.20`、`abs_floor=0.03` 是在 direct_classifier 实验的按年分块
    失真已经被观察到、并且已经看过保形/GRU 已保存输出里「按年估计与全量估计只差 1% 以内」之后才选定的——
    **不是盲选、也不是看到数据之前预先写死的**，早先版本的注释里「预先设定、不是从任何一次结果反推」这句话过头，
    已更正。选 20% 的理由只有两条：(1) 两个估计是对同一批数据的确定性重算而非独立抽样，没有统计噪声，
    所以门槛不需要覆盖抽样波动，只需要把「个别事件段恰在年份边界被切开」这种小扰动与「持续性告警被系统性切碎」
    区分开；(2) 取一个整数比例、留出足够余量。**判定结果对门槛具体取值不敏感**（这一点才是可辩护的依据，见
    scripts/audit_year_block_distortion.py 的回溯审计）：在已保存输出的 559 个按年估计里，除直接事件分类器外
    按年相对差最大只有 6.0%（GRU 事后配对差）；分类器落在相对分支的两行是 94%（h=20）与 74.1%（h=30），
    h=10 因真实值 −0.030 落在近零区走绝对分支（绝对差 0.338，相对差数值上约 1127%，但判定并未使用它）。
    因此相对门槛取 (6%, 74%) 之间任何值，被标记的行集合都完全相同；门槛落在这个区间之外才会改变结论。
    `abs_floor=0.03`：当 `true_pooled_estimate` 接近 0 时相对差的分母趋于 0、相对差本身失去意义，改用
    绝对差门槛，数值直接沿用本项目的 `MIN_PRACTICAL_DIFF`（同一个「小于它就不算有实际意义」的量级，不另立新数）；
    近零区里除分类器外，已保存输出的按年绝对差最大只有 0.0012，同样远低于 0.03。

    返回 dict：`distorted`（bool）、`abs_diff`、`rel_diff`（`true_pooled_estimate` 太接近 0 时为 None）、
    `basis`（"relative" 或 "absolute_near_zero"）、`recommendation`（中文说明，供直接打印/写日志）。
    """
    abs_diff = abs(year_estimate - true_pooled_estimate)
    if abs(true_pooled_estimate) >= abs_floor:
        rel_diff = abs_diff / abs(true_pooled_estimate)
        distorted = rel_diff > rel_threshold
        basis = "relative"
    else:
        rel_diff = None
        distorted = abs_diff > abs_floor
        basis = "absolute_near_zero"
    recommendation = ("按年分块失真，改用按站分块作为主口径" if distorted else "按年分块与全量不分块结果一致，可信，维持按年分块为主口径")
    return dict(distorted=bool(distorted), year_estimate=float(year_estimate), true_pooled_estimate=float(true_pooled_estimate),
               abs_diff=float(abs_diff), rel_diff=(None if rel_diff is None else float(rel_diff)), basis=basis,
               rel_threshold=float(rel_threshold), abs_floor=float(abs_floor), recommendation=recommendation)
