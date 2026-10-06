"""保形校准与事件段评估（预注册，不是事后分析）。

方法性质：使用保形的构造思路，得到经验校准的区间，不是有理论保证的区间。
声明：这是预注册实验，不是事后分析；设计、判据与分两阶段运行的流程在看到任何结果之前写入日志，并在运行前随脚本提交。

基线模型：v2 固定超参的 pooled HGB；预测直接使用已保存的预测（已截断到 [0, 100]；see the companion repository），
不重新训练。校准集 = main 验证集（2011–2015）。验证集在整个流程里被使用了两次：一次用来选 v2 的 HGB 迭代次数，一次用来做校准；
因此验证集上的覆盖率是样本内的，接近名义值只是构造的结果，不能说明校准成立。测试集预测只在阶段 B（--final）读取一次，
且只用于评估已冻结的配置，不得做其他探索。

方法（分裂保形，按步长 h=10/20/30 分别校准）：
  有符号残差 r = y − ŷ（ŷ 为已截断的点预测）。覆盖率档位 c ∈ {50, 60, 70, 80, 90, 95}%（主设定 90%），α = 1 − c。
  区间下界取第 k_lo = max(1, floor((n+1)·α/2)) 个次序统计量，上界取第 k_hi = min(n, ceil((n+1)·(1−α/2))) 个次序统计量
  （整数运算）；区间 [ŷ + q_lo, ŷ + q_hi] 再截断到 [0, 100]。
  主变体 = 边缘校准（每个 h 用全部验证样本）；次要变体 M = 按起点库容分四桶（<20、20–30、30–60、≥60，含下不含上）、
  每个 h、每个桶各自校准，其余设定相同。两者并列报告，不用于挑选；不得再增加其他变体。
事件段评估：告警规则「区间下界 < 阈值」（严格小于）；阈值 30%（主）、20%、40%（敏感性）；只统计起点不低于阈值的样本，
  按事件段计数（直接复用 ev.new_exceedance_metrics，把下界数组当作预测传入）。对照 = 点预测告警（ŷ < 阈值，即 v2 已有结果）。
  只报告事件段数、precision_seg、recall_seg；事件段数少于 30 的行标记 n_events_lt_30；不报告原始样本数。
覆盖率诊断：双侧边缘覆盖率 P(lo <= y <= hi)、下界一侧未覆盖率 P(y < lo)（名义值 (1−c)/2）、上侧 P(y > hi)；
  按 h、起点库容分桶、按年分别报告（验证集为样本内；测试集才是检验）。
不确定性：主设定（90%）与点预测的召回、精确率及配对差（保形 − 点预测），用 ev.bootstrap_by_year（主，起点年份为块）和
  ev.bootstrap_by_station（敏感性，7 个气象站为块）给 95% 百分位区间，n_boot=5000，种子固定。事件段指标在块内可加，
  所以先对每个块算一次事件段计数，再用「每个块一行」的伪样本喂给现成的 bootstrap 函数，与直接对全部样本重抽等价。
  局限：验证期与测试期各只有 5 个年份块；7 个气象站聚类且规模不均；按年分块会在年份边界处切开跨年的事件段（日志报告切开的段数）。
权衡曲线：召回对精确率，档位 {50, 60, 70, 80, 90, 95}% 加点预测端点；档位与主设定都是预先固定的。

预先设定的判据（先于结果写入日志；只用于描述结论，不用于挑选变体；分别对主变体与 M 变体判定）：
  (a) 校准（测试集，逐步长）：90% 区间的边缘覆盖率与 0.90 相差不超过 0.05，且下界一侧未覆盖率与 0.05 相差不超过 0.03，
      该步长记为「大体校准」，否则「未校准」；三个步长都「大体校准」才记整体「大体校准」，否则整体「未校准」，并如实报告数值。
  (b) 事件检测增益（测试集，30% 阈值，h=20 与 h=30 两个步长，主设定 90%）：保形下界告警相对点预测告警的召回提高，并且精确率不低于 0.3，
      记为「有用的增益」；只有召回提高、精确率低于 0.3，记为「以精确率为代价提高召回」；召回没有提高，记为「无增益」。
      两个步长的标记相同则取该标记，否则整体记为「混合（逐步长报告）」。
      验证集上也算同样的标记，但那是样本内的，只描述。

操作点规则（先于任何测试结果写入日志与冻结文档；see the companion repository）：对每个变体、每个 h，在验证集、30% 阈值下，
  从档位 {50, 60, 70, 80, 90, 95}% 中选使 F1（召回与精确率的调和平均，事件段口径）最大的档位 c*，并列（相差不超过 1e-12）取较低档位，
  精确率无定义的档位不参与；只用阶段 A 已有的验证集事件段数值计算，不重新拟合。c* 只用于在测试集单独报告一个「指定操作点」
  （召回、精确率、F1、覆盖率与其 bootstrap 区间）；主设定仍是 90%，判据 (a)(b) 仍按 90% 判定，不因 c* 改变。
流程（分两阶段）：
  阶段 A（--skip-test）：校准、验证集覆盖率、验证集事件段指标与曲线、验证集 bootstrap、验证集零对照、按年分位数（只描述，
      不影响冻结的分位数）；产出 calibration_proposed.json（边缘与 M 变体的全部分位数：每个 h、每个档位、每个桶）。
  阶段 B（--final）：工作区必须干净（git 无未提交改动）。先断言：(a) 冻结文件（阶段 A 的
      calibration_proposed.json 的字节相同副本）与提交版一致，且重新计算的验证集分位数与之逐位相同；(b) 操作点文件与提交版一致，且由重新计算的
      验证集事件段数值按规则得到的操作点与之相同。然后读取测试预测（只读取一次，读入内存），先做测试集零对照（分位数取 0 还原 v2 已提交的
      测试集事件段指标；零对照本身必须用测试预测的点预测，所以读取先于零对照），不通过就停止，此前不做任何保形评估；通过后只评估冻结的配置：
      两个变体 × 6 个档位 × 3 个阈值 × 3 个 h，加每个（变体，h）的 c* 操作点，不做其他探索。
  必须且只能指定 --final 或 --skip-test 之一，否则拒绝运行；没有 --final 时不读取测试预测，也不构造测试样本掩码。
零对照：把 q_lo = q_hi = 0，区间退化为点预测，告警必须还原 v2 已有的事件段指标（已保存的 v2 基线汇总表）：
  事件段数、命中数、告警数、重叠数逐项相等，召回与精确率一致。阶段 A 做验证集，阶段 B 做测试集；不通过就停止。
  （区间中点 ŷ + (q_lo+q_hi)/2 一般不等于 ŷ，所以零对照取 q=0。）
局限：验证集用了两次；边缘校准不是条件校准，覆盖率可能随起点库容、年份而变；验证残差在时间与水库间相依，且测试期存在分布漂移，
  交换性不成立，理论覆盖结论不适用，测试覆盖率只是描述；事件段数在 20% 阈值下较少。

用法: python scripts/run_conformal_events.py --skip-test      （阶段 A）
      python scripts/run_conformal_events.py --final          （阶段 B，冻结之后）
输出: 本脚本的输出目录
"""
import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import data, eval as ev


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.__name__ != "__main__"
    return mod


s1m = _load_module("run_baselines_sensitivity_s1", "run_baselines_sensitivity_s1.py")   # 只用 registered / assert_matches_v2_log
rb, dg = s1m.rb, s1m.dg
AbortRun = s1m.AbortRun

OUT = ROOT / "results" / "conformal"
PRED = ROOT / "results" / "predictions"
BASELINE_CSV = ROOT / "docs" / "baseline_v2" / "baseline_v2.csv"
FROZEN = ROOT / "docs" / "conformal" / "calibration_frozen.json"
OP_FILE = ROOT / "docs" / "conformal" / "operating_points_frozen.json"
NAME = "C0"                                                     # 阶段 A 的划分：train 1991–2010 + val 2011–2015，没有 test 段
HS = data.HORIZONS
LEVELS = (50, 60, 70, 80, 90, 95)
MAIN_LEVEL = 90
THRESHOLDS = (30.0, 20.0, 40.0)
MAIN_THRESHOLD = 30.0
BUCKETS = ("<20", "20-30", "30-60", ">=60")
VARIANTS = ("marginal", "M")
N_BOOT, BOOT_SEED, CI = 5000, 20260925, 0.95
N_EVENTS_MIN = 30
COVERAGE_TOL, LOWER_MISS_TOL, NOMINAL_LOWER, PRECISION_MIN = 0.05, 0.03, 0.05, 0.3
FLOAT_EPS = 1e-12                                               # 吸收浮点误差，使「不超过」的边界取等
GAIN_STEPS = (20, 30)
COUNT_COLS = ("events_seg", "events_seg_hit", "alerts_seg", "alerts_seg_overlap")
NO_GUARANTEE_SENTENCE = "使用保形的构造思路，得到经验校准的区间，不是有理论保证的区间"
VALIDATION_TWICE_TEXT = "验证集被使用了两次：一次用来选 v2 的 HGB 迭代次数，一次用来做保形校准；验证集上的覆盖率是样本内的。"
DECLARATION = ("预注册实验，不是事后分析：设计、判据与分两阶段运行的流程在看到任何结果之前写入日志，并在运行前随脚本提交。"
               "测试集预测只在阶段 B（--final）读取一次，且只用于评估已冻结的配置。")
F1_TIE_TOL = 1e-12
OPERATING_POINT_TEXT = (
    "对每个变体、每个 h，在验证集、30% 阈值下，从档位 {50,60,70,80,90,95} 中选使 F1（召回与精确率的调和平均，事件段口径）最大的档位 c*，"
    "并列（相差不超过 1e-12）取较低档位，精确率无定义的档位不参与；用阶段 A 的 val_event_metrics.csv 现成数值计算，不重新拟合。"
    "c* 只用于在测试集单独报告一个「指定操作点」；主设定仍是 90%，判据 (a)(b) 仍按 90% 判定，不因 c* 改变。")
CRITERIA_TEXT = dict(
    a=("测试集逐步长：90% 区间的边缘覆盖率与 0.90 相差不超过 0.05，且下界一侧未覆盖率与 0.05 相差不超过 0.03，记为「大体校准」，否则「未校准」；"
       "三个步长都「大体校准」才记整体「大体校准」，否则整体「未校准」，并如实报告数值。"),
    b=("测试集，30% 阈值，h=20 与 h=30，主设定 90%：保形下界告警相对点预测告警的召回提高且精确率不低于 0.3 记「有用的增益」；"
       "只有召回提高、精确率低于 0.3 记「以精确率为代价提高召回」；召回没有提高记「无增益」；两个步长标记相同则取该标记，否则「混合（逐步长报告）」。"),
    scope="分别对主变体（边缘校准）与 M 变体判定；判据只用于描述结论，不用于挑选变体。验证集上同样的标记是样本内的，只描述。")


# ----------------------------------------------------------------------------- 分位数与区间（纯函数，有测试）
def interval_ranks(n, level_pct):
    """整数运算：k_lo = max(1, floor((n+1)·α/2))，k_hi = min(n, ceil((n+1)·(1−α/2)))，α = 1 − level_pct/100。"""
    alpha = 100 - int(level_pct)
    k_lo = max(1, ((n + 1) * alpha) // 200)
    k_hi = min(n, -((-(n + 1) * (200 - alpha)) // 200))
    return int(k_lo), int(k_hi)


def conformal_quantiles(residuals, level_pct):
    """有符号残差的下、上分位数（次序统计量）与校准样本数。"""
    r = np.asarray(residuals, dtype=np.float64)
    n = len(r)
    if n == 0:
        raise AbortRun("校准样本数为 0")
    k_lo, k_hi = interval_ranks(n, level_pct)
    part = np.partition(r, sorted({k_lo - 1, k_hi - 1}))
    return dict(n=int(n), k_lo=k_lo, k_hi=k_hi, q_lo=float(part[k_lo - 1]), q_hi=float(part[k_hi - 1]))


def bucket_index(y0):
    """起点库容分桶编号 0..3（<20、20–30、30–60、≥60，含下不含上）。"""
    masks = dg.bucket_masks(np.asarray(y0))
    idx = np.full(len(y0), -1, dtype=np.int64)
    for i, b in enumerate(BUCKETS):
        idx[masks[b]] = i
    if (idx < 0).any():
        raise AbortRun("有样本不属于任何库容分桶")
    return idx


def calibrate_h(y, p, bidx, levels=LEVELS):
    """一个步长的校准：边缘（全部样本）与 M 变体（每个桶各自）在每个档位的分位数。键都是字符串，便于写成 JSON。"""
    r = np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    marginal = {str(c): conformal_quantiles(r, c) for c in levels}
    m = {b: {str(c): conformal_quantiles(r[bidx == i], c) for c in levels} for i, b in enumerate(BUCKETS)}
    return dict(marginal=marginal, M=m)


def make_bounds(variant, cal_h, level, p, bidx):
    """区间下界与上界（已截断到 [0, 100]）。M 变体按每个样本所在的桶取各自的分位数。"""
    p = np.asarray(p, dtype=np.float64)
    if variant == "marginal":
        q = cal_h["marginal"][str(level)]
        q_lo, q_hi = q["q_lo"], q["q_hi"]
    else:
        q_lo = np.array([cal_h["M"][b][str(level)]["q_lo"] for b in BUCKETS])[bidx]
        q_hi = np.array([cal_h["M"][b][str(level)]["q_hi"] for b in BUCKETS])[bidx]
    return np.clip(p + q_lo, 0.0, 100.0), np.clip(p + q_hi, 0.0, 100.0)


def coverage_stats(y, lo, hi):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return dict(n=0, coverage=np.nan, lower_miss=np.nan, upper_miss=np.nan)
    return dict(n=int(n), coverage=float(np.mean((y >= lo) & (y <= hi))), lower_miss=float(np.mean(y < lo)),
                upper_miss=float(np.mean(y > hi)))


def event_row(y, lo, y0, res, t, thr):
    """告警规则「下界 < 阈值」：把下界数组当作预测传给现成的事件段函数。只保留事件段口径的字段。"""
    e = ev.new_exceedance_metrics(y, lo, y0, res, t, thr)
    row = {k: e[k] for k in COUNT_COLS}
    row.update(precision_seg=e["precision_seg"], recall_seg=e["recall_seg"], n_events_lt_30=bool(e["events_seg"] < N_EVENTS_MIN))
    return row


def assert_matches_frozen(recomputed, frozen):
    """阶段 B：重新计算的验证集分位数必须与冻结文件完全相同（键、样本数、秩、分位数值）。"""
    for key in ("levels_pct", "n_val", "marginal", "M"):
        if recomputed.get(key) != frozen.get(key):
            raise AbortRun(f"重新计算的验证集校准（{key}）与冻结文件不完全相同；已停止，不读取测试预测")


# ----------------------------------------------------------------------------- 操作点（纯函数，有测试）
def f1_score(precision, recall):
    """召回与精确率的调和平均；任一为 NaN 则 NaN；两者都是 0 时记 0。"""
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def read_events_csv(path):
    """事件段指标 csv 的精确读取（float_precision="round_trip"）：默认的快速解析器可能与写出值差 1 个 ulp，
    而操作点要与重新计算的数值逐位比较。"""
    return pd.read_csv(path, float_precision="round_trip")


def select_operating_points(events, split="val", threshold=MAIN_THRESHOLD):
    """events：事件段指标表（event_table 的输出或其 csv）。返回 {变体: {h: {levels: {档位: {...}}, c_star}}}（键为字符串，便于写 JSON）。
    F1 最大的档位；并列（相差不超过 F1_TIE_TOL）取较低档位；精确率无定义的档位不参与。"""
    out = {}
    for variant in VARIANTS:
        out[variant] = {}
        for h in HS:
            d = events[(events.split == split) & (events.variant == variant) & (events.h == h) & (events.threshold == threshold)]
            d = d.sort_values("level")
            levels, cands = {}, []
            for r in d.itertuples():
                f1 = f1_score(r.precision_seg, r.recall_seg)
                levels[str(int(r.level))] = dict(recall_seg=float(r.recall_seg), precision_seg=None if np.isnan(r.precision_seg) else float(r.precision_seg),
                                                 f1=None if np.isnan(f1) else f1, events_seg=int(r.events_seg), alerts_seg=int(r.alerts_seg))
                if not np.isnan(f1):
                    cands.append((int(r.level), f1))
            if not cands:
                raise AbortRun(f"{variant} h={h}: 没有精确率有定义的档位，无法选择操作点")
            best = max(f for _, f in cands)
            c_star = min(c for c, f in cands if f >= best - F1_TIE_TOL)
            out[variant][str(h)] = dict(levels=levels, c_star=c_star)
    return out


def operating_points_document(points, source):
    return dict(status="frozen（先于任何测试结果）", rule=OPERATING_POINT_TEXT, threshold=MAIN_THRESHOLD, levels_pct=list(LEVELS),
                tie_tolerance=F1_TIE_TOL, source=source, points=points)


def _norm_newlines(b):
    return b.replace(b"\r\n", b"\n")


def committed_bytes(path):
    """path 在 HEAD 提交里的内容；无法读取则抛 AbortRun。"""
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        raise AbortRun(f"{rel} 不在 HEAD 提交里（冻结文件必须先提交）")
    return r.stdout


def assert_matches_committed(path):
    """工作区文件必须与提交版一致（忽略换行符转换：Windows 上 git 会做 CRLF 转换）。"""
    if _norm_newlines(Path(path).read_bytes()) != _norm_newlines(committed_bytes(path)):
        raise AbortRun(f"{Path(path).name} 与提交版不一致")


def assert_operating_points_match(recomputed_points, file_doc):
    """由重新计算的验证集事件段数值按规则得到的操作点，必须与冻结文件完全相同（经 JSON 往返，比较精确值）。"""
    if json.loads(json.dumps(recomputed_points)) != file_doc.get("points"):
        raise AbortRun("由重新计算的验证集数值得到的操作点与冻结的操作点文件不一致；已停止，不读取测试预测")


# ----------------------------------------------------------------------------- 判据（纯函数，有测试）
def criterion_a_step(coverage, lower_miss):
    ok = (abs(coverage - 0.90) <= COVERAGE_TOL + FLOAT_EPS) and (abs(lower_miss - NOMINAL_LOWER) <= LOWER_MISS_TOL + FLOAT_EPS)
    return "大体校准" if ok else "未校准"


def criterion_a(per_h):
    """per_h: {h: (90% 区间边缘覆盖率, 下界一侧未覆盖率)}。三个步长都「大体校准」才整体「大体校准」。"""
    steps = {int(h): criterion_a_step(c, m) for h, (c, m) in per_h.items()}
    return dict(per_h=steps, overall="大体校准" if all(v == "大体校准" for v in steps.values()) else "未校准",
                failing_h=[h for h, v in steps.items() if v != "大体校准"])


def criterion_b_step(recall_c, recall_p, precision_c):
    if not (recall_c > recall_p):
        return "无增益"
    if precision_c is not None and not np.isnan(precision_c) and precision_c >= PRECISION_MIN:
        return "有用的增益"
    return "以精确率为代价提高召回"


def criterion_b(per_h):
    """per_h: {h: (recall_c, recall_p, precision_c)}（h=20 与 30）。两个步长标记相同则取该标记，否则「混合（逐步长报告）」。"""
    steps = {int(h): criterion_b_step(*v) for h, v in per_h.items()}
    labels = set(steps.values())
    return dict(per_h=steps, overall=labels.pop() if len(labels) == 1 else "混合（逐步长报告）")


# ----------------------------------------------------------------------------- 块 bootstrap（与直接重抽等价）
def _ratio(a, b):
    return float(a) / float(b) if b else float("nan")


def block_counts(y, lo, y0, res, t, thr, block_ids):
    """按块（年份或气象站）分别算一次事件段计数：(块标签, (K, 4) 计数 [事件段, 命中, 告警段, 与事件重叠的告警段])。"""
    labels = np.unique(block_ids)
    rows = []
    for b in labels:
        s = block_ids == b
        e = ev.new_exceedance_metrics(y[s], lo[s], y0[s], res[s], t[s], thr)
        rows.append([e[k] for k in COUNT_COLS])
    return labels, np.array(rows, dtype=np.int64)


def bootstrap_quantity(counts_c, counts_p, labels, scheme, kind, n_boot=None, seed=None, ci=None):
    """用「每个块一行」的伪样本调用 ev.bootstrap_by_year / bootstrap_by_station。
    kind: recall_c、precision_c、recall_p、precision_p、diff_recall（保形 − 点预测）、diff_precision。"""
    n_boot, seed, ci = N_BOOT if n_boot is None else n_boot, BOOT_SEED if seed is None else seed, CI if ci is None else ci

    def stat(idx, rep):
        c, p = counts_c[idx].sum(axis=0), counts_p[idx].sum(axis=0)
        rc, pc, rp, pp = _ratio(c[1], c[0]), _ratio(c[3], c[2]), _ratio(p[1], p[0]), _ratio(p[3], p[2])
        return dict(recall_c=rc, precision_c=pc, recall_p=rp, precision_p=pp, diff_recall=rc - rp, diff_precision=pc - pp)[kind]
    fn = ev.bootstrap_by_year if scheme == "year" else ev.bootstrap_by_station
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # 全 NaN 的切片
        out = fn(stat, labels, n_boot=n_boot, seed=seed, ci=ci)
    out["n_nan"] = int(np.isnan(out["boots"]).sum())
    return out


# ----------------------------------------------------------------------------- 载入数据（测试预测只在阶段 B 读取）
def load_split_arrays(ds, split, split_name):
    """已保存的 v2 HGB 预测与样本键对齐。split ∈ {'val', 'test'}；split_name 是含该段的划分名
    （阶段 A 用只有 train/val 的 C0，阶段 B 用 main）。文件只在这里打开。"""
    stations = data.station_ids(ds)
    out = {}
    for h in HS:
        mask = data.make_split(ds, split_name, h).masks[split]
        t, res, y0, y = ev.samples_from_mask(ds.rvow, mask, h)
        with np.load(PRED / f"main_{split}_h{h}.npz") as f:
            z = {k: f[k] for k in f.files}
        if not (np.array_equal(z["t"], t) and np.array_equal(z["res"], res)):
            raise AbortRun(f"main_{split}_h{h}.npz 的（起点, 水库）与 make_split 不一致")
        out[h] = dict(t=t, res=res, y0=y0, y=y, p=z["pooled_hgb"], years=ds.years[t], bidx=bucket_index(y0), stations=stations[res])
    return out


def check_against_baseline(arrays, segment, n_res):
    """已保存预测的 HGB 汇总指标必须与已提交的 baseline_v2.csv 一致（相对差 <= 1e-5）。"""
    ref = pd.read_csv(BASELINE_CSV)
    for h in HS:
        a = arrays[h]
        row = ref[(ref.split == "main") & (ref.segment == segment) & (ref.h == h) & (ref.method == "pooled_hgb")
                  & (ref.group == "all") & (ref.threshold == 30.0)]
        if len(row) != 1:
            raise AbortRun(f"baseline_v2.csv 中找不到唯一的 {segment}/h{h}/pooled_hgb/all 行")
        reg = ev.regression_metrics(a["y"], a["p"], a["res"], n_res)
        if abs(reg["rmse"] - row.iloc[0].rmse) / abs(row.iloc[0].rmse) > 1e-5 or abs(reg["nse_median"] - row.iloc[0].nse_median) > 1e-5:
            raise AbortRun(f"{segment} h={h}: 已保存预测的 HGB 指标与 baseline_v2.csv 不符")


def zero_control(arrays, segment):
    """q_lo = q_hi = 0：区间退化为点预测，告警必须逐项还原 baseline_v2.csv 里 v2 已有的事件段指标；不通过抛 AbortRun。"""
    ref = pd.read_csv(BASELINE_CSV)
    rows = []
    for h in HS:
        a = arrays[h]
        for thr in THRESHOLDS:
            e = event_row(a["y"], a["p"], a["y0"], a["res"], a["t"], thr)
            r = ref[(ref.split == "main") & (ref.segment == segment) & (ref.h == h) & (ref.method == "pooled_hgb")
                    & (ref.group == "all") & (ref.threshold == thr)]
            if len(r) != 1:
                raise AbortRun(f"baseline_v2.csv 中找不到唯一的 {segment}/h{h}/{thr}% 行")
            r = r.iloc[0]
            same = lambda a_, b_: (np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) <= 1e-12
            ok = (all(int(e[k]) == int(r[k]) for k in COUNT_COLS) and same(e["precision_seg"], r.precision_seg)
                  and same(e["recall_seg"], r.recall_seg))
            rows.append(dict(segment=segment, h=h, threshold=thr, passed=bool(ok), **{k: int(e[k]) for k in COUNT_COLS},
                             **{f"{k}_baseline": int(r[k]) for k in COUNT_COLS}))
    zc = pd.DataFrame(rows)
    if not zc.passed.all():
        raise AbortRun(f"零对照未通过（{segment}）：分位数取 0 时告警没有还原 v2 已有的事件段指标；已停止")
    return zc


# ----------------------------------------------------------------------------- 表
def coverage_table(split, arrays, cal):
    rows = []
    for variant in VARIANTS:
        for c in LEVELS:
            for h in HS:
                a = arrays[h]
                lo, hi = make_bounds(variant, cal[str(h)], c, a["p"], a["bidx"])
                base = dict(split=split, variant=variant, level=c, h=h)
                rows.append(dict(base, by="overall", key="all", **coverage_stats(a["y"], lo, hi)))
                for i, b in enumerate(BUCKETS):
                    s = a["bidx"] == i
                    rows.append(dict(base, by="bucket", key=b, **coverage_stats(a["y"][s], lo[s], hi[s])))
                for yr in np.unique(a["years"]):
                    s = a["years"] == yr
                    rows.append(dict(base, by="year", key=str(int(yr)), **coverage_stats(a["y"][s], lo[s], hi[s])))
    return pd.DataFrame(rows)


def event_table(split, arrays, cal):
    rows = []
    for h in HS:
        a = arrays[h]
        bounds = {(v, c): make_bounds(v, cal[str(h)], c, a["p"], a["bidx"])[0] for v in VARIANTS for c in LEVELS}
        for thr in THRESHOLDS:
            rows.append(dict(split=split, variant="point", level=np.nan, h=h, threshold=thr,
                             **event_row(a["y"], a["p"], a["y0"], a["res"], a["t"], thr)))
            for (v, c), lo in bounds.items():
                rows.append(dict(split=split, variant=v, level=c, h=h, threshold=thr,
                                 **event_row(a["y"], lo, a["y0"], a["res"], a["t"], thr)))
    return pd.DataFrame(rows)


BOOT_KINDS = ("recall_c", "precision_c", "recall_p", "precision_p", "diff_recall", "diff_precision")


def _bootstrap_rows(a, cal_h, variant, level, thresholds):
    """一个（变体，档位，h）的召回、精确率及配对差的 bootstrap 行；同时记录块内求和与全样本的事件段数（被年份边界切开的段数）。"""
    rows = []
    lo = make_bounds(variant, cal_h, level, a["p"], a["bidx"])[0]
    for thr in thresholds:
        full_c = event_row(a["y"], lo, a["y0"], a["res"], a["t"], thr)
        full_p = event_row(a["y"], a["p"], a["y0"], a["res"], a["t"], thr)
        for scheme, blocks in (("year", a["years"]), ("station", a["stations"])):
            labels, cc = block_counts(a["y"], lo, a["y0"], a["res"], a["t"], thr, blocks)
            _, cp = block_counts(a["y"], a["p"], a["y0"], a["res"], a["t"], thr, blocks)
            for kind in BOOT_KINDS:
                b = bootstrap_quantity(cc, cp, labels, scheme, kind)
                full = dict(recall_c=full_c["recall_seg"], precision_c=full_c["precision_seg"], recall_p=full_p["recall_seg"],
                            precision_p=full_p["precision_seg"], diff_recall=full_c["recall_seg"] - full_p["recall_seg"],
                            diff_precision=full_c["precision_seg"] - full_p["precision_seg"])[kind]
                rows.append(dict(scheme=scheme, threshold=thr, quantity=kind, estimate_full=full, estimate_block_sum=b["estimate"],
                                 boot_mean=b["boot_mean"], ci_low=b["ci_low"], ci_high=b["ci_high"], n_blocks=b["n_blocks"],
                                 n_nan_draws=b["n_nan"], events_seg_full=full_c["events_seg"], events_seg_block_sum=int(cc[:, 0].sum()),
                                 n_boot=N_BOOT, seed=BOOT_SEED))
    return rows


def bootstrap_table(split, arrays, cal):
    """主设定（90%）与点预测：召回、精确率及配对差的 bootstrap 区间。"""
    rows = []
    for h in HS:
        for variant in VARIANTS:
            for r in _bootstrap_rows(arrays[h], cal[str(h)], variant, MAIN_LEVEL, THRESHOLDS):
                rows.append(dict(split=split, variant=variant, h=h, **r))
    return pd.DataFrame(rows)


def operating_point_tables(split, arrays, cal, cov, events, points):
    """每个（变体，h）的 c* 操作点（30% 阈值）：召回、精确率、F1（对照点预测与验证集上的值）、覆盖率诊断及其 bootstrap 区间。
    只取冻结档位，不评估其他配置。"""
    rows, boot_rows = [], []
    for variant in VARIANTS:
        for h in HS:
            c = int(points[variant][str(h)]["c_star"])
            e = events[(events.split == split) & (events.h == h) & (events.threshold == MAIN_THRESHOLD)]
            ec = e[(e.variant == variant) & (e.level == c)].iloc[0]
            ep = e[e.variant == "point"].iloc[0]
            cv = cov[(cov.split == split) & (cov.variant == variant) & (cov.level == c) & (cov.h == h) & (cov.by == "overall")].iloc[0]
            val_pt = points[variant][str(h)]["levels"][str(c)]
            rows.append(dict(split=split, variant=variant, h=h, c_star=c, threshold=MAIN_THRESHOLD,
                             events_seg=int(ec.events_seg), alerts_seg=int(ec.alerts_seg), recall_seg=ec.recall_seg, precision_seg=ec.precision_seg,
                             f1=f1_score(ec.precision_seg, ec.recall_seg), point_recall_seg=ep.recall_seg, point_precision_seg=ep.precision_seg,
                             point_f1=f1_score(ep.precision_seg, ep.recall_seg), val_recall_seg=val_pt["recall_seg"],
                             val_precision_seg=val_pt["precision_seg"], val_f1=val_pt["f1"], coverage=cv.coverage, nominal_coverage=c / 100,
                             lower_miss=cv.lower_miss, nominal_lower_miss=(100 - c) / 200, n_events_lt_30=bool(ec.n_events_lt_30)))
            for r in _bootstrap_rows(arrays[h], cal[str(h)], variant, c, (MAIN_THRESHOLD,)):
                boot_rows.append(dict(split=split, variant=variant, h=h, c_star=c, **r))
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def calibration_by_year(arrays, cal):
    """验证集按年（2011–2015）各自算一遍分位数，与全体分位数并排。只描述，不影响冻结的分位数。"""
    rows = []
    for h in HS:
        a = arrays[h]
        r = a["y"].astype(np.float64) - a["p"].astype(np.float64)
        for yr in np.unique(a["years"]):
            s = a["years"] == yr
            for c in LEVELS:
                q = conformal_quantiles(r[s], c)
                full = cal[str(h)]["marginal"][str(c)]
                rows.append(dict(variant="marginal", h=h, level=c, bucket="all", year=int(yr), n=q["n"], q_lo=q["q_lo"], q_hi=q["q_hi"],
                                 full_q_lo=full["q_lo"], full_q_hi=full["q_hi"], diff_lo=q["q_lo"] - full["q_lo"], diff_hi=q["q_hi"] - full["q_hi"]))
                for i, b in enumerate(BUCKETS):
                    sb = s & (a["bidx"] == i)
                    if not sb.any():
                        continue
                    q = conformal_quantiles(r[sb], c)
                    full = cal[str(h)]["M"][b][str(c)]
                    rows.append(dict(variant="M", h=h, level=c, bucket=b, year=int(yr), n=q["n"], q_lo=q["q_lo"], q_hi=q["q_hi"],
                                     full_q_lo=full["q_lo"], full_q_hi=full["q_hi"], diff_lo=q["q_lo"] - full["q_lo"], diff_hi=q["q_hi"] - full["q_hi"]))
    return pd.DataFrame(rows)


def criteria_table(split, cov, events, boots=None):
    """判据 (a)（测试集）与 (b) 的标记；validation 上的标记是样本内的，只描述。"""
    rows = []
    for variant in VARIANTS:
        c90 = cov[(cov.split == split) & (cov.variant == variant) & (cov.level == MAIN_LEVEL) & (cov.by == "overall")]
        per_h = {int(r.h): (r.coverage, r.lower_miss) for r in c90.itertuples()}
        a = criterion_a(per_h)
        for h, (c, m) in per_h.items():
            rows.append(dict(split=split, variant=variant, criterion="a", h=h, coverage=c, lower_miss=m, label=a["per_h"][h], overall=a["overall"],
                             note="样本内，只描述" if split == "val" else ""))
        per_step = {}
        for h in GAIN_STEPS:
            e = events[(events.split == split) & (events.h == h) & (events.threshold == MAIN_THRESHOLD)]
            ec = e[(e.variant == variant) & (e.level == MAIN_LEVEL)].iloc[0]
            ep = e[e.variant == "point"].iloc[0]
            per_step[h] = (ec.recall_seg, ep.recall_seg, ec.precision_seg)
        b = criterion_b(per_step)
        for h, (rc, rp, pc) in per_step.items():
            rows.append(dict(split=split, variant=variant, criterion="b", h=h, recall_conformal=rc, recall_point=rp, precision_conformal=pc,
                             label=b["per_h"][h], overall=b["overall"], note="样本内，只描述" if split == "val" else ""))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 权衡曲线
def plot_curves(events, path, split):
    """召回对精确率的小多图（行 = 阈值 30/20/40%，列 = h）。边缘 = 蓝，M = 橙，点预测 = 墨色菱形；
    形状不同作为第二编码；图例始终存在；只在两端标注档位；无告警（精确率无定义）的档位不画点。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = dict(marginal="#2a78d6", M="#eb6834")
    markers = dict(marginal="o", M="s")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(THRESHOLDS), len(HS), figsize=(11.5, 10.2), facecolor=surface, sharex=True, sharey=True)
    for i, thr in enumerate(THRESHOLDS):
        for j, h in enumerate(HS):
            ax = axes[i, j]
            ax.set_facecolor(surface)
            e = events[(events.split == split) & (events.h == h) & (events.threshold == thr)]
            for v in VARIANTS:
                d = e[e.variant == v].sort_values("level")
                d = d[d.precision_seg.notna()]
                ax.plot(d.precision_seg, d.recall_seg, color=colors[v], lw=1.5, marker=markers[v], ms=7, mfc=colors[v], mec=surface, mew=1, zorder=3)
                if v == "marginal" and len(d):
                    for r in (d.iloc[0], d.iloc[-1]):
                        ax.annotate(f"{int(r.level)}%", (r.precision_seg, r.recall_seg), textcoords="offset points", xytext=(6, 5), fontsize=8, color=ink2)
            pt = e[e.variant == "point"].iloc[0]
            if not np.isnan(pt.precision_seg):
                ax.plot([pt.precision_seg], [pt.recall_seg], marker="D", ms=8, mfc=ink, mec=surface, mew=1, ls="none", zorder=4)
            ax.set_title(f"阈值 {int(thr)}%，h={h}（{int(e[e.variant == 'point'].iloc[0].events_seg)} 个事件段）", fontsize=9.5, color=ink2, loc="left")
            ax.grid(color=grid, lw=0.6)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(axis)
            ax.tick_params(colors=muted, labelsize=8)
            if i == len(THRESHOLDS) - 1:
                ax.set_xlabel("精确率（事件段）", fontsize=9, color=ink2)
            if j == 0:
                ax.set_ylabel("召回（事件段）", fontsize=9, color=ink2)
    handles = [Line2D([0], [0], color=colors["marginal"], lw=1.5, marker="o", ms=7, label="保形下界告警：边缘校准"),
               Line2D([0], [0], color=colors["M"], lw=1.5, marker="s", ms=7, label="保形下界告警：M（按起点库容分桶）"),
               Line2D([0], [0], color=ink, marker="D", ms=8, ls="none", label="点预测告警（v2）")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.965), ncol=3, frameon=False, fontsize=9.5, labelcolor=ink2)
    fig.suptitle(f"权衡曲线（{'验证集，样本内' if split == 'val' else '测试集'}）：覆盖率档位 50%→95% 沿曲线召回上升、精确率下降", y=0.995, fontsize=11, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.005, "标注为覆盖率档位（两端）；无告警的档位不画点；数值表见对应的事件段指标 csv。经验校准的区间，不是有理论保证的区间。",
             fontsize=8, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ----------------------------------------------------------------------------- 主流程
def _write_log(path, log):
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final", action="store_true", help="阶段 B：读取冻结的分位数，读取测试预测并评估一次")
    p.add_argument("--skip-test", action="store_true", help="阶段 A：只用验证集，不读取测试预测")
    a = p.parse_args(argv)
    if a.final == a.skip_test:
        p.error("必须且只能指定 --final 或 --skip-test 之一（没有 --final 时不读取测试预测）")
    return a


def calibration_document(cal, arrays, md5s, git):
    return dict(status="proposed（阶段 A 产出；用户确认后冻结为 docs/conformal/calibration_frozen.json）",
                method=NO_GUARANTEE_SENTENCE, validation_twice=VALIDATION_TWICE_TEXT, levels_pct=list(LEVELS), buckets=list(BUCKETS),
                quantile_rule="k_lo = max(1, floor((n+1)*alpha/2))，k_hi = min(n, ceil((n+1)*(1-alpha/2)))；有符号残差 r = y - yhat；区间截断到 [0, 100]",
                n_val={str(h): int(len(arrays[h]["y"])) for h in HS},
                marginal={str(h): cal[str(h)]["marginal"] for h in HS}, M={str(h): cal[str(h)]["M"] for h in HS},
                data=md5s, git=git)


def run_stage_a(log, out):
    v2_log = json.loads(s1m.V2_LOG.read_text(encoding="utf-8"))
    s1m.assert_matches_v2_log(rb, v2_log)
    ds = data.load_dataset()
    n_res = len(ds.fac_codes)
    with s1m.registered({NAME: data.MAIN_TRAIN_YEARS}):                       # 只有 train 与 val 两段：测试样本掩码不会被构造
        arrays = load_split_arrays(ds, "val", NAME)
    check_against_baseline(arrays, "val", n_res)
    zc = zero_control(arrays, "val")
    zc.to_csv(out / "zero_control_val.csv", index=False)
    log["zero_control_val"] = json.loads(zc.to_json(orient="records"))
    print("验证集零对照通过：分位数取 0 时告警逐项还原 v2 已有的事件段指标", flush=True)

    cal = {str(h): calibrate_h(arrays[h]["y"], arrays[h]["p"], arrays[h]["bidx"]) for h in HS}
    doc = calibration_document(cal, arrays, log["data"], log["git_at_start"])
    (out / "calibration_proposed.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log["calibration"] = dict(n_val=doc["n_val"], marginal=doc["marginal"], M=doc["M"],
                              n_by_bucket={str(h): {b: cal[str(h)]["M"][b]["90"]["n"] for b in BUCKETS} for h in HS})
    _write_log(out / "conformal_log_stageA.json", log)
    cby = calibration_by_year(arrays, cal)
    cby.to_csv(out / "calibration_by_year.csv", index=False)
    cov, events, boots = coverage_table("val", arrays, cal), event_table("val", arrays, cal), bootstrap_table("val", arrays, cal)
    cov.to_csv(out / "val_coverage.csv", index=False)
    events.to_csv(out / "val_event_metrics.csv", index=False)
    boots.to_csv(out / "val_event_bootstrap.csv", index=False)
    crit = criteria_table("val", cov, events)
    crit.to_csv(out / "val_criteria_descriptive.csv", index=False)
    (out / "figures").mkdir(exist_ok=True)
    plot_curves(events, out / "figures" / "curve_val.png", "val")
    log["boot_cut_segments"] = {"year": int((boots[boots.scheme == "year"].events_seg_block_sum - boots[boots.scheme == "year"].events_seg_full).clip(lower=0).max()),
                                "station": int((boots[boots.scheme == "station"].events_seg_block_sum - boots[boots.scheme == "station"].events_seg_full).abs().max())}
    log["status"] = "completed（阶段 A）"
    log["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_log(out / "conformal_log_stageA.json", log)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    print("\n校准分位数（边缘，90%）：", {h: (cal[str(h)]["marginal"]["90"]["q_lo"], cal[str(h)]["marginal"]["90"]["q_hi"]) for h in HS})
    print("\n输出目录:", out)


def run_stage_b(log, out):
    git = rb.git_info()
    if git["dirty"] is not False:
        raise AbortRun("阶段 B 要求工作区干净（git status 为空）；已停止，不读取测试预测")
    for path in (FROZEN, OP_FILE):
        if not path.exists():
            raise AbortRun(f"找不到冻结文件 {path}；阶段 B 必须在分位数与操作点冻结并提交之后运行")
        assert_matches_committed(path)                                         # 与提交版一致
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    op_doc = json.loads(OP_FILE.read_text(encoding="utf-8"))
    log["frozen_file_md5"], log["operating_points_file_md5"] = rb.md5_of(FROZEN), rb.md5_of(OP_FILE)
    log["operating_points_frozen"] = op_doc["points"]
    v2_log = json.loads(s1m.V2_LOG.read_text(encoding="utf-8"))
    s1m.assert_matches_v2_log(rb, v2_log)
    ds = data.load_dataset()
    n_res = len(ds.fac_codes)
    val = load_split_arrays(ds, "val", "main")
    check_against_baseline(val, "val", n_res)
    cal = {str(h): calibrate_h(val[h]["y"], val[h]["p"], val[h]["bidx"]) for h in HS}
    doc = calibration_document(cal, val, log["data"], log["git_at_start"])
    assert_matches_frozen(doc, frozen)                                         # (a) 分位数逐位相同，不相同就停止，不读取测试预测
    assert_operating_points_match(select_operating_points(event_table("val", val, cal)), op_doc)      # (b) 操作点与冻结文件一致
    log["frozen_check"] = "重新计算的验证集分位数与冻结文件完全相同；操作点与冻结文件一致；两个冻结文件与提交版一致"
    log["final"] = dict(eval_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **{f"git_{k}": v for k, v in git.items()})
    _write_log(out / "conformal_log_stageB.json", log)
    print("冻结校验通过；读取测试预测（只读取一次），先做测试集零对照", flush=True)
    test = load_split_arrays(ds, "test", "main")                        # 唯一读取测试预测的地方
    check_against_baseline(test, "test", n_res)
    zc = zero_control(test, "test")                                     # 不通过抛 AbortRun：此前没有做任何保形评估
    zc.to_csv(out / "zero_control_test.csv", index=False)
    log["zero_control_test"] = json.loads(zc.to_json(orient="records"))
    _write_log(out / "conformal_log_stageB.json", log)
    print("测试集零对照通过；开始评估冻结的配置", flush=True)
    cov, events, boots = coverage_table("test", test, cal), event_table("test", test, cal), bootstrap_table("test", test, cal)
    cov.to_csv(out / "test_coverage.csv", index=False)
    events.to_csv(out / "test_event_metrics.csv", index=False)
    boots.to_csv(out / "test_event_bootstrap.csv", index=False)
    ops, ops_boot = operating_point_tables("test", test, cal, cov, events, op_doc["points"])
    ops.to_csv(out / "test_operating_points.csv", index=False)
    ops_boot.to_csv(out / "test_operating_points_bootstrap.csv", index=False)
    crit = criteria_table("test", cov, events)
    crit.to_csv(out / "criteria_final.csv", index=False)
    (out / "figures").mkdir(exist_ok=True)
    plot_curves(events, out / "figures" / "curve_test.png", "test")
    log["criteria_results"] = json.loads(crit.to_json(orient="records"))
    log["operating_points_test"] = json.loads(ops.to_json(orient="records"))
    log["status"] = "completed（阶段 B）"
    log["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_log(out / "conformal_log_stageB.json", log)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    print("\n判据:\n" + crit.drop(columns=["note"]).to_string(index=False))
    print("\n指定操作点（测试集，30%）:\n" + ops.drop(columns=["split", "threshold"]).to_string(index=False))
    print("\n输出目录:", out)


def main(argv=None):
    args = parse_args(argv)
    stage = "B" if args.final else "A"
    OUT.mkdir(parents=True, exist_ok=True)
    log = dict(status="running", stage=stage, preregistered=True, declaration=DECLARATION, method_note=NO_GUARANTEE_SENTENCE,
               validation_twice=VALIDATION_TWICE_TEXT, criteria=CRITERIA_TEXT, operating_point_rule=OPERATING_POINT_TEXT,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               config=dict(levels_pct=list(LEVELS), main_level_pct=MAIN_LEVEL, thresholds=list(THRESHOLDS), main_threshold=MAIN_THRESHOLD,
                           variants=list(VARIANTS), buckets=list(BUCKETS), horizons=list(HS), n_boot=N_BOOT, boot_seed=BOOT_SEED, ci=CI,
                           gain_steps=list(GAIN_STEPS), precision_min=PRECISION_MIN, coverage_tol=COVERAGE_TOL, lower_miss_tol=LOWER_MISS_TOL,
                           nominal_lower_miss=NOMINAL_LOWER, test_predictions_read=(stage == "B"), no_other_variants=True),
               data=rb.data_md5s(), git_at_start=rb.git_info(), environment=rb.environment_info())
    path = OUT / f"conformal_log_stage{stage}.json"
    _write_log(path, log)                                       # 规则、判据与声明在任何结果之前落盘
    try:
        (run_stage_b if stage == "B" else run_stage_a)(log, OUT)
    except AbortRun as e:
        log["status"], log["abort_reason"] = "aborted", str(e)
        _write_log(path, log)
        sys.exit(f"停止: {e}")
    except Exception:
        log["status"], log["abort_reason"] = "error", traceback.format_exc()
        _write_log(path, log)
        raise


if __name__ == "__main__":
    main()
