# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import re, unicodedata

st.set_page_config(page_title="印だけでEV判定（◎◯▲△×αβ→買える帯）", layout="wide")

# ==============================
# 定数
# ==============================
MARKS = ['◎','〇','▲','△','×','α','β']           # 印の優先リスト
NUM_TO_MARK = {str(i+1): m for i, m in enumerate(MARKS)}
ALIAS = {'○': '〇', '◯': '〇', 'O': '〇', 'o': '〇'} # ゆらぎ吸収

# 実測カリブレーション（印→想定的中率）
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},
}

# EVルール
P_FLOOR_BASE = {"wide": 0.25, "nifuku": 0.12, "nitan": 0.07, "sanpuku": 0.06, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60  # 期待値レンジ（=必要オッズ×(1+E_MIN..MAX)）

# ==============================
# ユーティリティ
# ==============================
def normalize_mark(val: str | int | None):
    """数字->印、ゆらぎ吸収、未知はNone"""
    if val is None: return None
    s = unicodedata.normalize("NFKC", str(val)).strip()
    if not s: return None
    if s.isdigit() and s in NUM_TO_MARK:
        return NUM_TO_MARK[s]
    s = ALIAS.get(s, s)
    return s if s in MARKS else None

def _sort_key_by_numbers(name: str) -> list[int]:
    return list(map(int, re.findall(r"\d+", str(name))))

def calibrate_probs(base_vec: np.ndarray, marks_by_car: dict[int, str], stat_key: str, expo: float):
    """印→目標率で各車の重みを補正して正規化"""
    m = np.ones_like(base_vec, dtype=float)
    for idx, car in enumerate(range(1, len(base_vec)+1)):
        mk = marks_by_car.get(car, 'β')  # 未指定はβ扱い
        target = float(RANK_STATS.get(mk, RANK_STATS['β'])[stat_key])
        ratio = target / max(float(base_vec[idx]), 1e-9)
        m[idx] = np.clip(ratio**(0.5*expo), 0.25, 2.5)
    p = base_vec * m
    p = p / p.sum()
    return p

def sample_order_from_probs(rng: np.random.Generator, pvec: np.ndarray) -> list[int]:
    """Gumbel-maxで順序サンプル"""
    g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
    score = np.log(pvec+1e-12) + g
    return (np.argsort(-score)+1).tolist()

def buy_band_from_p(p: float):
    """必要オッズ=1/p と EV帯（下限〜上限）"""
    need = 1.0 / max(p, 1e-12)
    return need, need*(1.0+E_MIN), need*(1.0+E_MAX)

# ==============================
# 入力
# ==============================
st.title("印だけでEV判定（◎〇▲△×αβ → 買える帯）")

c1, c2, c3 = st.columns([2,2,6])
with c1:
    n_cars = st.selectbox("出走数", [5,6,7,8,9], index=2)
with c2:
    trials = st.slider("シミュレーション試行回数", 1000, 20000, 8000, 1000)
with c3:
    expo = st.slider("温度（0.7=固め〜1.3=荒め）", 0.7, 1.3, 1.0, 0.05)

st.markdown("#### 各車の印を入力（数字1〜7でも、記号でもOK）")
cols = st.columns(n_cars)
marks_input = {}
for i in range(n_cars):
    with cols[i]:
        v = st.text_input(f"{i+1}番", value="", max_chars=2)
        mk = normalize_mark(v)
        marks_input[i+1] = mk

# 検証
vals = list(marks_input.values())
if '◎' not in vals:
    st.info("※『◎』が未入力です。『◎』を1頭だけ指定してください。")
    st.stop()
if vals.count('◎') != 1:
    st.warning("※『◎』は1頭のみ指定してください。")
    st.stop()
if not all((m in MARKS) for m in vals if m is not None):
    st.warning("※ 入力に不正な印があります。1〜7の数字、または ◎〇▲△×αβ（○/◯も可）。")
    st.stop()

# ==============================
# 計算（印→確率→EV）
# ==============================
# ベース＝均等（印だけ前提なので）
base = np.ones(n_cars, dtype=float) / n_cars

# 印カリブレーション
p3 = calibrate_probs(base, marks_input, "pTop3", expo)
p2 = calibrate_probs(base, marks_input, "pTop2", expo)
p1 = calibrate_probs(base, marks_input, "p1",    expo)

rng = np.random.default_rng(20250906)

anchor = [k for k,v in marks_input.items() if v=='◎'][0]
mates  = [k for k,v in marks_input.items() if v in ('〇','▲')]
others = [i for i in range(1, n_cars+1) if i != anchor]

# カウント
wide_counts = {k:0 for k in others}
qn_counts   = {k:0 for k in others}
ex_counts   = {k:0 for k in others}
trioC_counts = {}
st3_counts  = {}

# 三連複C候補（◎-相手-全）…相手は〇or▲が絡む形を優先
trioC_list = []
if mates:
    for a in others:
        for b in others:
            if a >= b:       continue
            if (a in mates) or (b in mates):
                t = tuple(sorted([anchor, a, b]))
                trioC_list.append(t)
    trioC_list = sorted(set(trioC_list))

for _ in range(trials):
    # Top3
    ord3 = sample_order_from_probs(rng, p3)
    top3 = set(ord3[:3])
    if anchor in top3:
        for k in wide_counts.keys():
            if k in top3:
                wide_counts[k] += 1
        if trioC_list:
            oth = list(top3 - {anchor})
            if len(oth)==2:
                a,b = sorted(oth)
                t = tuple(sorted([anchor, a, b]))
                if t in trioC_list:
                    trioC_counts[t] = trioC_counts.get(t, 0) + 1

    # Top2（二車複）
    ord2 = sample_order_from_probs(rng, p2)
    if anchor in set(ord2[:2]):
        for k in qn_counts.keys():
            if k in set(ord2[:2]):
                qn_counts[k] += 1

    # 1着（二車単 / 三連単）
    ord1 = sample_order_from_probs(rng, p1)
    if ord1[0] == anchor:
        k2 = ord1[1]
        if k2 in ex_counts:
            ex_counts[k2] += 1
        if mates:
            k3 = ord1[2]
            if (k2 in mates) and (k3 not in (anchor, k2)):
                st3_counts[(k2, k3)] = st3_counts.get((k2, k3), 0) + 1

# 展開評価（ざっくり）
conf = abs(float(p1[anchor-1]) - np.sort(p1)[-2])
confidence = "優位" if conf >= 0.02 else ("互角" if conf >= 0.01 else "混戦")

# Pフロア（展開で微調整）
P_FLOOR = P_FLOOR_BASE.copy()
if confidence == "優位":
    for k in P_FLOOR: P_FLOOR[k] *= 0.90
elif confidence == "混戦":
    for k in P_FLOOR: P_FLOOR[k] *= 1.10

# ==============================
# 出力
# ==============================
st.markdown(f"### 展開評価：**{confidence}**")
st.caption(f"印：{'  '.join(f'{i}番:{marks_input[i] or "(未)"}' for i in range(1, n_cars+1))}")

# 三連複C
st.markdown("#### 三連複C（◎-[相手]-全）※車番順")
rows=[]
for t in sorted(trioC_counts.keys()):
    cnt = trioC_counts[t]; p = cnt / trials
    if p < P_FLOOR["sanpuku"]: continue
    need, low, high = buy_band_from_p(p)
    rows.append({"買い目": f"{t[0]}-{t[1]}-{t[2]}", "p(想定的中率)": round(p,5), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
trioc_df = pd.DataFrame(rows)
if len(trioc_df):
    trioc_df = trioc_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
    st.dataframe(trioc_df, use_container_width=True)
else:
    st.info("対象外（Pフロア未満 or 相手条件なし）")

# ワイド
st.markdown("#### ワイド（◎-全）※車番順")
rows=[]
for k,cnt in wide_counts.items():
    p = cnt / trials
    if p < P_FLOOR["wide"]: continue
    need, _, _ = buy_band_from_p(p)
    rows.append({"買い目": f"{anchor}-{k}", "p(想定的中率)": round(p,5), "必要オッズ(=1/p)": round(need,2), "ルール":"必要オッズ以上"})
wide_df = pd.DataFrame(rows)
if len(wide_df):
    wide_df = wide_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
    st.dataframe(wide_df, use_container_width=True)
else:
    st.info("対象外")

# 二車複
st.markdown("#### 二車複（◎-全）※車番順")
rows=[]
for k,cnt in qn_counts.items():
    p = cnt / trials
    if p < P_FLOOR["nifuku"]: continue
    need, low, high = buy_band_from_p(p)
    rows.append({"買い目": f"{anchor}-{k}", "p(想定的中率)": round(p,5), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
qn_df = pd.DataFrame(rows)
if len(qn_df):
    qn_df = qn_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
    st.dataframe(qn_df, use_container_width=True)
else:
    st.info("対象外")

# 二車単
st.markdown("#### 二車単（◎→全）※車番順")
rows=[]
for k,cnt in ex_counts.items():
    p = cnt / trials
    if p < P_FLOOR["nitan"]: continue
    need, low, high = buy_band_from_p(p)
    rows.append({"買い目": f"{anchor}->{k}", "p(想定的中率)": round(p,5), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
ex_df = pd.DataFrame(rows)
if len(ex_df):
    ex_df = ex_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
    st.dataframe(ex_df, use_container_width=True)
else:
    st.info("対象外")

# 三連単（◎→[相手]→全）
st.markdown("#### 三連単（◎→[相手]→全）※車番順")
rows=[]
for (sec,thr), cnt in st3_counts.items():
    p = cnt / trials
    if p < P_FLOOR["santan"]: continue
    need, low, high = buy_band_from_p(p)
    rows.append({"買い目": f"{anchor}->{sec}->{thr}", "p(想定的中率)": round(p,5), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
st_df = pd.DataFrame(rows)
if len(st_df):
    st_df = st_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
    st.dataframe(st_df, use_container_width=True)
else:
    st.info("対象外")

st.caption("※ワイドは『必要オッズ以上で買い』。他は『買える帯』で買い。Pフロア未満はどんなオッズでも買わない。")

