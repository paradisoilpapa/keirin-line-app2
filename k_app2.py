# -*- coding: utf-8 -*-
# 政春さんの「印だけ」EVバランス帯チェッカー（軽量版）
import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="政春版：印→EVバランス帯チェッカー", layout="centered")

# ===== 固定パラメータ =====
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},
}
FALLBACK = "α"   # 未指定の車に与える印

# EV判定帯（オッズ=必要オッズ×(1+E_MIN〜E_MAX)）
E_MIN, E_MAX = 0.10, 0.60

# Pフロア（この想定的中率未満は対象外）
P_FLOOR = {"wide":0.25, "nifuku":0.12, "nitan":0.07, "sanpuku":0.06, "santan":0.03}

def _sort_key_nums(name: str):
    return list(map(int, re.findall(r"\d+", str(name))))

st.title("🎯 政春版：印→EVバランス帯（ワンタッチ）")

# ===== 入力：頭数・印 =====
cols_top = st.columns(3)
with cols_top[0]:
    n_cars = st.selectbox("出走数", [5,6,7,8,9], index=2)
with cols_top[1]:
    trials = st.slider("シミュレーション回数", 1000, 20000, 8000, 1000)
with cols_top[2]:
    seed = st.number_input("乱数シード", value=20250904, step=1)

st.markdown("#### 印の割り当て（各記号 = 1台）")
st.caption("未入力の車は自動で **α/β** を使い確率校正します（期待値漏れを防止）。")

marks = ["◎","〇","▲","△","×","α","β"]
sel = {}
c1, c2, c3, c4 = st.columns(4)
with c1:
    sel["◎"] = st.number_input("◎", 1, n_cars, 1, key="m_wheel")
    sel["〇"] = st.number_input("〇", 1, n_cars, min(2, n_cars), key="m_maru")
with c2:
    sel["▲"] = st.number_input("▲", 1, n_cars, min(3, n_cars), key="m_san")
    sel["△"] = st.number_input("△", 1, n_cars, min(4, n_cars), key="m_shita")
with c3:
    sel["×"] = st.number_input("×", 1, n_cars, min(5, n_cars), key="m_batsu")
    sel["α"] = st.number_input("α", 1, n_cars, min(6, n_cars), key="m_alpha")
with c4:
    sel["β"] = st.number_input("β", 1, n_cars, min(7, n_cars), key="m_beta")

# 同じ車に複数記号が付いたら後勝ちで上書き（◎を先にチェック→最後にβ…の順）
order_apply = ["β","α","×","△","▲","〇","◎"]  # 弱→強の順で最終上書き＝強印優先
mark_by_car = {i: None for i in range(1, n_cars+1)}
for mk in order_apply:
    car = sel.get(mk, None)
    if car is not None and 1 <= int(car) <= n_cars:
        mark_by_car[int(car)] = mk

# フォールバック印補完（未指定はαに寄せる）
for i in range(1, n_cars+1):
    if mark_by_car[i] is None:
        mark_by_car[i] = FALLBACK

st.caption("割り当て結果：" + "  ".join([f"{i}番:{mark_by_car[i]}" for i in range(1, n_cars+1)]))

# ===== baseを均等→印で確率校正（p1, pTop2, pTop3 別々に正規化） =====
base = np.ones(n_cars, dtype=float) / n_cars

def calibrate(base_vec: np.ndarray, key: str) -> np.ndarray:
    m = np.ones(n_cars, dtype=float)
    for idx, car in enumerate(range(1, n_cars+1)):
        mk = mark_by_car[car]
        tgt = float(RANK_STATS.get(mk, RANK_STATS[FALLBACK])[key])
        # ratioの平方根で過補正を緩和
        ratio = tgt / max(float(base_vec[idx]), 1e-9)
        m[idx] = float(np.clip(ratio**0.5, 0.25, 2.5))
    pv = base_vec * m
    pv = pv / pv.sum()
    return pv

p3 = calibrate(base, "pTop3")
p2 = calibrate(base, "pTop2")
p1 = calibrate(base, "p1")

rng = np.random.default_rng(int(seed))

def sample_order(pvec: np.ndarray) -> list[int]:
    # Gumbel-max trick for PL近似
    g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
    score = np.log(pvec+1e-12) + g
    return (np.argsort(-score)+1).tolist()

# ◎から固定
one = sel["◎"]
mates = [sel["〇"], sel["▲"]]  # 三連複C, 三連単の2着候補
others = [i for i in range(1, n_cars+1) if i != one]

# カウント器
wide_ct = {k:0 for k in others}
qn_ct   = {k:0 for k in others}
ex_ct   = {k:0 for k in others}
triC_ct = {}       # (a,b,◎) a<b, {a,b} ∩ {〇,▲} ≠ ∅
st3_ct  = {}       # (sec, thr) with sec in {〇,▲}

triC_list = []
if any(mates):
    for a in others:
        for b in others:
            if a>=b: continue
            if (a in mates) or (b in mates):
                triC_list.append(tuple(sorted([a,b,one])))
    triC_list = sorted(set(triC_list))

# ===== シミュレーション =====
for _ in range(trials):
    # Top3系
    ord3 = sample_order(p3)
    top3 = set(ord3[:3])
    if one in top3:
        for k in wide_ct.keys():
            if k in top3:
                wide_ct[k] += 1
        if len(triC_list)>0:
            others3 = list(top3 - {one})
            if len(others3)==2:
                a, b = sorted(others3)
                t = tuple(sorted([a,b,one]))
                if t in triC_list:
                    triC_ct[t] = triC_ct.get(t,0)+1

    # Top2系
    ord2 = sample_order(p2)
    top2 = set(ord2[:2])
    if one in top2:
        for k in qn_ct.keys():
            if k in top2:
                qn_ct[k]+=1

    # 1着系
    ord1 = sample_order(p1)
    if ord1[0]==one:
        k2 = ord1[1]
        if k2 in ex_ct:
            ex_ct[k2]+=1
        if k2 in mates and len(ord1)>=3:
            k3 = ord1[2]
            if k3 not in (one, k2):
                st3_ct[(k2,k3)] = st3_ct.get((k2,k3),0)+1

def band_from_cnt(cnt:int) -> str:
    if cnt<=0: return "-"
    p = cnt/float(trials)
    need = 1.0/p
    low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
    return f"{low:.1f}〜{high:.1f}倍"

def need_from_cnt(cnt:int):
    if cnt<=0: return "-"
    p = cnt/float(trials)
    return round(1.0/p, 2)

st.divider()
st.markdown("### 出力（◎固定 / EV=バランス帯）")

# === ワイド（◎-全）固定表示 ===
rows=[]
for k in sorted(others):
    cnt = wide_ct[k]; p = cnt/float(trials)
    need = need_from_cnt(cnt)
    rows.append({
        "買い目": f"{one}-{k}",
        "p(想定)": round(p,4),
        "必要オッズ(=1/p)": need,
        "判定": "買い" if p>=P_FLOOR["wide"] and need!="-" else "見送り"
    })
df_wide = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s:s.map(_sort_key_nums)).reset_index(drop=True)
st.markdown("#### ワイド（◎-全）")
st.dataframe(df_wide, use_container_width=True)

# === 二車複（◎-全）固定表示 ===
rows=[]
for k in sorted(others):
    cnt = qn_ct[k]; p = cnt/float(trials)
    rows.append({
        "買い目": f"{one}-{k}",
        "p(想定)": round(p,4),
        "バランス帯": band_from_cnt(cnt),
        "判定": "買い" if p>=P_FLOOR["nifuku"] else "見送り"
    })
df_qn = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s:s.map(_sort_key_nums)).reset_index(drop=True)
st.markdown("#### 二車複（◎-全）")
st.dataframe(df_qn, use_container_width=True)

# === 二車単（◎→全）固定表示 ===
rows=[]
for k in sorted(others):
    cnt = ex_ct[k]; p = cnt/float(trials)
    rows.append({
        "買い目": f"{one}->{k}",
        "p(想定)": round(p,4),
        "バランス帯": band_from_cnt(cnt),
        "判定": "買い" if p>=P_FLOOR["nitan"] else "見送り"
    })
df_ex = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s:s.map(_sort_key_nums)).reset_index(drop=True)
st.markdown("#### 二車単（◎→全）")
st.dataframe(df_ex, use_container_width=True)

# === 三連複C（◎-[相手]-全）===
st.markdown("#### 三連複C（◎-[相手(〇/▲)]-全）")
if len(triC_list)>0:
    rows=[]
    for t in triC_list:
        cnt = triC_ct.get(t,0); p = cnt/float(trials)
        rows.append({
            "買い目": f"{t[0]}-{t[1]}-{t[2]}",
            "p(想定)": round(p,5),
            "バランス帯": band_from_cnt(cnt),
            "判定": "買い" if p>=P_FLOOR["sanpuku"] else "見送り"
        })
    df_triC = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s:s.map(_sort_key_nums)).reset_index(drop=True)
    st.dataframe(df_triC, use_container_width=True)
else:
    st.info("相手（〇/▲）のいずれかが未設定で三連複Cは非表示")

# === 三連単（◎→[相手(〇/▲)]→全）===
st.markdown("#### 三連単（◎→[相手(〇/▲)]→全）")
rows=[]
for (sec, thr), cnt in st3_ct.items():
    p = cnt/float(trials)
    rows.append({
        "買い目": f"{one}->{sec}->{thr}",
        "p(想定)": round(p,5),
        "バランス帯": band_from_cnt(cnt),
        "判定": "買い" if p>=P_FLOOR["santan"] else "見送り"
    })
if rows:
    df_st = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s:s.map(_sort_key_nums)).reset_index(drop=True)
    st.dataframe(df_st, use_container_width=True)
else:
    st.info("該当なし")

st.divider()
st.markdown("### 🔖 note貼り付け用（バランス帯）")
def to_lines(df: pd.DataFrame, title: str):
    if df is None or len(df)==0: return f"{title}\n対象外"
    out=[]
    for _,r in df.iterrows():
        name = str(r["買い目"])
        if "バランス帯" in r and isinstance(r["バランス帯"], str) and r["バランス帯"]!="-":
            out.append(f"{name}：{r['バランス帯']}")
        elif "必要オッズ(=1/p)" in r and r["必要オッズ(=1/p)"]!="-":
            out.append(f"{name}：{r['必要オッズ(=1/p)']}倍以上")
    out = sorted(out, key=_sort_key_nums)
    return f"{title}\n" + "\n".join(out) if out else f"{title}\n対象外"

txt = []
txt.append(to_lines(df_qn,  "二車複（◎-全）"))
txt.append(to_lines(df_ex,  "二車単（◎→全）"))
if 'df_triC' in locals():
    txt.append(to_lines(df_triC, "三連複C（◎-[相手]-全）"))
txt.append(to_lines(df_wide, "ワイド（◎-全）"))
if 'df_st' in locals():
    txt.append(to_lines(df_st,  "三連単（◎→[相手]→全）"))

note_text = "\n\n".join(txt) + "\n\n" + \
    "※このオッズ**以下**は期待値未満を想定しています。また、この帯から極端な高オッズに離れるほど、的中率とのバランスが崩れハイリスクになります。"

st.text_area("コピー用テキスト", note_text, height=360)


