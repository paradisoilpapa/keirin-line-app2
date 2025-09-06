# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="政春さん向け：印→EVバランス帯", layout="wide")

st.title("🎯 印だけ入れて期待値レンジ（バランス帯）を見る簡易版")

# ========= パラメータ =========
st.sidebar.header("設定")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)
trials = st.sidebar.slider("シミュレーション回数", 2000, 20000, 8000, 1000)
seed   = st.sidebar.number_input("乱数シード", value=20250904, step=1)

# EV 下限（=Pフロア）と バランス帯（=必要オッズ×(1+α〜1+β)）
P_FLOOR = {"wide": 0.25, "nifuku": 0.12, "nitan": 0.07, "sanpuku": 0.06, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60  # バランス帯：+10%〜+60%

st.sidebar.caption("Pフロア（最低想定的中率）: ワイド25%/二複12%/二単7%/三複6%/三単3%")
st.sidebar.caption("バランス帯: 必要オッズ×(1+10%〜1+60%)")

# ========= 印入力（一記号一数字） =========
st.subheader("印入力（1記号=1車番）")
cols = st.columns(7)
with cols[0]: car_om = st.number_input("◎", min_value=1, max_value=n_cars, value=1, step=1, key="om")  # 本命（必須）
with cols[1]: car_mr = st.number_input("〇", min_value=1, max_value=n_cars, value=min(2,n_cars), step=1, key="mr")
with cols[2]: car_an = st.number_input("▲", min_value=1, max_value=n_cars, value=min(3,n_cars), step=1, key="an")
with cols[3]: car_dt = st.number_input("△", min_value=1, max_value=n_cars, value=min(4,n_cars), step=1, key="dt")
with cols[4]: car_x  = st.number_input("×", min_value=1, max_value=n_cars, value=min(5,n_cars), step=1, key="xx")
with cols[5]: car_a  = st.number_input("α", min_value=1, max_value=n_cars, value=min(6,n_cars), step=1, key="aa")
with cols[6]: car_b  = st.number_input("β", min_value=1, max_value=n_cars, value=min(7,n_cars), step=1, key="bb")

marks_input = {"◎":int(car_om), "〇":int(car_mr), "▲":int(car_an), "△":int(car_dt),
               "×":int(car_x), "α":int(car_a), "β":int(car_b)}

# 同一車重複を許す（手入力優先）。未指定車は「無印」として扱う
mark_by_car = {i:"無" for i in range(1, n_cars+1)}
for mk, car in marks_input.items():
    if 1 <= car <= n_cars:
        mark_by_car[car] = mk

st.caption("印： " + "  ".join(f"{i}番:{mark_by_car[i]}" for i in range(1, n_cars+1)))

# ========= 印→確率倍率（連続係数） =========
# ここは“固定テンプレ”。αとβは同値、無印もβと同値にしています。
MARK_MUL = {
    # p1/p2/p3 それぞれの倍率。相対比だけを使い、最後は正規化します。
    "◎": {"p1":1.15, "p2":1.08, "p3":1.05},
    "〇": {"p1":1.05, "p2":1.04, "p3":1.03},
    "▲": {"p1":0.98, "p2":1.02, "p3":1.03},
    "△": {"p1":0.96, "p2":0.99, "p3":1.01},
    "×": {"p1":0.95, "p2":0.98, "p3":1.00},
    "α": {"p1":0.97, "p2":0.99, "p3":1.01},
    "β": {"p1":0.97, "p2":0.99, "p3":1.01},
    "無": {"p1":0.97, "p2":0.99, "p3":1.01},  # 無印＝βと同値
}

# ========= 確率ベクトルを作成 =========
def build_probs(n, which:"p1|p2|p3"):
    base = np.ones(n, dtype=float)/n  # ベースは一様
    m = np.array([MARK_MUL.get(mark_by_car[i+1], MARK_MUL["無"])[which] for i in range(n)], dtype=float)
    p = base * m
    p = p / p.sum()
    return p

probs_p1 = build_probs(n_cars, "p1")  # 1着用（→二車単/三連単）
probs_p2 = build_probs(n_cars, "p2")  # Top2用（→二車複）
probs_p3 = build_probs(n_cars, "p3")  # Top3用（→ワイド/三連複）

# ========= PL風サンプラー =========
rng = np.random.default_rng(int(seed))
def sample_order_from_probs(pvec: np.ndarray) -> list[int]:
    # Gumbel-Max trick
    g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
    score = np.log(pvec+1e-12) + g
    return (np.argsort(-score)+1).tolist()  # 1-indexed車番

# ========= カウント器セットアップ =========
om = marks_input["◎"]
all_others = [i for i in range(1, n_cars+1) if i != om]

# ワイド / 二車複 / 二車単
wide_counts = {k:0 for k in all_others}
qn_counts   = {k:0 for k in all_others}
ex_counts   = {k:0 for k in all_others}

# 三連複（◎-相手-全）
trio_counts = {}  # key=(om, a, b sorted)

# 三連単（◎→相手→全）
st3_counts = {}   # key=(om, sec, thr)

# ========= シミュレーション =========
for _ in range(trials):
    # Top3系（ワイド/三複）
    order3 = sample_order_from_probs(probs_p3)
    top3 = set(order3[:3])
    if om in top3:
        for k in all_others:
            if k in top3:
                wide_counts[k] += 1
        # 三連複：◎+{a,b}（a<b）
        others = sorted(list(top3 - {om}))
        if len(others)==2:
            a,b = others
            key = tuple(sorted([om, a, b]))
            trio_counts[key] = trio_counts.get(key, 0) + 1

    # Top2系（二車複）
    order2 = sample_order_from_probs(probs_p2)
    top2 = set(order2[:2])
    if om in top2:
        for k in all_others:
            if k in top2:
                qn_counts[k] += 1

    # 1着系（二車単/三連単）
    order1 = sample_order_from_probs(probs_p1)
    if order1[0] == om:
        sec = order1[1]
        if sec in ex_counts:
            ex_counts[sec] += 1
        thr = order1[2]
        if thr != om and thr != sec:
            key = (om, sec, thr)
            st3_counts[key] = st3_counts.get(key, 0) + 1

def _need(cnt): 
    if cnt <= 0: return None
    p = cnt / trials
    return (p, 1.0/p)

def _band_text(need):
    if need is None: return "-"
    low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
    return f"{low:.1f}〜{high:.1f}倍"

# ========= 出力（◎から固定表示／バランス帯のみ） =========
st.markdown("### 二車複（◎-全）")
rows=[]
for k in all_others:
    p_need = _need(qn_counts[k])
    if p_need is None or p_need[0] < P_FLOOR["nifuku"]:
        # Pフロア未満も表示はする（判定廃止）。ただし帯は "-" 表示。
        rows.append({"買い目": f"{om}-{k}", "的中率(推定p)": f"{(qn_counts[k]/trials):.4f}", "バランス帯": "-"})
    else:
        rows.append({"買い目": f"{om}-{k}", "的中率(推定p)": f"{p_need[0]:.4f}", "バランス帯": _band_text(p_need[1])})
qn_df = pd.DataFrame(rows)
def _key_nums(s): return list(map(int, re.findall(r"\d+", str(s))))
st.dataframe(qn_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True), use_container_width=True)

st.markdown("### 二車単（◎→全）")
rows=[]
for k in all_others:
    p_need = _need(ex_counts[k])
    if p_need is None or p_need[0] < P_FLOOR["nitan"]:
        rows.append({"買い目": f"{om}->{k}", "的中率(推定p)": f"{(ex_counts[k]/trials):.4f}", "バランス帯": "-"})
    else:
        rows.append({"買い目": f"{om}->{k}", "的中率(推定p)": f"{p_need[0]:.4f}", "バランス帯": _band_text(p_need[1])})
ex_df = pd.DataFrame(rows)
st.dataframe(ex_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True), use_container_width=True)

st.markdown("### 三連複（◎-[相手]-全）")
rows=[]
# ◎-a-b（a<b, a,b!=◎）を全列挙
pairs=[]
for i,a in enumerate(all_others):
    for b in all_others[i+1:]:
        key = tuple(sorted([om,a,b]))
        cnt = trio_counts.get(key, 0)
        p = cnt / trials
        if p <= 0 or p < P_FLOOR["sanpuku"]:
            rows.append({"買い目": f"{om}-{a}-{b}", "的中率(推定p)": f"{p:.5f}", "バランス帯": "-"})
        else:
            need = 1.0/p
            rows.append({"買い目": f"{om}-{a}-{b}", "的中率(推定p)": f"{p:.5f}", "バランス帯": _band_text(need)})
trio_df = pd.DataFrame(rows)
st.dataframe(trio_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True), use_container_width=True)

st.markdown("### 三連単（◎→[相手]→全）")
rows=[]
for sec in all_others:
    for thr in [t for t in range(1, n_cars+1) if t not in (om, sec)]:
        key = (om, sec, thr)
        cnt = st3_counts.get(key, 0)
        p = cnt / trials
        if p <= 0 or p < P_FLOOR["santan"]:
            rows.append({"買い目": f"{om}->{sec}->{thr}", "的中率(推定p)": f"{p:.5f}", "バランス帯": "-"})
        else:
            need = 1.0/p
            rows.append({"買い目": f"{om}->{sec}->{thr}", "的中率(推定p)": f"{p:.5f}", "バランス帯": _band_text(need)})
st_df = pd.DataFrame(rows)
st.dataframe(st_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True), use_container_width=True)

st.markdown("### ワイド（◎-全）")
rows=[]
for k in all_others:
    cnt = wide_counts[k]
    p = cnt / trials
    if p <= 0 or p < P_FLOOR["wide"]:
        rows.append({"買い目": f"{om}-{k}", "的中率(推定p)": f"{p:.4f}", "バランス帯": "-"})
    else:
        need = 1.0/p
        rows.append({"買い目": f"{om}-{k}", "的中率(推定p)": f"{p:.4f}", "バランス帯": _band_text(need)})
wide_df = pd.DataFrame(rows)
st.dataframe(wide_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True), use_container_width=True)

# ========= 注釈（説明） =========
st.markdown("""
**注記**  
- ここに表示する「バランス帯」は、`必要オッズ(=1/p)` に対して **+10%〜+60%** の幅をかけた参考レンジです。  
- **この帯より安いオッズ（低倍率）は期待値が不足する想定** です。逆に、帯から大きく離れて高倍率になるほど **的中率とのバランスが崩れリスク増** と見なします。  
- 本ツールは印の相対関係だけを使う簡易版です。選手能力やライン情報は反映していません。
""")

# ========= note貼り付け用（見出し＋買い目だけ） =========
def _section_lines(df: pd.DataFrame, title: str) -> list[str]:
    if df is None or df.empty: 
        return [f"{title}", "対象外"]
    out = [title]
    for _, r in df.iterrows():
        out.append(f"{r['買い目']}：{r['バランス帯']}")
    return out

marks_line = " ".join(f"{mk}{car}" for mk,car in marks_input.items())
note_text = "\n".join(
    ["印　"+marks_line, ""] +
    _section_lines(qn_df,   "二車複（◎-全）") + [""] +
    _section_lines(ex_df,   "二車単（◎→全）") + [""] +
    _section_lines(trio_df, "三連複（◎-[相手]-全）") + [""] +
    _section_lines(st_df,   "三連単（◎→[相手]→全）") + [""] +
    _section_lines(wide_df, "ワイド（◎-全）") + ["",
    "（※表示は全通り。Pフロア未満は “-” 表示）"]
)

st.markdown("### 📋 note貼り付け用（買い目＋バランス帯）")
st.text_area("ここを選択してコピー", value=note_text, height=320)
