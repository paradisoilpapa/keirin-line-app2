# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="政春さん用：印→EVバランス帯（決定版）", layout="wide")

# ==============================
# 定数（ユーザー実測そのまま）
# ==============================
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},
}
FALLBACK_MARK = "α"

# EVルール（固定）
E_MIN, E_MAX = 0.10, 0.60   # バランス帯：+10%〜+60%を推奨レンジとして表示
# ※Pフロアは使いません（“全通り表示”のため）。必要なら閾値をここに追加できます。

# ==============================
# ユーティリティ
# ==============================
def need_from_p(p: float) -> float | None:
    """必要オッズ=1/p（p<=0ならNone）"""
    if p is None or p <= 0: return None
    return 1.0 / p

def band_from_p(p: float) -> str:
    """バランス帯（=必要オッズ×(1+E_MIN〜E_MAX)）。p<=0は'-'"""
    need = need_from_p(p)
    if need is None: return "-"
    low = need * (1.0 + E_MIN)
    high = need * (1.0 + E_MAX)
    return f"{low:.1f}〜{high:.1f}倍"

def marks_to_probs(mark: str) -> tuple[float,float,float]:
    """印→(p1, pTop2, pTop3)。未知はFALLBACK_MARK"""
    mk = mark if mark in RANK_STATS else FALLBACK_MARK
    d = RANK_STATS[mk]
    return d["p1"], d["pTop2"], d["pTop3"]

def parse_int(x):
    try:
        v = int(x)
        return v if 1 <= v <= 9 else None
    except:
        return None

# ==============================
# 入力
# ==============================
st.title("政春さん用：印→期待値バランス帯（決定的計算版）")

st.caption("印（◎〜β）に対応する車番を1つずつ入力してください。未入力の車番は自動的にα扱い（フォールバック）になります。")

colL, colR = st.columns([1,3])
with colL:
    n_cars = st.selectbox("出走数（5〜9）", [5,6,7,8,9], index=4)

# 1記号1数字の入力（固定）
mk_cols = st.columns(7)
mark_labels = ["◎","〇","▲","△","×","α","β"]
mark_input = {}
for i, mk in enumerate(mark_labels):
    with mk_cols[i]:
        mark_input[mk] = st.text_input(mk, value="", max_chars=1, help=f"{mk}の車番（1〜{n_cars}）")

# ◎は必須
anchor_car = parse_int(mark_input["◎"])
if anchor_car is None:
    st.warning("◎（本命）の車番を1つ入力してください。")
    st.stop()

# 各印→車番 の辞書（数値化）。重複は後勝ち（最後に入力した方を優先）
mark_car = {}
used = set()
for mk in mark_labels:
    num = parse_int(mark_input[mk])
    if num is not None and 1 <= num <= n_cars:
        mark_car[mk] = num
        used.add(num)

# 全車の印割当：明示された車番はその印、未指定はα
car_mark = {}
for i in range(1, n_cars+1):
    found = None
    for mk, num in mark_car.items():
        if num == i:
            found = mk
            break
    car_mark[i] = found if found else FALLBACK_MARK

# 概要の表示
st.caption("印の割当：" + "　".join([f"{i}番:{car_mark[i]}" for i in range(1, n_cars+1)]))

# ==============================
# 決定的な想定確率（独立近似）
# ==============================
# 各車に p1/p2/p3 を割当
p1 = {}
p2 = {}
p3 = {}
for i in range(1, n_cars+1):
    _p1, _p2, _p3 = marks_to_probs(car_mark[i])
    p1[i] = float(_p1)
    p2[i] = float(_p2)
    p3[i] = float(_p3)

# ==============================
# 出力：◎からの固定表示（ワイド/二複/二単＝全通り、三複＝◎+任意2、三単＝◎→相手→全）
# ==============================
st.markdown("### 🎯 買い目（◎から固定表示／バランス帯）")
st.caption("※この数値は“印→実測率”のみから決定的に計算しています（乱数・シミュレーション不使用）。")

others = [i for i in range(1, n_cars+1) if i != anchor_car]

# ---- ワイド（◎-全） ----
rows = []
for b in others:
    # 近似：両者がTop3に入るイベントを独立近似で p ≈ p3_a * p3_b
    p = p3[anchor_car] * p3[b]
    rows.append({
        "買い目": f"{anchor_car}-{b}",
        "想定p": round(p, 4),
        "バランス帯": band_from_p(p),
    })
wide_df = pd.DataFrame(rows).sort_values(
    by="買い目",
    key=lambda s: s.map(lambda name: list(map(int, re.findall(r"\d+", str(name)))) )
).reset_index(drop=True)
st.markdown("#### ワイド（◎-全）※車番順")
st.dataframe(wide_df, use_container_width=True)

# ---- 二車複（◎-全） ----
rows = []
for b in others:
    # 近似：両者がTop2に入る p ≈ p2_a * p2_b
    p = p2[anchor_car] * p2[b]
    rows.append({
        "買い目": f"{anchor_car}-{b}",
        "想定p": round(p, 4),
        "バランス帯": band_from_p(p),
    })
qn_df = pd.DataFrame(rows).sort_values(
    by="買い目",
    key=lambda s: s.map(lambda name: list(map(int, re.findall(r"\d+", str(name)))) )
).reset_index(drop=True)
st.markdown("#### 二車複（◎-全）※車番順")
st.dataframe(qn_df, use_container_width=True)

# ---- 二車単（◎→全） ----
rows = []
for b in others:
    # 近似：1着が◎、もう一方が連対圏 p ≈ p1_a * p2_b
    p = p1[anchor_car] * p2[b]
    rows.append({
        "買い目": f"{anchor_car}->{b}",
        "想定p": round(p, 4),
        "バランス帯": band_from_p(p),
    })
ex_df = pd.DataFrame(rows).sort_values(
    by="買い目",
    key=lambda s: s.map(lambda name: list(map(int, re.findall(r"\d+", str(name)))) )
).reset_index(drop=True)
st.markdown("#### 二車単（◎→全）※車番順")
st.dataframe(ex_df, use_container_width=True)

# ---- 三連複（◎-[相手]-全） ----
rows = []
for i in range(len(others)):
    for j in range(i+1, len(others)):
        b, c = others[i], others[j]
        # 近似：3者がTop3に入る p ≈ p3_a * p3_b * p3_c
        p = p3[anchor_car] * p3[b] * p3[c]
        rows.append({
            "買い目": f"{anchor_car}-{b}-{c}",
            "想定p": round(p, 5),
            "バランス帯": band_from_p(p),
        })
trio_df = pd.DataFrame(rows).sort_values(
    by="買い目",
    key=lambda s: s.map(lambda name: list(map(int, re.findall(r"\d+", str(name)))) )
).reset_index(drop=True)
st.markdown("#### 三連複（◎-[相手]-全）※車番順")
st.dataframe(trio_df, use_container_width=True)

# ---- 三連単（◎→[相手]→全） ----
rows = []
for b in others:
    for c in [x for x in range(1, n_cars+1) if x not in (anchor_car, b)]:
        # 近似：1着=◎、2着=b（連対圏の近似）＆3着=c（Top3近似）
        # 決定的・単純化のため p ≈ p1_a * p2_b * p3_c
        p = p1[anchor_car] * p2[b] * p3[c]
        rows.append({
            "買い目": f"{anchor_car}->{b}->{c}",
            "想定p": round(p, 6),
            "バランス帯": band_from_p(p),
        })
santan_df = pd.DataFrame(rows).sort_values(
    by="買い目",
    key=lambda s: s.map(lambda name: list(map(int, re.findall(r"\d+", str(name)))) )
).reset_index(drop=True)
st.markdown("#### 三連単（◎→[相手]→全）※車番順")
st.dataframe(santan_df, use_container_width=True)

# ==============================
# メモ用（note貼り付けスタイル）
# ==============================
st.markdown("### 📝 コピー用（note貼り付けスタイル）")
def lines_from_df(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty: return f"{title}\n対象外"
    lines = [f"{row['買い目']}：{row['バランス帯']}" for _, row in df.iterrows()]
    return f"{title}\n" + "\n".join(lines)

note_text = (
    f"◎{anchor_car} "
    + " ".join([mk+str(mark_car[mk]) for mk in mark_labels if mk in mark_car and mk != '◎'])
    + "\n\n"
    + lines_from_df(trio_df,  "三連複（◎-[相手]-全）") + "\n\n"
    + lines_from_df(santan_df,"三連単（◎→[相手]→全）") + "\n\n"
    + lines_from_df(qn_df,    "二車複（◎-全）") + "\n\n"
    + lines_from_df(ex_df,    "二車単（◎→全）") + "\n\n"
    + lines_from_df(wide_df,  "ワイド（◎-全）") + "\n\n"
    + "※このオッズ以下は期待値以下を想定しています。また、このオッズから高オッズに離れるほどに的中率バランスが崩れハイリスクになります。"
)
st.text_area("ここを選択してコピー", note_text, height=320)

