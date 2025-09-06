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
# 上位5車BOX（印優先）トグル
# ==============================
st.markdown("---")
show_box = st.checkbox("上位5車のBOX買いを表示（印優先）", value=False,
                       help="印の強さで5車を自動選出（◎>〇>▲>△>×>α>β）。未指定はα扱い。")

if show_box:
    st.markdown("### 🧺 上位5車BOX（印優先）")

    # 印の優先度（高→低）
    MARK_PRI = {"◎": 6, "〇": 5, "▲": 4, "△": 3, "×": 2, "α": 1, "β": 0}
    def mark_score(mk: str) -> int:
        return MARK_PRI.get(mk, MARK_PRI["α"])

    # 全車を（印スコア desc, 車番 asc）で並べ、先頭5車をBOX対象に
    all_cars = sorted(
        [(i, mark_score(car_mark[i])) for i in range(1, n_cars+1)],
        key=lambda x: (-x[1], x[0])
    )
    box_cars = [i for (i, _) in all_cars[:min(5, n_cars)]]
    st.caption(f"BOX対象：{box_cars}（印優先で自動選出）")

    # 共通：数字抽出ソートキー
    numkey = lambda s: list(map(int, re.findall(r"\d+", str(s))))

    # ---- ワイド（5車BOX：全ての組合せ）----
    rows = []
    for i in range(len(box_cars)):
        for j in range(i+1, len(box_cars)):
            a, b = box_cars[i], box_cars[j]
            p = p3[a] * p3[b]              # 独立近似（両者Top3）
            rows.append({"買い目": f"{a}-{b}", "想定p": round(p,4), "バランス帯": band_from_p(p)})
    wide_box_df = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s: s.map(numkey)).reset_index(drop=True)

    # ---- 二車複（5車BOX）----
    rows = []
    for i in range(len(box_cars)):
        for j in range(i+1, len(box_cars)):
            a, b = box_cars[i], box_cars[j]
            p = p2[a] * p2[b]              # 独立近似（両者Top2）
            rows.append({"買い目": f"{a}-{b}", "想定p": round(p,4), "バランス帯": band_from_p(p)})
    qn_box_df = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s: s.map(numkey)).reset_index(drop=True)

    # ---- 三連複（5車BOX：3通り全組合せ）----
    rows = []
    for i in range(len(box_cars)):
        for j in range(i+1, len(box_cars)):
            for k in range(j+1, len(box_cars)):
                a, b, c = box_cars[i], box_cars[j], box_cars[k]
                p = p3[a] * p3[b] * p3[c]   # 独立近似（3者Top3）
                rows.append({"買い目": f"{a}-{b}-{c}", "想定p": round(p,5), "バランス帯": band_from_p(p)})
    trio_box_df = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s: s.map(numkey)).reset_index(drop=True)

    # ---- 二車単（5車BOX：順序あり全通り）----
    rows = []
    for i in range(len(box_cars)):
        for j in range(len(box_cars)):
            if i == j: continue
            a, b = box_cars[i], box_cars[j]
            p = p1[a] * p2[b]              # 近似：1着a × bが連対圏
            rows.append({"買い目": f"{a}->{b}", "想定p": round(p,5), "バランス帯": band_from_p(p)})
    ex_box_df = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s: s.map(numkey)).reset_index(drop=True)

    # ---- 三連単（5車BOX：順序あり全通り=5P3=60）----
    rows = []
    for a in box_cars:
        for b in box_cars:
            if b == a: continue
            for c in box_cars:
                if c == a or c == b: continue
                p = p1[a] * p2[b] * p3[c]   # 近似：1着a × 2着b連対 × 3着cTop3
                rows.append({"買い目": f"{a}->{b}->{c}", "想定p": round(p,6), "バランス帯": band_from_p(p)})
    st_box_df = pd.DataFrame(rows).sort_values(by="買い目", key=lambda s: s.map(numkey)).reset_index(drop=True)

    # 表示（重い場合は必要な券種だけに絞ってもOK）
    st.markdown("#### ワイド（5車BOX）")
    st.dataframe(wide_box_df, use_container_width=True)

    st.markdown("#### 二車複（5車BOX）")
    st.dataframe(qn_box_df, use_container_width=True)

    st.markdown("#### 三連複（5車BOX）")
    st.dataframe(trio_box_df, use_container_width=True)

    st.markdown("#### 二車単（5車BOX）")
    st.dataframe(ex_box_df, use_container_width=True)

    st.markdown("#### 三連単（5車BOX）")
    st.dataframe(st_box_df, use_container_width=True)


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


