# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import re
import unicodedata

st.set_page_config(page_title="政春さんEVチェック（印だけ）", layout="wide")

# ========= 定数 =========
MARKS = ["", "◎", "〇", "▲", "△", "×", "α", "β"]

# エシャロッテさん実測の印別率（そのまま“重み”として利用し、正規化で確率化）
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},
}
FALLBACK = "α"

# 期待値ルール（ワイドは上限撤廃＝「必要◯倍以上」表示）
E_MIN, E_MAX = 0.10, 0.60

def _sort_key_by_numbers(name: str):
    return list(map(int, re.findall(r"\d+", str(name))))

# ========= サイドバー：基本設定 =========
st.sidebar.header("設定")
n_cars = st.sidebar.selectbox("出走数", [5,6,7,8,9], index=2)
trials  = st.sidebar.slider("シミュレーション試行回数", 2000, 30000, 12000, 1000)
tau     = st.sidebar.slider("順位温度(散らし)", 0.5, 2.0, 1.0, 0.05,
                            help="1.0=標準。小さいと硬め/大きいと荒れ気味。")
seed    = st.sidebar.number_input("乱数シード", value=20250904, step=1)

st.title("政春さんEVチェック（印だけで“必要オッズ”表示）")
st.caption("◎からの全通り（ワイド/二車複/三連複/二車単）＋三連単（◎→相手→全）。足切りなしで“必要オッズ”を一覧化。")

# ========= 印入力 =========
st.subheader("印入力（◎は必ず1つ）")
cols = st.columns(n_cars)
marks = {}
for i in range(n_cars):
    with cols[i]:
        car = i+1
        marks[car] = st.selectbox(f"{car}番", MARKS, index=0, key=f"mk_{car}")

# ◎チェック
anchors = [c for c in range(1, n_cars+1) if marks.get(c,"")=="◎"]
if len(anchors)!=1:
    st.warning("◎をちょうど1つ選んでください。")
    st.stop()
one = anchors[0]

# 表示用：印の横並び
def marks_line(mdict):
    order = ["◎","〇","▲","△","×","α","β"]
    rev = {}
    for c,m in mdict.items():
        if m:
            rev.setdefault(m, []).append(str(c))
    parts=[]
    for m in order:
        if m in rev:
            parts.append(f"{m}{'・'.join(rev[m])}")
    return "　".join(parts)

st.markdown("#### 印")
st.write(marks_line(marks))

# ========= 単一順位生成（PL） =========
# 印→“1着強さ”重み（p1値）を芯に、温度tauで散らす
w1 = np.array([RANK_STATS.get(marks.get(i+1,"") or FALLBACK, RANK_STATS[FALLBACK])["p1"]
               for i in range(n_cars)], dtype=float)

# 温度スケール（tau<1で硬め、>1で荒れ）
w1 = np.power(np.maximum(w1, 1e-9), 1.0/tau)
w1 = w1 / w1.sum()

rng = np.random.default_rng(int(seed))

def sample_order_from_weights(weights: np.ndarray) -> list[int]:
    # Gumbel-MaxでPL順位サンプル
    g = -np.log(-np.log(np.clip(rng.random(len(weights)), 1e-12, 1-1e-12)))
    score = np.log(weights+1e-12) + g
    return (np.argsort(-score)+1).tolist()  # 1-indexed car番号

# ========= カウント器 =========
others = [k for k in range(1,n_cars+1) if k!=one]
wide_counts = {k:0 for k in others}
qn_counts   = {k:0 for k in others}
ex_counts   = {k:0 for k in others}
tri_counts  = {}               # 三連複（◎-a-b） unordered
st3_counts  = {}               # 三連単（◎->a->b） ordered

# 三連複の全通り（◎固定、a<bで全列挙）
tri_all = []
for i,a in enumerate(others):
    for b in others[i+1:]:
        t = tuple(sorted([one, a, b]))
        tri_all.append(t)
        tri_counts[t] = 0

# 三連単（◎→[相手]→全）全通り
st3_all=[]
for a in others:
    for b in range(1,n_cars+1):
        if b in (one, a): continue
        st3_all.append((a,b))
        st3_counts[(a,b)] = 0

# ========= シミュレーション =========
for _ in range(trials):
    order = sample_order_from_weights(w1)  # 例: [1,5,2,3,4,6,7]
    top3 = set(order[:3]); top2 = set(order[:2])

    # ワイド（◎-k）：両者Top3
    if one in top3:
        for k in others:
            if k in top3: wide_counts[k] += 1

    # 二車複（◎-k）：両者Top2
    if one in top2:
        for k in others:
            if k in top2: qn_counts[k] += 1

    # 二車単（◎->k）
    if order[0]==one:
        k2 = order[1]
        if k2 in ex_counts: ex_counts[k2] += 1

    # 三連複（◎-a-b）
    if one in top3:
        other2 = sorted(list(top3 - {one}))
        if len(other2)==2:
            t = tuple(sorted([one, other2[0], other2[1]]))
            if t in tri_counts:
                tri_counts[t] += 1

    # 三連単（◎->a->b）
    if order[0]==one:
        a = order[1]; b = order[2]
        if (a,b) in st3_counts:
            st3_counts[(a,b)] += 1

def need_from_cnt(cnt: int) -> float | None:
    if cnt<=0: return None
    p = cnt/float(trials)
    return round(1.0/max(p,1e-12), 2)

# ========= 表（全通り・足切りなし） =========
st.markdown("### 出力（コピー用）")

# 三連複（◎-[相手]-全）… 全通り
rows=[]
for t in sorted(tri_all, key=lambda x: (x[0],x[1],x[2])):
    cnt = tri_counts.get(t,0)
    need = need_from_cnt(cnt)
    if need is None:
        rows.append({"買い目": f"{t[0]}-{t[1]}-{t[2]}", "買える帯": "—"})
    else:
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({"買い目": f"{t[0]}-{t[1]}-{t[2]}", "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
tri_df = pd.DataFrame(rows)

# ワイド（◎-全）… 全通り（“以上”表記）
rows=[]
for k in sorted(others):
    cnt = wide_counts.get(k,0)
    need = need_from_cnt(cnt)
    rows.append({
        "買い目": f"{one}-{k}",
        "必要（=1/p）": "—" if need is None else f"{need:.1f}倍以上で買い"
    })
wide_df = pd.DataFrame(rows)

# 二車複（◎-全）… 全通り
rows=[]
for k in sorted(others):
    cnt = qn_counts.get(k,0)
    need = need_from_cnt(cnt)
    if need is None:
        rows.append({"買い目": f"{one}-{k}", "買える帯": "—"})
    else:
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({"買い目": f"{one}-{k}", "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
qn_df = pd.DataFrame(rows)

# 二車単（◎→全）… 全通り
rows=[]
for k in sorted(others):
    cnt = ex_counts.get(k,0)
    need = need_from_cnt(cnt)
    if need is None:
        rows.append({"買い目": f"{one}->{k}", "買える帯": "—"})
    else:
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({"買い目": f"{one}->{k}", "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
ex_df = pd.DataFrame(rows)

# 三連単（◎→[相手]→全）… 全通り
rows=[]
for a,b in sorted(st3_all, key=lambda x: (x[0],x[1])):
    cnt = st3_counts.get((a,b),0)
    need = need_from_cnt(cnt)
    if need is None:
        rows.append({"買い目": f"{one}->{a}->{b}", "買える帯": "—"})
    else:
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({"買い目": f"{one}->{a}->{b}", "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
st3_df = pd.DataFrame(rows)

# ===== 表示（note貼付けしやすく、かつ見やすく） =====
st.markdown("#### 三連複（◎-[相手]-全）※全通り・車番順")
st.dataframe(tri_df, use_container_width=True)

st.markdown("#### ワイド（◎-全）※全通り・車番順（※ワイドは上限撤廃）")
st.dataframe(wide_df, use_container_width=True)

st.markdown("#### 二車複（◎-全）※全通り・車番順")
st.dataframe(qn_df, use_container_width=True)

st.markdown("#### 二車単（◎→全）※全通り・車番順")
st.dataframe(ex_df, use_container_width=True)

st.markdown("#### 三連単（◎→[相手]→全）※全通り・車番順")
st.dataframe(st3_df, use_container_width=True)

# ===== コピー用テキスト（note用ミニ整形） =====
def lines_from_df(df: pd.DataFrame, key: str, suffix_key: str) -> list[str]:
    if df is None or len(df)==0: return []
    out=[]
    for _,r in df.iterrows():
        name = str(r.get(key,""))
        if suffix_key in r and isinstance(r[suffix_key], str) and r[suffix_key]!="":
            out.append(f"{name}：{r[suffix_key]}")
        else:
            out.append(f"{name}：—")
    out.sort(key=lambda s: _sort_key_by_numbers(s))
    return out

txt = []
txt.append(marks_line(marks))
txt.append("")
txt.append("三連複（◎-[相手]-全）")
txt += lines_from_df(tri_df, "買い目", "買える帯")
txt.append("")
txt.append("ワイド（◎-全）")
txt += lines_from_df(wide_df, "買い目", "必要（=1/p）")
txt.append("")
txt.append("二車複（◎-全）")
txt += lines_from_df(qn_df, "買い目", "買える帯")
txt.append("")
txt.append("二車単（◎→全）")
txt += lines_from_df(ex_df, "買い目", "買える帯")
txt.append("")
txt.append("三連単（◎→[相手]→全）")
txt += lines_from_df(st3_df, "買い目", "買える帯")
txt.append("")
txt.append("（※表示された数値は「期待値成立に必要なオッズの下限」です。これ以下では期待値不足、"
           "これ以上ではオッズが離れるほどに的中率バランスが崩れうるためハイリスクです。ワイドは上限撤廃＝“必要◯倍以上”。）")

st.markdown("### 📋 note貼り付け用")
st.text_area("ここを選択してコピー", "\n".join(txt), height=420)
