# streamlit run mark_ev_simple.py
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="印だけでEV判定（簡易版）", layout="wide")

# ==== あなたの最新RANK_STATSをここに貼り替えOK（初期値は現行値）====
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

# ==== 期待値ルール（現行に合わせる）====
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60  # +10%〜+60%

# ==== UI ====
st.title("印だけでEV判定（◎〇▲△×αβ → 買える帯）")
c1, c2, c3 = st.columns([2,2,3])
with c1:
    n_cars = st.selectbox("出走数", [5,6,7,8,9], index=2)
with c2:
    trials = st.slider("シミュレーション試行回数", 2000, 30000, 8000, 1000)
with c3:
    tau = st.slider("温度 τ（散らし具合）", 0.5, 2.0, 1.0, 0.1)

st.subheader("各車の印を入力（1〜{}）".format(n_cars))
marks_input = {}
cols = st.columns(9)
for i in range(1, n_cars+1):
    with cols[(i-1)%9]:
        marks_input[i] = st.text_input(f"{i}番", value="" if i>3 else ("◎" if i==1 else ("〇" if i==2 else "▲")), max_chars=1)

# ◎/〇/▲ の抽出（相手集合は 〇/▲）
anchor = next((i for i,m in marks_input.items() if m=="◎"), None)
mates = [i for i,m in marks_input.items() if m in ("〇","▲") and i!=anchor]
all_others = [i for i in range(1, n_cars+1) if i!=anchor]

if anchor is None:
    st.warning("◎ が未入力です。『◎』を1頭だけ指定してください。")
    st.stop()

# ==== 単一順位用の“強さベクトル”作成 ====
#   印→(pTop3,pTop2,p1) を重みつき合成し、温度スケールで散らす
w = np.zeros(n_cars, dtype=float)
for idx, i in enumerate(range(1, n_cars+1)):
    mk = marks_input.get(i, "")
    if mk not in RANK_STATS: mk = FALLBACK
    stat = RANK_STATS[mk]
    # 合成強さ：Top3を主、Top2/1着を副（好みで微調整可）
    base = 0.55*stat["pTop3"] + 0.30*stat["pTop2"] + 0.15*stat["p1"]
    w[idx] = base
# 温度 τ：τ<1でシャープ、>1で分散大きく
w = np.power(np.clip(w, 1e-9, 1.0), 1.0/tau)
w = w / w.sum()

# ==== 単一PLサンプルで一貫性を確保（Gumbel-Softmaxトリック）====
rng = np.random.default_rng(20250906)
def sample_order(weights: np.ndarray) -> list[int]:
    g = -np.log(-np.log(np.clip(rng.random(len(weights)), 1e-12, 1-1e-12)))
    score = np.log(weights+1e-12) + g
    # 返すのは車番（1..n）
    return (np.argsort(-score)+1).tolist()

# ==== カウント器 ====
wide_counts = {k:0 for k in all_others}          # ◎-k がTop3内のワイド
qn_counts   = {k:0 for k in all_others}          # 二車複（◎-k がTop2）
ex_counts   = {k:0 for k in all_others}          # 二車単（◎→k）
trioC_counts= {}                                  # 三連複C（◎-[相手]-全）
st3_counts  = {}                                  # 三連単（◎→[相手]→全）

# 三連複Cの対象トリプル列挙（◎ + [〇/▲のどちらかを含む] + もう1頭）
trioC_list = []
if mates:
    for a in all_others:
        for b in all_others:
            if a>=b: continue
            if (a in mates) or (b in mates):
                t = tuple(sorted([anchor, a, b]))
                trioC_list.append(t)
    trioC_list = sorted(set(trioC_list))

# ==== シミュレーション ====
for _ in range(trials):
    order = sample_order(w)  # 1回の試行で全券種イベントを整合的に評価
    top3 = set(order[:3])
    top2 = set(order[:2])

    # ワイド（◎-k）
    if anchor in top3:
        for k in wide_counts.keys():
            if k in top3:
                wide_counts[k] += 1
        # 三連複C
        if mates and len(top3)==3:
            others = sorted(list(top3 - {anchor}))
            if len(others)==2:
                a,b = others
                if (a in mates) or (b in mates):
                    trioC_counts[(min(anchor,a,b), sorted([anchor,a,b])[1], max(anchor,a,b))] = \
                        trioC_counts.get((min(anchor,a,b), sorted([anchor,a,b])[1], max(anchor,a,b)), 0) + 1

    # 二車複
    if anchor in top2:
        for k in qn_counts.keys():
            if k in top2:
                qn_counts[k] += 1

    # 二車単/三連単
    if order[0] == anchor:
        k2 = order[1]
        if k2 in ex_counts:
            ex_counts[k2] += 1
        if mates:
            k3 = order[2]
            if (k2 in mates) and (k3 not in (anchor, k2)):
                st3_counts[(k2, k3)] = st3_counts.get((k2, k3), 0) + 1

# ==== ユーティリティ ====
def _key_nums(s): return list(map(int, re.findall(r"\d+", str(s))))
def need_from_cnt(cnt:int) -> float|None:
    if cnt<=0: return None
    p = cnt/float(trials)
    return round(1.0/p, 2)

def band_from_p(p:float):
    need = 1.0/max(p,1e-12)
    return (need*(1.0+E_MIN), need*(1.0+E_MAX))

# ==== 三連複C ====
st.markdown("### 三連複C（◎-[相手]-全）")
rows=[]
for t in sorted(trioC_counts.keys() if trioC_counts else [], key=lambda x: (x[0],x[1],x[2])):
    cnt = trioC_counts.get(t,0)
    p = cnt/float(trials)
    if p < P_FLOOR["sanpuku"]:  # Pフロア
        continue
    low, high = band_from_p(p)
    rows.append({"買い目": f"{t[0]}-{t[1]}-{t[2]}", "p(想定的中率)": round(p,4), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
trioC_df = pd.DataFrame(rows)
if not trioC_df.empty:
    trioC_df = trioC_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True)
    st.dataframe(trioC_df, use_container_width=True)
else:
    st.info("対象外（Pフロア未満 or 相手未設定）")

# 三複バスケット合成オッズ（下限基準）と相手集合S
S = set()
O_combo = None
if not trioC_df.empty:
    need_list=[]
    for _,r in trioC_df.iterrows():
        nums = list(map(int, re.findall(r"\d+", r["買い目"])))
        others = [x for x in nums if x != anchor]
        S.update(others)
        # 下限は “必要オッズ=1/p”
        p = float(r["p(想定的中率)"])
        if p>0: need_list.append(1.0/p)
    if need_list:
        denom = sum(1.0/x for x in need_list if x>0)
        if denom>0:
            O_combo = float(f"{(1.0/denom):.2f}")
if O_combo is not None and len(S)>0:
    st.caption(f"三連複バスケット合成オッズ（下限基準）：**{O_combo:.2f}倍** / 相手集合S：{sorted(S)}")

# ==== ワイド（上限撤廃：Sは合成オッズ基準 / S外は必要オッズ基準）====
st.markdown("### ワイド（◎-全）")
rows=[]
for k in sorted(wide_counts.keys()):
    p = wide_counts[k]/float(trials)
    if p < P_FLOOR["wide"]: continue
    need = 1.0/p
    rule = "必要オッズ以上"
    ok = True
    if (O_combo is not None) and (k in S):
        if need >= O_combo:
            rule = f"三複被り→合成{O_combo:.2f}倍以上"
        else:
            ok = False
    if ok:
        rows.append({"買い目": f"{anchor}-{k}", "p(想定的中率)": round(p,4), "必要オッズ(=1/p)": round(need,2), "ルール": rule})
wide_df = pd.DataFrame(rows)
if not wide_df.empty:
    wide_df = wide_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True)
    st.dataframe(wide_df, use_container_width=True)
    st.caption("※ワイドは上限撤廃：三連複で使用した相手は合成オッズ以上／三連複から漏れた相手は必要オッズ以上。")
else:
    st.info("対象外（Pフロア未満 or 合成基準で除外）")

# ==== 二車複 ====
st.markdown("### 二車複（◎-全）")
rows=[]
for k in sorted(qn_counts.keys()):
    p = qn_counts[k]/float(trials)
    if p < P_FLOOR["nifuku"]: continue
    low, high = band_from_p(p)
    rows.append({"買い目": f"{anchor}-{k}", "p(想定的中率)": round(p,4), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
qn_df = pd.DataFrame(rows)
if not qn_df.empty:
    qn_df = qn_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True)
    st.dataframe(qn_df, use_container_width=True)
else:
    st.info("対象外")

# ==== 二車単 ====
st.markdown("### 二車単（◎→全）")
rows=[]
for k in sorted(ex_counts.keys()):
    p = ex_counts[k]/float(trials)
    if p < P_FLOOR["nitan"]: continue
    low, high = band_from_p(p)
    rows.append({"買い目": f"{anchor}->{k}", "p(想定的中率)": round(p,4), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
ex_df = pd.DataFrame(rows)
if not ex_df.empty:
    ex_df = ex_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True)
    st.dataframe(ex_df, use_container_width=True)
else:
    st.info("対象外")

# ==== 三連単（◎→[相手]→全）====
st.markdown("### 三連単（◎→[相手]→全）")
rows=[]
for (sec, thr), cnt in st3_counts.items():
    p = cnt/float(trials)
    if p < P_FLOOR["santan"] or p<=0: continue
    need = 1.0/p
    low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
    rows.append({"買い目": f"{anchor}->{sec}->{thr}", "p(想定的中率)": round(p,5), "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"})
st_df = pd.DataFrame(rows)
if not st_df.empty:
    st_df = st_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums)).reset_index(drop=True)
    st.dataframe(st_df, use_container_width=True)
else:
    st.info("対象外")

