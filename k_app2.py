# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="印→期待値チェッカー（簡易版）", layout="wide")

# ========= ユーザーが更新している実測値 =========
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},  # 無印フォールバック1
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},  # 無印フォールバック2（値は同じでも可）
}
RANK_FALLBACK_MARK = "α"

# ===== 足切り・帯設定（既存ルールを踏襲） =====
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60  # +10%〜+60%

# ========= ユーティリティ =========
def _sort_key_by_numbers(name: str) -> list[int]:
    return list(map(int, re.findall(r"\d+", str(name))))

def _format_zone_from_p(name: str, bet_type: str, p: float) -> str | None:
    floor = P_FLOOR[bet_type]
    if p < floor or p <= 0:
        return None
    need = 1.0 / p
    if bet_type == "wide":
        return f"{name}：{need:.1f}倍以上で買い"  # ワイドは上限撤廃
    low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
    return f"{name}：{low:.1f}〜{high:.1f}倍なら買い"

def _calibrate_by_marks(n_cars: int, marks_by_car: dict[int,str], key: str) -> np.ndarray:
    """
    key ∈ {'p1','pTop2','pTop3'}
    ベース一様分布を、印別の実測ターゲットに合わせてスケール→正規化
    """
    base = np.ones(n_cars, dtype=float)/n_cars
    m = np.ones(n_cars, dtype=float)
    for idx in range(n_cars):
        mk = marks_by_car.get(idx+1, None)
        if mk not in RANK_STATS:
            mk = RANK_FALLBACK_MARK
        tgt = float(RANK_STATS[mk][key])
        ratio = tgt / max(float(base[idx]), 1e-9)
        # 過剰反応しないようルートで緩和、クリップも控えめ
        m[idx] = float(np.clip(ratio**0.5, 0.35, 2.5))
    v = base * m
    v = v / v.sum()
    return v

# ========= サイドバー：印入力（1記号=1車番） =========
st.sidebar.header("印の入力（1記号=1車番）")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)

def _num_input_for_mark(label, key):
    v = st.sidebar.number_input(label, min_value=0, max_value=n_cars, step=1, value=0, key=key)
    return int(v) if v else None  # 0は未設定として扱う

m_anchor = _num_input_for_mark("◎", "m_ex")   # 必須推奨
m_maru   = _num_input_for_mark("〇", "m_maru")
m_san    = _num_input_for_mark("▲", "m_san")
m_delta  = _num_input_for_mark("△", "m_delta")
m_batsu  = _num_input_for_mark("×", "m_batsu")
m_alpha  = _num_input_for_mark("α", "m_alpha")
m_beta   = _num_input_for_mark("β", "m_beta")

# 競技車番→印 の対応
mark_map = {}
if m_anchor: mark_map[m_anchor] = "◎"
if m_maru:   mark_map[m_maru]   = "〇"
if m_san:    mark_map[m_san]    = "▲"
if m_delta:  mark_map[m_delta]  = "△"
if m_batsu:  mark_map[m_batsu]  = "×"
if m_alpha:  mark_map[m_alpha]  = "α"
if m_beta:   mark_map[m_beta]   = "β"

# 1つの車に複数印が重複しないよう軽いチェック
dup = [num for num in range(1, n_cars+1) if list(mark_map.keys()).count(num)>1]
if dup:
    st.sidebar.error("同じ車番に複数の印が重なっています。印を調整してください。")

# ◎が未入力なら止める
if not m_anchor:
    st.warning("◎（本命）の車番を入力してください。")
    st.stop()

# 無印はフォールバック印に割り当て
marks_by_car = {}
for i in range(1, n_cars+1):
    marks_by_car[i] = mark_map.get(i, RANK_FALLBACK_MARK)

# 画面上部：印のまとめ表示
st.caption("印： " + "  ".join(
    f"{sym}{num}" for sym,num in [
        ("◎",m_anchor),("〇",m_maru),("▲",m_san),("△",m_delta),("×",m_batsu),("α",m_alpha),("β",m_beta)
    ] if num
))

# ========= 確率分布（印→実測率） =========
probs_p3 = _calibrate_by_marks(n_cars, marks_by_car, "pTop3")  # ワイド/三連複
probs_p2 = _calibrate_by_marks(n_cars, marks_by_car, "pTop2")  # 二車複
probs_p1 = _calibrate_by_marks(n_cars, marks_by_car, "p1")     # 二車単/三連単

# ========= EVしきい値（必要オッズ=1/p）生成 =========
# ◎の相手プール
others = [k for k in range(1, n_cars+1) if k != m_anchor]

# --- ワイド（◎–全：上限撤廃、足切りあり） ---
rows_wide = []
for k in others:
    # Top3分布から：◎とkがTop3同時入りの“粗い”近似として min(p3◎, p3k) の下限を採用
    # ※一致性よりも安全側に寄せるための簡易ルール
    p_rough = min(float(probs_p3[m_anchor-1]), float(probs_p3[k-1]))
    line = _format_zone_from_p(f"{m_anchor}-{k}", "wide", p_rough)
    if line:
        rows_wide.append((f"{m_anchor}-{k}", p_rough, line))

if rows_wide:
    df_wide = pd.DataFrame([{"買い目":n, "p(粗)Top3":round(p,4), "基準":txt} for (n,p,txt) in rows_wide])
    df_wide = df_wide.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
else:
    df_wide = pd.DataFrame(columns=["買い目","p(粗)Top3","基準"])

# --- 二車複（◎–全：足切りあり） ---
rows_qn = []
for k in others:
    # 連対の粗い近似：min(p2◎, p2k)
    p_rough = min(float(probs_p2[m_anchor-1]), float(probs_p2[k-1]))
    line = _format_zone_from_p(f"{m_anchor}-{k}", "nifuku", p_rough)
    if line:
        rows_qn.append((f"{m_anchor}-{k}", p_rough, line))
if rows_qn:
    df_qn = pd.DataFrame([{"買い目":n, "p(粗)Top2":round(p,4), "買える帯":txt} for (n,p,txt) in rows_qn])
    df_qn = df_qn.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
else:
    df_qn = pd.DataFrame(columns=["買い目","p(粗)Top2","買える帯"])

# --- 二車単（◎→全：足切りあり） ---
rows_ex = []
for k in others:
    # 1着の粗い近似：p1◎ × p2k を採用（安全側に寄せるため p2k を掛ける）
    p_rough = float(probs_p1[m_anchor-1]) * float(probs_p2[k-1])
    line = _format_zone_from_p(f"{m_anchor}->{k}", "nitan", p_rough)
    if line:
        rows_ex.append((f"{m_anchor}->{k}", p_rough, line))
if rows_ex:
    df_ex = pd.DataFrame([{"買い目":n, "p(粗)1着×相手Top2":round(p,4), "買える帯":txt} for (n,p,txt) in rows_ex])
    df_ex = df_ex.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
else:
    df_ex = pd.DataFrame(columns=["買い目","p(粗)1着×相手Top2","買える帯"])

# --- 三連複（◎-[全相手]-全：全通り、足切りあり） ---
tri_rows = []
for i in range(len(others)):
    for j in range(i+1, len(others)):
        a, b = others[i], others[j]
        # Top3の粗い近似：min(p3◎, p3a, p3b)
        p_rough = float(min(probs_p3[m_anchor-1], probs_p3[a-1], probs_p3[b-1]))
        line = _format_zone_from_p(f"{min(a,b)}-{max(a,b)}-{m_anchor}", "sanpuku", p_rough)
        if line:
            # 車番順で ◎を含めた三つ組を正規化
            tri = sorted([m_anchor, a, b])
            name = f"{tri[0]}-{tri[1]}-{tri[2]}"
            tri_rows.append((name, p_rough, line))
if tri_rows:
    df_tri = pd.DataFrame([{"買い目":n, "p(粗)Top3":round(p,4), "買える帯":txt} for (n,p,txt) in tri_rows])
    df_tri = df_tri.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
else:
    df_tri = pd.DataFrame(columns=["買い目","p(粗)Top3","買える帯"])

# --- 三連単（◎→[相手]→全）：2着は 〇/▲ に限定、足切りあり ---
st2_candidates = [x for x in [m_maru, m_san] if x and x != m_anchor]
rows_st = []
for sec in st2_candidates:
    for thr in [c for c in range(1, n_cars+1) if c not in (m_anchor, sec)]:
        # 粗い近似：p1◎ × p2sec × p3thr（控えめ）
        p_rough = float(probs_p1[m_anchor-1]) * float(probs_p2[sec-1]) * float(probs_p3[thr-1])
        if p_rough >= P_FLOOR["santan"]:
            need = 1.0/p_rough
            low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
            rows_st.append({
                "買い目": f"{m_anchor}->{sec}->{thr}",
                "p(粗)": round(p_rough,5),
                "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"
            })
if rows_st:
    df_st = pd.DataFrame(rows_st)
    df_st = df_st.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
else:
    df_st = pd.DataFrame(columns=["買い目","p(粗)","買える帯"])

# ========= 表示 =========
left, right = st.columns(2)
with left:
    st.subheader("ワイド（◎-全）")
    if len(df_wide)>0:
        st.dataframe(df_wide, use_container_width=True)
    else:
        st.info("対象外（Pフロア未満）")

    st.subheader("二車複（◎-全）")
    if len(df_qn)>0:
        st.dataframe(df_qn, use_container_width=True)
    else:
        st.info("対象外（Pフロア未満）")

with right:
    st.subheader("二車単（◎→全）")
    if len(df_ex)>0:
        st.dataframe(df_ex, use_container_width=True)
    else:
        st.info("対象外（Pフロア未満）")

    st.subheader("三連単（◎→[相手]→全）※2着は〇/▲限定")
    if len(df_st)>0:
        st.dataframe(df_st, use_container_width=True)
    else:
        st.info("対象外（候補なし or Pフロア未満）")

st.subheader("三連複（◎-[相手]-全）")
if len(df_tri)>0:
    st.dataframe(df_tri, use_container_width=True)
else:
    st.info("対象外（Pフロア未満）")

# ========= note貼付け用（簡潔版） =========
def _lines_from_df(df: pd.DataFrame, bet_key: str) -> list[str]:
    if df is None or len(df)==0 or "買い目" not in df.columns: return []
    out=[]
    if "買える帯" in df.columns:
        for _,r in df.iterrows():
            out.append(f"{r['買い目']}：{r['買える帯']}")
    elif "基準" in df.columns:
        for _,r in df.iterrows():
            out.append(str(r["基準"]))
    return out

note_lines = []
note_lines.append("三連複（◎-[相手]-全）")
note_lines += _lines_from_df(df_tri, "sanpuku")
note_lines.append("\n三連単（◎→[相手]→全）")
note_lines += _lines_from_df(df_st, "santan")
note_lines.append("\nワイド（◎-全）")
note_lines += _lines_from_df(df_wide, "wide")
note_lines.append("\n二車複（◎-全）")
note_lines += _lines_from_df(df_qn, "nifuku")
note_lines.append("\n二車単（◎→全）")
note_lines += _lines_from_df(df_ex, "nitan")
note_lines.append("\n※このオッズ以下は期待値以下を想定しています。また、このオッズから高オッズに離れるほどに的中率バランスが崩れハイリスクになります。")
note_lines.append("※返金は受け付けておりません。ご了承の上お楽しみください。")

st.markdown("### 📋 note用（貼り付け）")
st.text_area("ここを選択してコピー", "\n".join(note_lines), height=380)
