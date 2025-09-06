# ==============================
# 買い目（想定的中率 → 必要オッズ=1/p）  ※◎から“固定表示”
# ==============================


one = result_marks.get("◎", None)
two = result_marks.get("〇", None)
three = result_marks.get("▲", None)

if one is None:
    st.warning("◎未決定のため買い目はスキップ")
    trioC_df = wide_df = qn_df = ex_df = santan_df = None
else:
    # base：SBなしスコア → softmax
    strength_map = dict(velobi_wo)
    xs = np.array([strength_map.get(i,0.0) for i in range(1, n_cars+1)], dtype=float)
    if xs.std() < 1e-12:
        base = np.ones_like(xs)/len(xs)
    else:
        z = (xs - xs.mean())/(xs.std()+1e-12)
        base = np.exp(z); base = base/base.sum()

    # 印→確率のゆる校正
    mark_by_car = {car: None for car in range(1, n_cars+1)}
    for mk, car in result_marks.items():
        if car is not None and 1 <= car <= n_cars:
            mark_by_car[car] = mk

    expo = 0.7 if confidence == "優位" else (1.0 if confidence == "互角" else 1.3)

    def calibrate_probs(base_vec: np.ndarray, stat_key: str) -> np.ndarray:
        m = np.ones(n_cars, dtype=float)
        for idx, car in enumerate(range(1, n_cars+1)):
            mk = mark_by_car.get(car)
            if mk not in RANK_STATS:
                mk = RANK_FALLBACK_MARK
            tgt = float(RANK_STATS[mk][stat_key])
            ratio = tgt / max(float(base_vec[idx]), 1e-9)
            m[idx] = float(np.clip(ratio**(0.5*expo), 0.25, 2.5))
        probs = base_vec * m
        probs = probs / probs.sum()
        return probs

    # 券種ごとの“イベント確率”用ベクトル
    probs_p3 = calibrate_probs(base, "pTop3")  # ワイド・三連複
    probs_p2 = calibrate_probs(base, "pTop2")  # 二車複
    probs_p1 = calibrate_probs(base, "p1")     # 二車単・三連単

    rng = np.random.default_rng(20250830)
    trials = st.slider("シミュレーション試行回数", 1000, 20000, 8000, 1000)

    def sample_order_from_probs(pvec: np.ndarray) -> list[int]:
        # Plackett–Luce風のGumbelノイズ順位決定
        g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
        score = np.log(pvec+1e-12) + g
        return (np.argsort(-score)+1).tolist()

    all_others = [i for i in range(1, n_cars+1) if i != one]

    # === カウント器（◎から“全相手”＝固定表示用） ===
    wide_counts = {k:0 for k in all_others}
    qn_counts   = {k:0 for k in all_others}
    ex_counts   = {k:0 for k in all_others}
    st3_counts  = {}  # 三連単（◎→相手→全）は従来どおり（必要なら固定表示化も可能）

    # 三連複は「◎＋任意の2頭（全相手）」の全組み合わせ
    trio_all = []
    for a_i in range(len(all_others)):
        for b_i in range(a_i+1, len(all_others)):
            a, b = all_others[a_i], all_others[b_i]
            t = tuple(sorted([one, a, b]))
            trio_all.append(t)
    trio_counts = {t:0 for t in trio_all}

    # === シミュレーション ===
    for _ in range(trials):
        # p3：Top3イベント（ワイド・三連複）
        order_p3 = sample_order_from_probs(probs_p3)
        top3_p3 = set(order_p3[:3])
        if one in top3_p3:
            # ワイド：◎-k がともにTop3ならカウント
            for k in all_others:
                if k in top3_p3:
                    wide_counts[k] += 1
            # 三連複：◎を含むTop3が確定したとき、その組合せが一致なら加算
            others = list(top3_p3 - {one})
            if len(others) == 2:
                a, b = sorted(others)
                t = tuple(sorted([one, a, b]))
                if t in trio_counts:
                    trio_counts[t] += 1

        # p2：Top2イベント（二車複）
        order_p2 = sample_order_from_probs(probs_p2)
        top2_p2 = set(order_p2[:2])
        if one in top2_p2:
            for k in all_others:
                if k in top2_p2:
                    qn_counts[k] += 1

        # p1：1着イベント（二車単・三連単）
        order_p1 = sample_order_from_probs(probs_p1)
        if order_p1[0] == one:
            k2 = order_p1[1]
            if k2 in ex_counts:
                ex_counts[k2] += 1
            # 三連単は従来運用：◎→[相手]→全（固定表示にしたい場合は全相手へ拡張可）
            k3 = order_p1[2]
            if (k2 in all_others) and (k3 not in (one, k2)):
                st3_counts[(k2, k3)] = st3_counts.get((k2, k3), 0) + 1

    # ====== Pフロア（最低想定p）とEV帯（“バランス帯”へ表記変更） ======
    P_FLOOR = globals().get("P_FLOOR", {
        "wide": 0.060, "sanpuku": 0.040, "nifuku": 0.050, "nitan": 0.040, "santan": 0.030
    })
    # 展開で複系だけ微調整（±10%）
    scale = 1.00
    if confidence == "優位":   scale = 0.90
    elif confidence == "混線": scale = 1.10
    for k in ("wide","sanpuku","nifuku"):
        P_FLOOR[k] *= scale

    E_MIN = globals().get("E_MIN", 0.00)   # 期待値下限（0%）
    E_MAX = globals().get("E_MAX", 0.50)   # 期待値上限（+50%）

    def _p_from_cnt(cnt: int) -> float:
        return cnt / trials if cnt>0 else 0.0

    # === ワイド（◎-全）— 常に全相手を表示（固定表示） ===
    rows = []
    for k in sorted(all_others):
        p = _p_from_cnt(wide_counts[k])
        need = (1.0/p) if p>0 else None
        if p >= P_FLOOR["wide"] and need is not None:
            eval_txt = "バランス帯（下限）"
            rows.append({
                "買い目": f"{one}-{k}",
                "p(想定的中率)": round(p, 4),
                "必要オッズ(=1/p)": round(need, 2),
                "評価": eval_txt
            })
        else:
            rows.append({
                "買い目": f"{one}-{k}",
                "p(想定的中率)": round(p, 4),
                "必要オッズ(=1/p)": "-" if need is None else round(need,2),
                "評価": "対象外（Pフロア未満）"
            })
    wide_df = pd.DataFrame(rows)
    st.markdown("#### ワイド（◎-全）※車番順（固定表示）")
    if len(wide_df) > 0:
        wide_df = wide_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
        st.dataframe(wide_df, use_container_width=True)

    # === 二車複（◎-全）— 常に全相手を表示（固定表示） ===
    rows = []
    for k in sorted(all_others):
        p = _p_from_cnt(qn_counts[k])
        if p <= 0:
            rows.append({"買い目": f"{one}-{k}", "p(想定的中率)": 0.0, "バランス帯": "-", "評価": "対象外（Pフロア未満）"})
            continue
        need = 1.0/p
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        if p >= P_FLOOR["nifuku"]:
            eval_txt = "バランス帯"
            rows.append({
                "買い目": f"{one}-{k}",
                "p(想定的中率)": round(p, 4),
                "バランス帯": f"{low:.1f}〜{high:.1f}倍",
                "評価": eval_txt
            })
        else:
            rows.append({
                "買い目": f"{one}-{k}",
                "p(想定的中率)": round(p, 4),
                "バランス帯": "-",
                "評価": "対象外（Pフロア未満）"
            })
    qn_df = pd.DataFrame(rows)
    st.markdown("#### 二車複（◎-全）※車番順（固定表示）")
    if len(qn_df) > 0:
        qn_df = qn_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
        st.dataframe(qn_df, use_container_width=True)

    # === 二車単（◎→全）— 常に全相手を表示（固定表示） ===
    rows = []
    for k in sorted(all_others):
        p = _p_from_cnt(ex_counts[k])
        if p <= 0:
            rows.append({"買い目": f"{one}->{k}", "p(想定的中率)": 0.0, "バランス帯": "-", "評価": "対象外（Pフロア未満）"})
            continue
        need = 1.0/p
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        if p >= P_FLOOR["nitan"]:
            eval_txt = "バランス帯"
            rows.append({
                "買い目": f"{one}->{k}",
                "p(想定的中率)": round(p, 4),
                "バランス帯": f"{low:.1f}〜{high:.1f}倍",
                "評価": eval_txt
            })
        else:
            rows.append({
                "買い目": f"{one}->{k}",
                "p(想定的中率)": round(p, 4),
                "バランス帯": "-",
                "評価": "対象外（Pフロア未満）"
            })
    ex_df = pd.DataFrame(rows)
    st.markdown("#### 二車単（◎→全）※車番順（固定表示）")
    if len(ex_df) > 0:
        ex_df = ex_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
        st.dataframe(ex_df, use_container_width=True)

    # === 三連複（◎-[相手]-全）— 常に全相手2頭の組合せを表示（固定表示） ===
    rows = []
    for t in sorted(trio_all, key=lambda name: _sort_key_by_numbers("-".join(map(str,name)))):
        cnt = trio_counts.get(t, 0)
        p = _p_from_cnt(cnt)
        if p <= 0:
            rows.append({"買い目": f"{t[0]}-{t[1]}-{t[2]}", "p(想定的中率)": 0.0, "バランス帯": "-", "評価": "対象外（Pフロア未満）"})
            continue
        need = 1.0 / p
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        if p >= P_FLOOR["sanpuku"]:
            eval_txt = "バランス帯"
            rows.append({
                "買い目": f"{t[0]}-{t[1]}-{t[2]}",
                "p(想定的中率)": round(p, 4),
                "バランス帯": f"{low:.1f}〜{high:.1f}倍",
                "評価": eval_txt
            })
        else:
            rows.append({
                "買い目": f"{t[0]}-{t[1]}-{t[2]}",
                "p(想定的中率)": round(p, 4),
                "バランス帯": "-",
                "評価": "対象外（Pフロア未満）"
            })
    trioC_df = pd.DataFrame(rows)
    st.markdown("#### 三連複（◎-[相手]-全）※車番順（固定表示）")
    if len(trioC_df) > 0:
        # 文字列キー用の並べ替え
        def _key_nums_tri(s): return list(map(int, re.findall(r"\d+", s)))
        trioC_df = trioC_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_tri)).reset_index(drop=True)
        st.dataframe(trioC_df, use_container_width=True)

# ==============================
# note用：ヘッダー〜“バランス帯”まとめ
# ==============================
st.markdown("### 📋 note用（“バランス帯”表示）")

def _format_line_zone_note(name: str, bet_type: str, p: float) -> str | None:
    floor = P_FLOOR.get(bet_type, 0.03 if bet_type=="santan" else 0.0)
    if p < floor: return None
    need = 1.0 / max(p, 1e-12)
    if bet_type == "wide":
        return f"{name}：{need:.1f}倍以上でバランス"
    low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
    return f"{name}：{low:.1f}〜{high:.1f}倍（バランス帯）"

def _zone_lines_from_df(df: pd.DataFrame | None, bet_type_key: str) -> list[str]:
    if df is None or len(df) == 0 or "買い目" not in df.columns:
        return []
    rows = []
    for _, r in df.iterrows():
        name = str(r["買い目"])
        if "バランス帯" in r and r["バランス帯"] and r.get("評価","").startswith("バランス"):
            rows.append((name, f"{name}：{r['バランス帯']}" if bet_type_key!="wide" else f"{name}：{r.get('必要オッズ(=1/p)','-')}倍以上でバランス"))
    rows_sorted = sorted(rows, key=lambda x: _sort_key_by_numbers(x[0]))
    return [ln for _, ln in rows_sorted]

def _section_text(title: str, lines: list[str]) -> str:
    if not lines: return f"{title}\n対象外（Pフロア未満）"
    return f"{title}\n" + "\n".join(lines)

line_text = "　".join([x for x in line_inputs if str(x).strip()])
score_order_text = " ".join(str(no) for no,_ in velobi_wo)
marks_line = " ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)

txt_trioC = _section_text("三連複（◎-[相手]-全）",
                          _zone_lines_from_df(trioC_df, "sanpuku") if one is not None else [])
txt_wide  = _section_text("ワイド（◎-全）",
                          _zone_lines_from_df(wide_df, "wide") if one is not None else [])
txt_qn    = _section_text("二車複（◎-全）",
                          _zone_lines_from_df(qn_df, "nifuku") if one is not None else [])
txt_ex    = _section_text("二車単（◎→全）",
                          _zone_lines_from_df(ex_df, "nitan") if one is not None else [])

note_text = (
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}\n"
    f"\n"
    f"{txt_trioC}\n\n"
    f"{txt_wide}\n\n"
    f"{txt_qn}\n\n"
    f"{txt_ex}\n"
    "\n※このオッズ以下は期待値（EV）下振れを想定。離れるほどに確率—オッズのバランスが崩れ、リスクが高まります。\n"
    "※▲は「期待値調整枠」。的中率が低すぎる場合やオッズバランスが悪い場合は買い目に反映されないことがあります（印はシグナル）。\n"
    "※返金は受け付けておりません。ご了承の上お楽しみください。"
)
st.text_area("ここを選択してコピー", note_text, height=360)


