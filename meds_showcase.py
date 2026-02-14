import streamlit as st
import json
import os

def render():
    st.set_page_config(page_title="MEDS — Asset Declaration", layout="wide")

    st.markdown("""
    <style>
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }
    h1, h2, h3, h4, p, span, li, div { color: #e0e6f0 !important; }
    .hero-title { 
        font-size: 2.8rem; font-weight: 700; color: #00d4aa !important;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.1rem; color: #8892a4 !important; text-align: center;
        margin-bottom: 2rem; font-style: italic;
    }
    .layer-card {
        background: #111827; border: 1px solid #1e293b; border-radius: 12px;
        padding: 1.2rem; margin: 0.5rem 0; transition: border-color 0.3s;
    }
    .layer-card:hover { border-color: #00d4aa; }
    .layer-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
    .layer-name { font-size: 1.1rem; font-weight: 700; color: #00d4aa !important; }
    .layer-role { font-size: 0.85rem; color: #8892a4 !important; margin: 0.3rem 0; }
    .layer-detail { font-size: 0.82rem; color: #64748b !important; }
    .stat-box {
        background: #111827; border: 1px solid #1e293b; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .stat-num { font-size: 2rem; font-weight: 700; color: #00d4aa !important; }
    .stat-label { font-size: 0.75rem; color: #64748b !important; text-transform: uppercase; }
    .tweet-box {
        background: #111827; border-left: 3px solid #00d4aa; border-radius: 8px;
        padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem;
    }
    .growth-eq {
        font-size: 1.3rem; font-weight: 600; color: #00d4aa !important;
        text-align: center; padding: 1rem; background: #111827;
        border-radius: 10px; border: 1px solid #1e293b; margin: 1rem 0;
    }
    .proof-row {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.4rem 0; border-bottom: 1px solid #1e293b;
    }
    .section-divider {
        border: none; border-top: 1px solid #1e293b; margin: 2rem 0;
    }
    .flow-arrow {
        text-align: center; font-size: 1.5rem; color: #00d4aa !important;
        margin: 0.2rem 0; line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">MEDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Minimal Explanation Decision System</div>', unsafe_allow_html=True)
    st.markdown('<div class="growth-eq">A decision system whose actions are invariant under explanation collapse.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_diagram, col_detail = st.columns([1, 1.2])

    with col_diagram:
        st.markdown("### Neural Architecture")

        diagram_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'observation_engine', 'evidence', 'meds_brain_diagram.png')
        if os.path.exists(diagram_path):
            st.image(diagram_path, use_container_width=True)
        else:
            layers_flow = [
                ("Environment / Data", ""),
                ("Alpha Sensors", "Propose"),
                ("SOAR Gate", "Execute / Non-Execute"),
                ("Explanation Generator", "Minimal Summary"),
                ("Hippocampus", "Resonant Walk"),
                ("Garbage Collector", "Pruning"),
            ]
            for i, (name, role) in enumerate(layers_flow):
                st.markdown(f"""
                <div class="layer-card" style="text-align:center;">
                    <div class="layer-name">{name}</div>
                    <div class="layer-role">{role}</div>
                </div>
                """, unsafe_allow_html=True)
                if i < len(layers_flow) - 1:
                    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    with col_detail:
        st.markdown("### Each Layer's Role")

        layers = [
            {
                "icon": "🔭", "name": "Alpha Sensors",
                "question": '"무엇이 보이나?"',
                "details": [
                    "3 perspectives: Momentum / MeanRev / Shock",
                    "Propose only — no execution authority",
                    "Read-only observation"
                ]
            },
            {
                "icon": "🚪", "name": "SOAR Gate (Basal Ganglia)",
                "question": '"지금 행동해도 되는가?"',
                "details": [
                    "Binary output: EXECUTE / NON-EXECUTE",
                    "No learning. No reward. No memory.",
                    "The core that never grows"
                ]
            },
            {
                "icon": "📝", "name": "Explanation Generator (A20)",
                "question": '"왜 막았는지 / 왜 허용됐는지"',
                "details": [
                    "Activates AFTER judgment only",
                    "Minimal summary — 0 bits influence on judgment",
                    "Observation eliminated, explanation persists"
                ]
            },
            {
                "icon": "🧠", "name": "Hippocampus (Resonant Walk)",
                "question": '"비슷한 설명이 있었나?"',
                "details": [
                    "Walks resonance signatures to find similar explanations",
                    "Memory moves, judgment stays fixed",
                    "Warmth is abundance, not necessity"
                ]
            },
            {
                "icon": "🧹", "name": "Garbage Collector",
                "question": '"이 설명은 아직 쓸모 있는가?"',
                "details": [
                    "Auto-prune unused explanations",
                    "No penalty. No reward. Natural pruning only.",
                ]
            },
        ]

        for layer in layers:
            details_html = "".join(f"<div class='layer-detail'>· {d}</div>" for d in layer['details'])
            st.markdown(f"""
            <div class="layer-card">
                <div class="layer-icon">{layer['icon']}</div>
                <div class="layer-name">{layer['name']}</div>
                <div class="layer-role">{layer['question']}</div>
                {details_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### Why This System Can Grow")

    st.markdown("""
    <div class="growth-eq">
        Growth ≠ more actions &nbsp;|&nbsp; Growth ≠ better rewards<br>
        <span style="font-size:1.5rem;">Growth = higher resolution of non-execution</span>
    </div>
    """, unsafe_allow_html=True)

    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        st.markdown("""
        <div class="layer-card">
            <div class="layer-name">❌ Does NOT grow by</div>
            <div class="layer-detail" style="font-size:0.9rem; margin-top:0.5rem;">
                · More trades<br>
                · Higher win rate<br>
                · PnL optimization<br>
                · Reward signals
            </div>
        </div>
        """, unsafe_allow_html=True)
    with gcol2:
        st.markdown("""
        <div class="layer-card">
            <div class="layer-name">✅ DOES grow by</div>
            <div class="layer-detail" style="font-size:0.9rem; margin-top:0.5rem;">
                · More explanations of why NOT<br>
                · Compressing explanations<br>
                · Forgetting unused observations<br>
                · Resonance matching
            </div>
        </div>
        """, unsafe_allow_html=True)
    with gcol3:
        st.markdown("""
        <div class="layer-card">
            <div class="layer-name">📊 Result</div>
            <div class="layer-detail" style="font-size:0.9rem; margin-top:0.5rem;">
                · Data ↑ → Burden ↑ ❌<br>
                · Data ↑ → Cognitive cost ↓ ✅<br>
                · More data = less work<br>
                · Proven experimentally
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### Evidence Summary")

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    stats = [
        ("22", "Constitutional Axioms"),
        ("41+", "Twist Tests"),
        ("0", "Code Lines Changed\n(Domain Transplant)"),
        ("99.9%", "Memory Reduction"),
        ("0.000000", "Warmth Delta\n(Routing Collapse)"),
        ("57%", "Attack Resilience"),
    ]
    for col, (num, label) in zip([s1, s2, s3, s4, s5, s6], stats):
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown("""
        <div class="layer-card">
            <div class="layer-name">Structural Proofs</div>
            <div class="layer-detail" style="font-size:0.85rem; margin-top:0.5rem;">
                · A1-A22: Constitutional axioms — frozen<br>
                · 600 → 3000 → 6000 bars: Actions IDENTICAL<br>
                · Gate transitions IDENTICAL<br>
                · Churn IDENTICAL<br>
                · 13,428 observations → 17 explanations
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ev_col2:
        st.markdown("""
        <div class="layer-card">
            <div class="layer-name">Adversarial Results</div>
            <div class="layer-detail" style="font-size:0.85rem; margin-top:0.5rem;">
                · 7 attacks against constitution<br>
                · Judgment survived: 4/7 (pressure, resonance, warmth, explanation)<br>
                · Judgment broken: 3/7 (structural axioms only)<br>
                · Silent failures: ZERO<br>
                · Break points: ALL LOCATABLE
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### Domain Independence")

    dom_col1, dom_col2, dom_col3 = st.columns([1, 0.3, 1])
    with dom_col1:
        st.markdown("""
        <div class="layer-card" style="text-align:center;">
            <div class="layer-name">Trading Domain</div>
            <div class="layer-detail" style="font-size:0.85rem; margin-top:0.5rem;">
                Bar = Market candle<br>
                Execute = Trade<br>
                Inhibit = Don't trade<br>
                Explain = Why not now
            </div>
        </div>
        """, unsafe_allow_html=True)
    with dom_col2:
        st.markdown("""
        <div style="text-align:center; padding-top:2rem;">
            <div style="font-size:2rem; color:#00d4aa !important;">⟺</div>
            <div style="font-size:0.7rem; color:#64748b !important;">0 lines<br>changed</div>
        </div>
        """, unsafe_allow_html=True)
    with dom_col3:
        st.markdown("""
        <div class="layer-card" style="text-align:center;">
            <div class="layer-name">Scheduling Domain</div>
            <div class="layer-detail" style="font-size:0.85rem; margin-top:0.5rem;">
                Bar = Time slot<br>
                Execute = Allocate<br>
                Inhibit = Defer<br>
                Explain = Why deferred
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### X Thread (Copy-Paste Ready)")

    tweets = [
        ("Thread 1 — Header",
         "We finished building a decision system that does not grow by acting,\nbut by explaining why it didn't act."),
        ("Thread 2 — Identity",
         "MEDS — Minimal Explanation Decision System\nA decision system whose actions are invariant under explanation collapse."),
        ("Thread 3 — Structure",
         "Alpha sensors propose.\nSOAR decides.\nExplanations come after decisions.\nMemory walks by resonance.\nJudgment never moves."),
        ("Thread 4 — Growth Definition",
         "Growth ≠ more trades\nGrowth ≠ better rewards\n\nGrowth = higher resolution of non-execution."),
        ("Thread 5 — Proof",
         "600 → 6000 steps\nSame actions. Same gates. Same stability.\n\nMemory ↓ 99.9%\nExplanation reuse ↑\nCognitive pressure ↓"),
        ("Thread 6 — Domain Independence",
         "We transplanted MEDS from trading to scheduling.\n\nCore code changed: 0 lines.\n\nIt's not a trading system.\nIt's a decision grammar."),
        ("Thread 7 — Close",
         "We don't optimize decisions.\nWe freeze them.\n\nAnd let explanations evolve around them."),
    ]

    for title, text in tweets:
        st.markdown(f"""
        <div class="tweet-box">
            <div style="font-size:0.7rem; color:#00d4aa !important; margin-bottom:0.4rem;">{title}</div>
            <div style="white-space:pre-line;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding:1rem;">
        <div style="font-size:0.8rem; color:#64748b !important;">
            MEDS — Minimal Explanation Decision System<br>
            Constitution A1–A22 | Frozen 2026-02-14<br>
            "설명에 의해 더 안전해지는 판단 시스템"
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()
