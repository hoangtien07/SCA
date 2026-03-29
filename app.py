"""
app.py — Streamlit MVP

Skincare AI — Personalized Regimen Consultation
Run: streamlit run app.py

Flow:
  1. Upload face photo (optional)
  2. Fill skin questionnaire
  3. AI analyzes + retrieves evidence + generates regimen
  4. View structured AM/PM routine with citations
"""
import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import settings

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.evidence-A { color: #0F6E56; font-weight: 500; }
.evidence-B { color: #854F0B; font-weight: 500; }
.evidence-C { color: #5F5E5A; font-weight: 500; }
.warning-box { background: #FCEBEB; border-left: 3px solid #A32D2D;
               padding: 10px 14px; border-radius: 4px; margin: 8px 0; }
.info-box    { background: #E6F1FB; border-left: 3px solid #185FA5;
               padding: 10px 14px; border-radius: 4px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✦ Skincare AI")
    st.caption("Evidence-based personalized regimens")
    st.divider()

    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password",
                                value=settings.openai_api_key or "")
    anthropic_key = st.text_input("Anthropic API Key", type="password",
                                   value=settings.anthropic_api_key or "")

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Evidence chunks to retrieve", 4, 16, 8)
    evidence_filter = st.multiselect(
        "Minimum evidence level",
        ["A", "B", "C"],
        default=["A", "B"],
        help="A=RCT/meta-analysis, B=cohort/review, C=expert opinion"
    )

    st.divider()
    st.subheader("Advanced options")
    show_xai = st.checkbox(
        "Show AI explanation heatmap",
        value=False,
        help=(
            "Uses LIME + MobileNetV2 surrogate to highlight skin regions that influenced "
            "the analysis. Requires xai_enabled=True in settings. "
            "Note: this is an approximation of GPT-4o's reasoning."
        ),
    )
    use_async = st.checkbox(
        "Use async generation (requires Redis)",
        value=False,
        help="Submits to background queue and polls for result. Shows progress bar.",
    )

    st.divider()
    st.caption("⚠️ For informational purposes only. Not a substitute for dermatologist advice.")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Personalized Skincare Regimen")
st.caption("Powered by peer-reviewed dermatology research")

tab_input, tab_result = st.tabs(["📋 Your skin profile", "✦ Your regimen"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Input
# ══════════════════════════════════════════════════════════════════════════════
with tab_input:
    col_photo, col_form = st.columns([1, 2], gap="large")

    # ── Photo upload ──────────────────────────────────────────────────────────
    with col_photo:
        st.subheader("Skin photo (optional)")
        uploaded_file = st.file_uploader(
            "Upload a clear, well-lit photo of your face (no filters)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Your uploaded photo", use_column_width=True)
            st.info("AI will analyze skin texture, tone, and visible conditions.")
        else:
            st.info("Photo analysis is optional — you can fill the questionnaire alone.")

    # ── Questionnaire ─────────────────────────────────────────────────────────
    with col_form:
        st.subheader("Skin questionnaire")

        with st.form("skin_profile_form"):
            # Basic info
            c1, c2 = st.columns(2)
            with c1:
                age_range = st.selectbox(
                    "Age range",
                    ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
                )
                skin_type = st.selectbox(
                    "Skin type (self-assessed)",
                    ["Oily", "Dry", "Combination", "Normal", "Sensitive", "Not sure"]
                )
            with c2:
                fitzpatrick = st.selectbox(
                    "Fitzpatrick skin type",
                    ["I — Very fair, always burns",
                     "II — Fair, usually burns",
                     "III — Medium, sometimes burns",
                     "IV — Olive, rarely burns",
                     "V — Brown, very rarely burns",
                     "VI — Dark, never burns"],
                    help="Used to tailor recommendations safely for your skin tone"
                )
                climate = st.selectbox(
                    "Climate / environment",
                    ["Humid tropical", "Dry / arid", "Temperate", "Cold / dry", "Polluted urban"]
                )

            st.markdown("**Primary skin concerns** (select all that apply)")
            concern_cols = st.columns(3)
            all_concerns = [
                "Acne / breakouts", "Blackheads / whiteheads", "Cystic acne",
                "Oiliness / shine", "Dryness / flakiness", "Dehydration",
                "Fine lines / wrinkles", "Loss of firmness", "Dullness",
                "Dark spots / hyperpigmentation", "Melasma", "Uneven skin tone",
                "Redness / rosacea", "Sensitive / reactive skin", "Large pores",
                "Rough texture", "Eczema / dermatitis", "Psoriasis",
            ]
            selected_concerns = []
            for i, concern in enumerate(all_concerns):
                col = concern_cols[i % 3]
                if col.checkbox(concern, key=f"concern_{i}"):
                    selected_concerns.append(concern)

            primary_goal = st.text_input(
                "Primary goal (in your own words)",
                placeholder="e.g. Clear hormonal acne while keeping skin hydrated"
            )

            st.markdown("**Current routine & history**")
            c3, c4 = st.columns(2)
            with c3:
                current_actives = st.text_area(
                    "Active ingredients you currently use",
                    placeholder="e.g. Retinol 0.3%, Niacinamide 10%, SPF 50",
                    height=80,
                )
                previous_treatments = st.text_area(
                    "Treatments tried before (and results)",
                    placeholder="e.g. Benzoyl peroxide — worked but too drying",
                    height=80,
                )
            with c4:
                allergies = st.text_area(
                    "Known allergies / sensitivities",
                    placeholder="e.g. Fragrance, essential oils, salicylic acid",
                    height=80,
                )
                medications = st.text_area(
                    "Current medications (topical or oral)",
                    placeholder="e.g. Isotretinoin, oral contraceptives, topical clindamycin",
                    height=80,
                )

            c5, c6 = st.columns(2)
            with c5:
                pregnancy = st.checkbox("Currently pregnant or breastfeeding")
                routine_complexity = st.select_slider(
                    "Routine complexity preference",
                    options=["Minimal (3 steps)", "Simple (5 steps)",
                             "Moderate (7 steps)", "Full (10+ steps)"],
                    value="Simple (5 steps)"
                )
            with c6:
                budget = st.selectbox(
                    "Monthly skincare budget",
                    ["< $30", "$30–$60", "$60–$100", "$100–$200", "$200+"]
                )
                skin_sensitivity = st.select_slider(
                    "Skin sensitivity",
                    options=["Not sensitive", "Mildly sensitive",
                             "Moderately sensitive", "Very sensitive"],
                    value="Mildly sensitive"
                )

            submitted = st.form_submit_button(
                "✦ Generate my personalized regimen",
                type="primary",
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# Processing
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    if not openai_key or not anthropic_key:
        st.error("Please enter your OpenAI and Anthropic API keys in the sidebar.")
        st.stop()

    if not selected_concerns and not primary_goal:
        st.warning("Please select at least one skin concern or describe your goal.")
        st.stop()

    # Build questionnaire profile dict
    profile_raw = {
        "age_range": age_range,
        "skin_type": skin_type.lower().replace(" ", "_"),
        "fitzpatrick": fitzpatrick.split("—")[0].strip(),
        "climate": climate,
        "concerns": selected_concerns,
        "primary_goal": primary_goal,
        "current_actives": [a.strip() for a in current_actives.split(",") if a.strip()],
        "previous_treatments": previous_treatments,
        "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
        "medications": [m.strip() for m in medications.split(",") if m.strip()],
        "pregnancy": pregnancy,
        "routine_complexity": routine_complexity,
        "budget": budget,
        "skin_sensitivity": skin_sensitivity,
    }

    # ── Async path (Prompt #06) ───────────────────────────────────────────────
    if use_async:
        st.info("Submitting to async queue... (requires Redis)")
        try:
            from src.workers.tasks import generate_regimen_task
            task = generate_regimen_task.delay(
                profile_dict=profile_raw,
            )
            task_id = task.id

            progress_bar = st.progress(0, text="Queued...")
            status_text = st.empty()
            max_polls = 80  # 80 × 1.5s = 2 min timeout

            for _ in range(max_polls):
                import time as _time
                _time.sleep(1.5)

                try:
                    from celery.result import AsyncResult
                    from src.workers.celery_app import celery_app
                    result = AsyncResult(task_id, app=celery_app)
                    meta = result.info or {}
                    status = meta.get("status", result.state)
                    progress = meta.get("progress", 0)
                except Exception:
                    status = "UNKNOWN"
                    progress = 0

                status_text.text(f"Status: {status}")
                progress_bar.progress(min(progress, 100), text=status)

                if status == "SUCCESS":
                    st.success("Regimen ready!")
                    # Store result in session state — result is in meta["result"]
                    # For full async display, reload page
                    st.info("Async result stored. Refresh to see full results.")
                    st.stop()
                elif status == "FAILURE":
                    st.error(f"Generation failed: {meta.get('error', 'Unknown error')}")
                    st.stop()

            st.warning("Timed out waiting for async result. Try synchronous mode.")
            st.stop()
        except Exception as e:
            st.warning(f"Async mode unavailable ({e}). Falling back to synchronous.")
            # Fall through to synchronous path

    with st.spinner("Analyzing your skin profile..."):
        # ── Vision analysis ───────────────────────────────────────────────
        vision_data = None
        xai_result = None
        if uploaded_file:
            try:
                from src.agents.vision_analyzer import VisionAnalyzer
                analyzer = VisionAnalyzer(api_key=openai_key)
                uploaded_file.seek(0)
                image_bytes_for_xai = uploaded_file.read()
                vision_data = analyzer.analyze_bytes(
                    image_bytes_for_xai,
                    media_type=f"image/{uploaded_file.type.split('/')[-1]}"
                )
                # Merge vision + questionnaire
                profile = analyzer.merge_with_questionnaire(vision_data, profile_raw)

                # XAI explanation (Prompt #12) — opt-in
                if show_xai and settings.xai_enabled:
                    try:
                        from src.agents.xai_explainer import VisionExplainer
                        explainer = VisionExplainer(
                            num_samples=settings.xai_num_samples,
                            surrogate_model=settings.xai_surrogate_model,
                        )
                        primary_cond = vision_data.detected_conditions[0] if vision_data.detected_conditions else ""
                        xai_result = explainer.explain_analysis(image_bytes_for_xai, condition=primary_cond)
                    except Exception as e:
                        st.warning(f"XAI explanation failed ({e}), continuing without heatmap.")

            except Exception as e:
                st.warning(f"Vision analysis failed ({e}), proceeding with questionnaire only.")
                profile = profile_raw
        else:
            profile = profile_raw

    with st.spinner("Retrieving scientific evidence..."):
        try:
            from src.pipeline.indexer import ChromaIndexer
            from src.agents.rag_retriever import RAGRetriever

            indexer = ChromaIndexer(
                persist_dir=settings.chroma_persist_dir,
                embedding_model=settings.embedding_model,
                openai_api_key=openai_key,
            )
            retriever = RAGRetriever(indexer=indexer, top_k=top_k)

            query = retriever.build_query_from_profile(profile)
            evidence = retriever.retrieve(
                query=query,
                evidence_levels=evidence_filter if evidence_filter else None,
            )

            if not evidence:
                st.error("No evidence retrieved. Make sure you've run the indexing script first.")
                st.code("python scripts/run_collection.py\npython scripts/run_indexing.py")
                st.stop()

        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.caption("Make sure the knowledge base is built: `python scripts/run_indexing.py`")
            st.stop()

    with st.spinner("Generating your personalized regimen..."):
        try:
            from src.agents.regimen_generator import RegimenGenerator
            from src.agents.safety_guard import SafetyGuard

            generator = RegimenGenerator(api_key=anthropic_key)
            regimen = generator.generate(profile, evidence)

            guard = SafetyGuard()
            safety_report = guard.check(regimen, profile)

            # Append safety warnings to regimen contraindications
            if safety_report.modified_contraindications:
                regimen.contraindications.extend(safety_report.modified_contraindications)

            st.session_state["regimen"] = regimen
            st.session_state["profile"] = profile
            st.session_state["evidence"] = evidence
            st.session_state["safety"] = safety_report
            st.session_state["vision"] = vision_data
            st.session_state["xai"] = xai_result

        except Exception as e:
            st.error(f"Regimen generation failed: {e}")
            st.stop()

    st.success("✦ Regimen ready! See the 'Your regimen' tab.")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_result:
    if "regimen" not in st.session_state:
        st.info("Fill in your skin profile and click 'Generate' to see your personalized regimen.")
        st.stop()

    regimen = st.session_state["regimen"]
    profile = st.session_state["profile"]
    evidence = st.session_state["evidence"]
    safety = st.session_state["safety"]
    vision = st.session_state.get("vision")
    xai = st.session_state.get("xai")

    # ── Safety flags ──────────────────────────────────────────────────────────
    if safety.has_warnings:
        for flag in safety.flags:
            if flag.severity == "warning":
                st.markdown(
                    f'<div class="warning-box">⚠️ {flag.message}</div>',
                    unsafe_allow_html=True
                )

    # ── Profile summary ───────────────────────────────────────────────────────
    st.subheader("Profile summary")
    st.write(regimen.profile_summary)

    if vision:
        with st.expander("Vision analysis details"):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Detected skin type", vision.overall_skin_type)
                st.metric("Acne severity", vision.acne_severity)
                st.metric("Hyperpigmentation", vision.hyperpigmentation)
            with c2:
                st.metric("Redness level", vision.redness_level)
                st.metric("Visible pores", vision.visible_pores)
                st.metric("Fitzpatrick estimate", vision.fitzpatrick_estimate)
            st.caption(f"Vision confidence note: {vision.confidence_note}")

    # ── XAI heatmap (Prompt #12) ──────────────────────────────────────────────
    if xai and getattr(xai, "heatmap_base64", ""):
        with st.expander("AI Explanation Heatmap (LIME surrogate)"):
            st.warning(
                "APPROXIMATION: This heatmap uses MobileNetV2 (ImageNet) as a surrogate "
                "for GPT-4o Vision. It highlights regions that influenced the surrogate's "
                "skin classification and may not perfectly reflect GPT-4o's reasoning."
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Heatmap overlay**")
                st.markdown(
                    f'<img src="data:image/png;base64,{xai.heatmap_base64}" '
                    f'style="width:100%; border-radius:8px;">',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("**Top active regions**")
                for region in getattr(xai, "top_regions", []):
                    st.markdown(f"- {region}")
                if hasattr(xai, "confidence"):
                    st.metric("Surrogate confidence", f"{xai.confidence:.1%}")
                if hasattr(xai, "explanation_text"):
                    st.caption(xai.explanation_text)

    st.divider()

    # ── Routine display ───────────────────────────────────────────────────────
    def display_routine(steps: list, label: str) -> None:
        st.subheader(label)
        if not steps:
            st.caption("No steps for this routine.")
            return
        for step in steps:
            with st.expander(
                f"Step {step.step_number} — {step.product_type}",
                expanded=True,
            ):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    if step.active_ingredients:
                        st.markdown(f"**Actives:** {', '.join(step.active_ingredients)}")
                with c2:
                    st.markdown(f"**Concentration:** {step.concentration_range}")
                with c3:
                    ev = step.evidence_grade
                    label_class = f"evidence-{ev}"
                    st.markdown(
                        f'**Evidence:** <span class="{label_class}">Grade {ev}</span>',
                        unsafe_allow_html=True
                    )
                st.write(step.application_notes)
                if step.citations:
                    st.caption("📚 Sources: " + " · ".join(step.citations[:3]))

    col_am, col_pm = st.columns(2)
    with col_am:
        display_routine(regimen.am_routine, "☀️ Morning routine")
    with col_pm:
        display_routine(regimen.pm_routine, "🌙 Evening routine")

    if regimen.weekly_treatments:
        display_routine(regimen.weekly_treatments, "📅 Weekly treatments")

    st.divider()

    # ── Avoid + warnings ──────────────────────────────────────────────────────
    c_avoid, c_lifestyle = st.columns(2)
    with c_avoid:
        if regimen.ingredients_to_avoid:
            st.subheader("⛔ Ingredients to avoid")
            for item in regimen.ingredients_to_avoid:
                st.markdown(f"- {item}")
        if regimen.contraindications:
            st.subheader("⚠️ Contraindications")
            for item in regimen.contraindications:
                st.markdown(
                    f'<div class="warning-box">{item}</div>',
                    unsafe_allow_html=True
                )

    with c_lifestyle:
        if regimen.lifestyle_notes:
            st.subheader("💡 Lifestyle notes")
            for note in regimen.lifestyle_notes:
                st.markdown(f"- {note}")
        st.info(f"📅 Follow-up review recommended in **{regimen.follow_up_weeks} weeks**")

    st.divider()
    st.caption(regimen.disclaimer)

    # ── Evidence sources ──────────────────────────────────────────────────────
    with st.expander(f"📚 Scientific evidence used ({len(evidence)} sources)"):
        for i, chunk in enumerate(evidence, 1):
            grade = chunk.evidence_level
            st.markdown(
                f"**[{i}]** {chunk.title} "
                f"({chunk.year}) — *{chunk.journal}* — "
                f"<span class='evidence-{grade}'>Grade {grade}</span>",
                unsafe_allow_html=True,
            )
            if chunk.url:
                st.caption(f"🔗 {chunk.url}")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    export_data = json.dumps(regimen.model_dump(), indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download regimen (JSON)",
        data=export_data,
        file_name="my_skincare_regimen.json",
        mime="application/json",
    )
