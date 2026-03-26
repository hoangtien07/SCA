# SKILL INDEX — Skincare AI System

## Purpose of this file

This index allows an AI assistant (Claude, GPT-4o, etc.) to quickly
identify which skill file to read before helping with a specific task.

Read this file first, then read the specific SKILL file referenced.

---

## Skill registry

| Task                                                | Skill file                    | Key contents                                                |
| --------------------------------------------------- | ----------------------------- | ----------------------------------------------------------- |
| Collect papers from Semantic Scholar or PubMed      | `SKILL_data_collection.md`    | API endpoints, rate limits, query patterns, quality filters |
| Improve search queries for new skin conditions      | `SKILL_data_collection.md`    | Query templates, MeSH terms, prompt to expand queries       |
| Debug empty retrieval results                       | `SKILL_rag_pipeline.md`       | Hybrid retrieval algorithm, filter syntax, debug script     |
| Add new ingredients or conditions to taxonomy       | `config/skin_conditions.yaml` | Full taxonomy, interaction rules, evidence mapping          |
| Improve regimen generation quality                  | `SKILL_regimen_generation.md` | Prompt templates, output schema, iteration log              |
| Debug JSON parse errors from LLM                    | `SKILL_regimen_generation.md` | Fence stripping, Pydantic defaults, failure patterns        |
| Analyze skin photos more accurately                 | `SKILL_vision_analysis.md`    | Vision prompt, zone analysis, Fitzpatrick guide             |
| Handle privacy for image data                       | `SKILL_vision_analysis.md`    | In-memory processing, no-persistence rules                  |
| Understand system architecture                      | `docs/ARCHITECTURE.md`        | Full data flow, component swap guide, DB schema             |
| Evaluate before launch                              | `docs/EVALUATION.md`          | 30 test cases, eval script, RAGAS metrics, scoring rubric   |
| See full project roadmap, task status, dependencies | `SKILL_project_roadmap.md`    | All 20 tasks, dependency graph, actionable prompts per task |
| Plan next implementation step                       | `SKILL_project_roadmap.md`    | Status per task (DONE/IMPROVE/READY/PENDING), critical path |

---

## How to use these skills with an AI assistant

### When improving data collection:

```
Read skills/SKILL_data_collection.md first.
Then ask: "Help me add queries for [new condition] following the patterns
in this skill file."
```

### When debugging retrieval:

```
Read skills/SKILL_rag_pipeline.md first.
Paste output of the debug script.
Ask: "Retrieval scores are low for [condition]. What should I fix?"
```

### When the regimen output is wrong:

```
Read skills/SKILL_regimen_generation.md first.
Paste the bad output + what was wrong.
Ask: "Improve the prompt to fix [specific problem]."
```

### When adding a new feature:

```
Read docs/ARCHITECTURE.md first.
Ask: "I want to add [feature]. Where does it fit in the architecture
and what should I modify?"
```

---

## Project status tracker

Update this section as you progress:

| Milestone                             | Status      | Notes                                                               |
| ------------------------------------- | ----------- | ------------------------------------------------------------------- |
| Phase 0: Data collection script       | ✅ Complete | Collectors ready, needs execution                                   |
| Phase 0: Metadata taxonomy            | ✅ Complete | INCI synonyms, phototoxicity, concentration limits added            |
| Phase 0: Chunking pipeline            | ✅ Complete | Section-aware chunking + enriched headers added                     |
| Phase 0: Vector DB setup              | ✅ Complete | ChromaDB local + Qdrant scaffolded                                  |
| Phase 0: BM25 sparse index            | ✅ Complete | BM25Index + RRF fusion in src/pipeline/bm25_index.py                |
| Phase 0: Knowledge base indexed       | ⏳ Pending  | Run scripts after collection                                        |
| Phase 1: RAG retrieval code           | ✅ Complete | Hybrid dense+BM25, multi-query, cross-encoder rerank                |
| Phase 1: Regimen generation code      | ✅ Complete | Prompts extracted to config/prompts/                                |
| Phase 1: Vision analysis code         | ✅ Complete | GPT-4o integration done                                             |
| Phase 1: Skin questionnaire           | ✅ Complete | In app.py Streamlit form                                            |
| Phase 2: Streamlit MVP                | ✅ Complete | Full UI with questionnaire, photo, regimen display                  |
| Phase 2: Safety guardrails code       | ✅ Complete | 7 checks: pregnancy, drugs, phototoxicity, age, allergen, conc, sev |
| Phase 2: FastAPI backend              | ✅ Complete | 6 endpoints in src/api/ with tracing + citation checking            |
| Phase 2: Deployed to cloud            | ⏳ Pending  | No Dockerfile/CI yet                                                |
| Phase 3: RAGAS evaluation run         | ⏳ Pending  | Test cases in EVALUATION.md, script not written                     |
| Phase 3: Safety regression tests      | ✅ Complete | 16 test cases in tests/test_safety_regression.py                    |
| Phase 3: 10+ manual test cases passed | ⏳ Pending  |                                                                     |

---

## Prompts for common improvement tasks

### Expand knowledge base coverage

```
I am building a skincare AI knowledge base using the patterns in
skills/SKILL_data_collection.md.

My current query coverage is: {paste search_queries.yaml}

I want to add coverage for: {new condition or ingredient}

Generate 5 new search queries following the same patterns (mix of
Semantic Scholar natural language and PubMed MeSH-style queries).
Output as YAML list items ready to paste into search_queries.yaml.
```

### Debug low-quality regimen

```
I am using the regimen generation system described in
skills/SKILL_regimen_generation.md.

Here is the skin profile input:
{profile_json}

Here is the generated regimen (problem: {describe problem}):
{regimen_json}

Suggest specific prompt improvements to fix this issue.
Show the exact text change to make in the system or user prompt.
```

### Add new skin condition support

```
I want to add support for {new condition} to my skincare AI.
The taxonomy is in config/skin_conditions.yaml.

Help me:
1. Add the condition to the taxonomy with keywords, subtypes, severity levels
2. Generate 5 search queries to collect papers for this condition
3. Identify any known ingredient interactions specific to this condition
4. Add any pregnancy or safety warnings relevant to common treatments

Follow the existing YAML structure exactly.
```

### Improve safety guardrails

```
I want to add a new safety check to src/agents/safety_guard.py.

The check should catch: {describe the unsafe scenario}
Example case: {describe a specific profile + regimen that would trigger it}

Write the check method following the patterns in the existing _check_* methods.
Include the flag severity, message, and any modifications to contraindications.
```
