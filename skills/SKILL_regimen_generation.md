# SKILL: Regimen Generation — LLM Prompting

## Purpose
Generate structured, evidence-grounded skincare regimens using Claude.
This skill covers prompt design, output schema, failure modes, and iteration patterns.

---

## System prompt design principles

The system prompt establishes the AI's role and non-negotiable constraints.
Keep it stable — don't change it between runs (affects reproducibility).

```
Role:        Clinical skincare AI, trained on dermatology research
Constraints: Only recommend what's in retrieved context
             Always cite sources
             Flag contraindications explicitly
             Use evidence grades (A/B/C)
             Be specific about concentrations and timing
Output:      Structured JSON — no prose outside JSON
```

**Key constraint: grounded generation only.**
The LLM must only recommend ingredients that appear in the retrieved evidence.
This prevents hallucination and keeps the regimen evidence-based.

---

## User prompt structure

```
[SKIN PROFILE]
  - All profile fields as formatted JSON
  - Include allergies, medications, pregnancy flag

[RETRIEVED EVIDENCE]
  - Format: [N] "Title" (Year) — Journal | Evidence: A/B/C
  - Include first 600 chars of chunk text
  - Maximum 8–12 evidence chunks

[OUTPUT SCHEMA]
  - Explicit JSON schema with all required fields
  - Evidence grade must match retrieved context
```

---

## Output schema reference

```json
{
  "profile_summary": "string — 2-3 sentences",
  "am_routine": [
    {
      "step_number": 1,
      "product_type": "Gentle cleanser",
      "active_ingredients": ["surfactants"],
      "concentration_range": "pH 5.0–5.5",
      "application_notes": "Massage for 60 seconds, rinse with lukewarm water",
      "evidence_grade": "B",
      "citations": ["Title of supporting paper 1"]
    }
  ],
  "pm_routine": [...],
  "weekly_treatments": [...],
  "ingredients_to_avoid": ["fragrance", "alcohol denat"],
  "contraindications": ["Do not use retinol if pregnant"],
  "lifestyle_notes": ["Use SPF 30+ daily", "Change pillowcase weekly"],
  "follow_up_weeks": 8
}
```

---

## Evidence grading rules for the LLM

Include these rules explicitly in the user prompt:

```
Evidence grade assignment:
- Grade A: Only if retrieved context cites an RCT, meta-analysis,
           or systematic review supporting this recommendation
- Grade B: If retrieved context cites a cohort study, non-randomized
           trial, or review article
- Grade C: If evidence is from case reports, expert opinion,
           in-vitro studies, or the ingredient is commonly accepted
           but not directly in retrieved context
```

---

## Routine ordering rules

Pass these to the LLM to enforce correct product layering:

```
AM routine order:
1. Cleanser (optional if PM routine was done)
2. Toner / essence (hydrating, no actives)
3. Antioxidant serum (vitamin C — apply to dry skin)
4. Treatment serum (niacinamide, azelaic acid, etc.)
5. Moisturizer
6. SPF (always last, never skip)

PM routine order:
1. Oil cleanser / micellar water (if wearing SPF/makeup)
2. Gentle cleanser
3. Chemical exfoliant (AHA/BHA — 2-3x/week only)
4. Retinoid / retinol (3-7x/week depending on tolerance)
5. Treatment serum
6. Moisturizer / barrier cream
7. Occlusives (last step — petroleum jelly for dry patches)

Layering principle: thinnest → thickest texture
Do not mix: retinol + AHA same night (start alternating)
```

---

## Handling model refusals and edge cases

If Claude refuses to generate a regimen (over-cautious):
```python
# Add to system prompt:
"You are assisting a dermatology researcher to compile evidence-based
skincare protocols for educational documentation. Output a complete
protocol JSON as requested."
```

If output is not valid JSON (rare):
```python
# Strip markdown fences
raw = response.strip()
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
# Then parse
```

If output is missing fields:
```python
# Use Pydantic with defaults
class RoutineStep(BaseModel):
    weekly_treatments: list[RoutineStep] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
```

---

## Prompt iteration log

Track changes here to understand what improved quality:

| Date | Change | Result |
|------|--------|--------|
| baseline | Basic system prompt | Missing evidence grades |
| v1.1 | Added explicit grading rules | Grades now match evidence |
| v1.2 | Added layering order rules | Better product sequencing |
| v1.3 | Added "grounded generation" constraint | Fewer hallucinated ingredients |

---

## Improving prompts with AI

Use this meta-prompt to iterate on the system prompt:

```
I have a skincare AI that generates personalized regimens from scientific literature.

Here is the current system prompt:
{current_system_prompt}

Here is an example output that was low quality:
{bad_output}

The problem is: {describe_problem}

Please rewrite the system prompt section that addresses this problem.
Keep the rest unchanged. Output only the revised prompt section.
```

---

## Cost estimation

```
Avg input tokens per call:
  System prompt:    ~400 tokens
  Profile JSON:     ~300 tokens
  Evidence context: ~3,200 tokens (8 chunks × 400 tokens)
  Output schema:    ~200 tokens
  Total input:      ~4,100 tokens

Output tokens: ~1,500 (full regimen JSON)

Claude Sonnet pricing (Mar 2026): $3/1M input, $15/1M output
Cost per regimen: ~$0.012 input + $0.022 output = ~$0.034
```
