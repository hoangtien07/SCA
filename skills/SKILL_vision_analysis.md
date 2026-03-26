# SKILL: Vision Analysis — Skin Image Analysis with GPT-4o

## Purpose
Analyze skin photographs to detect conditions, skin type, and severity levels.
Output structured data that merges with the questionnaire for a complete skin profile.

---

## Model choice (March 2026)

**Recommended: GPT-4o Vision**
- Best general-purpose multimodal model
- No fine-tuning required
- API-accessible, production-ready
- Cost: ~$0.01–0.02 per image (high detail mode)

**Alternatives considered:**
- PanDerm: SOTA dermatology model, not yet public API
- Claude 3.5 Sonnet Vision: Good alternative, similar quality
- Gemini 1.5 Pro Vision: Comparable, use if OpenAI unavailable

---

## Image quality requirements

Communicate these to users before capture:

```
✅ Good image:
  - Natural daylight or neutral indoor lighting
  - No filters, no makeup (or minimal)
  - Straight-on face angle (not selfie-mode distortion)
  - 1–3 feet distance
  - Clean, dry skin for most accurate analysis

❌ Will degrade accuracy:
  - Flash photography (creates hotspots, hides texture)
  - Instagram/Snapchat filters
  - Heavy foundation or makeup
  - Extreme angles (side profile, upward)
  - Low-resolution or blurry images
```

---

## Prompt design

### System prompt role
```
Dermatology AI assistant specialized in skin image analysis.
Focus on visible conditions only — do not speculate beyond what's visible.
Always note confidence limitations.
Return valid JSON only.
```

### What to analyze

```json
{
  "overall_skin_type": "oily|dry|combination|normal|sensitive",
  "fitzpatrick_estimate": "I|II|III|IV|V|VI",
  "zones": [
    {
      "zone": "forehead|nose|cheeks|chin|under_eyes",
      "conditions": ["acne", "dryness", "..."],
      "severity": "none|mild|moderate|severe"
    }
  ],
  "detected_conditions": ["flat list of all conditions"],
  "texture_notes": "string",
  "hyperpigmentation": "none|mild|moderate|severe",
  "visible_pores": "minimal|moderate|enlarged",
  "redness_level": "none|mild|moderate|severe",
  "acne_severity": "none|mild|moderate|severe|cystic",
  "estimated_age_range": "25-35",
  "confidence_note": "limitations note",
  "raw_description": "2-3 sentence summary"
}
```

---

## Merging vision + questionnaire

Priority rules when data conflicts:

```
Self-reported data WINS for:
  - Skin type (user knows their history)
  - Allergies (user knows their reactions)
  - Medications (user knows what they take)
  - Pregnancy status

Vision data ADDS for:
  - Additional detected conditions not self-reported
  - Objective severity (user may underreport)
  - Zone-specific analysis
  - Visible texture details
```

Merge function:
```python
concerns = list(dict.fromkeys(
    questionnaire["concerns"] + vision.detected_conditions
))
# dict.fromkeys preserves order + deduplicates
# questionnaire concerns appear first (higher priority)
```

---

## Fitzpatrick scale reference

| Type | Phenotype | Sun reaction | Notes |
|------|-----------|-------------|-------|
| I | Very fair, red/blonde hair, blue eyes | Always burns, never tans | High melanoma risk |
| II | Fair, blonde hair, blue/green eyes | Usually burns, rarely tans | |
| III | Medium, any hair/eye color | Sometimes burns, always tans | |
| IV | Olive/light brown | Rarely burns, tans easily | PIH risk higher |
| V | Brown skin | Very rarely burns | Higher PIH risk |
| VI | Dark brown/black | Never burns | Highest PIH risk, avoid aggressive exfoliation |

**Clinical relevance:**
- Types IV–VI: higher post-inflammatory hyperpigmentation (PIH) risk
- Avoid aggressive AHA for types V–VI without dermatologist guidance
- Vitamin C + niacinamide safe for all types
- Hydroquinone: use with caution for types IV–VI

---

## Common failure modes

| Problem | Cause | Fix |
|---------|-------|-----|
| GPT-4o refuses to analyze | Privacy concern triggered | Rephrase prompt, emphasize educational purpose |
| "Unable to determine" on all zones | Poor image quality | Prompt user to retake |
| Wrong skin type detected | Lighting artifacts | Weight questionnaire self-report higher |
| Misses cystic acne | No depth perception in 2D | Ask user to self-report severity |
| Overestimates Fitzpatrick | Warm lighting | Note this as "estimate" not diagnosis |

---

## Improving vision prompts with AI

```
I am analyzing skin images with GPT-4o for a skincare AI.
Current prompt detects [list what it detects].

It misses: {problem_type}
Example image description: {describe image that failed}

Suggest 3 improvements to the vision prompt specifically for detecting
{problem_type} more accurately. Keep changes minimal and targeted.
```

---

## Privacy and data handling

```python
# DO: Process and discard immediately
image_bytes = uploaded_file.read()
analysis = analyzer.analyze_bytes(image_bytes)
del image_bytes                    # don't store raw image

# DO: Store only the structured analysis
profile["vision_detected_conditions"] = analysis.detected_conditions

# DON'T: Store raw face images on server
# DON'T: Log base64 image data
# DON'T: Send to any service other than the configured vision model
```

GDPR/Privacy considerations:
- Face images are biometric data in many jurisdictions
- For MVP: process in-memory, never persist to disk or DB
- For production: explicit consent required, right-to-deletion mechanism needed
- Consider on-device processing (future: CoreML, MediaPipe) to avoid server upload
