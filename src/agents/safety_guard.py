"""
src/agents/safety_guard.py

Safety guardrail layer — runs AFTER regimen generation, BEFORE output.
Checks for:
  - Pregnancy contraindications
  - Known drug-cosmetic interactions (expanded: 7+ drug classes)
  - Ingredient conflict pairs
  - Severity-appropriate escalation (refer to dermatologist)
  - Phototoxicity warnings (must include SPF)
  - Age-tier safety (pediatric ≤15, geriatric ≥65)
  - Concentration limit validation (OTC maximums)

If issues found, modifies regimen warnings rather than blocking output entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── Load taxonomy once ────────────────────────────────────────────────────────
_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "config" / "skin_conditions.yaml"
with open(_TAXONOMY_PATH, encoding="utf-8") as f:
    _TAXONOMY = yaml.safe_load(f)

_PREGNANCY_AVOID = {i.lower() for i in _TAXONOMY.get("pregnancy_avoid", [])}
_INTERACTIONS = _TAXONOMY.get("known_interactions", {})
_PHOTOTOXICITY = _TAXONOMY.get("phototoxicity", {})
_CONCENTRATION_LIMITS = _TAXONOMY.get("concentration_limits", {})
_AGE_SAFETY = _TAXONOMY.get("age_safety", {})
_INCI_SYNONYMS = _TAXONOMY.get("inci_synonyms", {})


@dataclass
class SafetyFlag:
    severity: str                           # "warning" | "caution" | "info"
    message: str
    affected_ingredients: list[str] = field(default_factory=list)


@dataclass
class SafetyReport:
    flags: list[SafetyFlag] = field(default_factory=list)
    is_safe_to_show: bool = True            # False only for severe issues
    modified_contraindications: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.flags)

    def summary(self) -> str:
        if not self.flags:
            return "No safety concerns detected."
        return " | ".join(f.message for f in self.flags)


class SafetyGuard:
    """
    Checks a generated regimen against a user profile for safety issues.

    Usage:
        guard = SafetyGuard()
        report = guard.check(regimen=regimen, profile=profile)
        if report.has_warnings:
            # append report.modified_contraindications to regimen
    """

    def check(self, regimen, profile: dict) -> SafetyReport:
        """
        Run all safety checks.

        Args:
            regimen: Regimen Pydantic object from RegimenGenerator
            profile: Skin profile dict

        Returns:
            SafetyReport with flags and any modified warnings
        """
        report = SafetyReport()

        # Collect all ingredients in the regimen
        all_ingredients = self._collect_ingredients(regimen)

        # Run checks (order: most critical first)
        self._check_pregnancy(all_ingredients, profile, report)
        self._check_medication_interactions(all_ingredients, profile, report)
        self._check_ingredient_conflicts(all_ingredients, report)
        self._check_allergens(all_ingredients, profile, report)
        self._check_phototoxicity(all_ingredients, regimen, report)
        self._check_age_safety(all_ingredients, profile, report)
        self._check_concentration_limits(regimen, report)
        self._check_severity_escalation(profile, report)

        return report

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_pregnancy(
        self,
        ingredients: list[str],
        profile: dict,
        report: SafetyReport,
    ) -> None:
        if not profile.get("pregnancy", False):
            return

        flagged = [
            ing for ing in ingredients
            if any(avoid in ing.lower() for avoid in _PREGNANCY_AVOID)
        ]

        if flagged:
            report.flags.append(SafetyFlag(
                severity="warning",
                message=(
                    f"Pregnancy safety: {', '.join(flagged)} should be avoided "
                    "during pregnancy. Consult your OB-GYN or dermatologist."
                ),
                affected_ingredients=flagged,
            ))
            report.modified_contraindications.append(
                f"PREGNANCY WARNING: Avoid {', '.join(flagged)}"
            )

    def _check_ingredient_conflicts(
        self,
        ingredients: list[str],
        report: SafetyReport,
    ) -> None:
        ing_lower = [i.lower() for i in ingredients]

        for pair in _INTERACTIONS.get("avoid_together", []):
            a, b, reason = pair[0], pair[1], pair[2] if len(pair) > 2 else ""
            a_present = any(a.replace("_", " ") in ing for ing in ing_lower)
            b_present = any(b.replace("_", " ") in ing for ing in ing_lower)

            if a_present and b_present:
                report.flags.append(SafetyFlag(
                    severity="caution",
                    message=f"Ingredient conflict: {a} + {b} — {reason}",
                    affected_ingredients=[a, b],
                ))
                report.modified_contraindications.append(
                    f"Do not use {a} and {b} together in the same routine. {reason}"
                )

    def _check_allergens(
        self,
        ingredients: list[str],
        profile: dict,
        report: SafetyReport,
    ) -> None:
        """Match profile allergies against regimen ingredients using INCI synonym groups."""
        allergies = [a.lower() for a in profile.get("allergies", [])]
        if not allergies:
            return

        ing_lower = [i.lower() for i in ingredients]

        # Build reverse map: synonym → canonical name
        synonym_to_canonical: dict[str, str] = {}
        for canonical, synonyms in _INCI_SYNONYMS.items():
            canonical_l = canonical.lower().replace("_", " ")
            synonym_to_canonical[canonical_l] = canonical_l
            for syn in synonyms:
                synonym_to_canonical[syn.lower().replace("_", " ")] = canonical_l

        # Resolve each allergy to its canonical group
        allergy_groups: set[str] = set()
        for allergy in allergies:
            allergy_norm = allergy.replace("_", " ")
            canonical = synonym_to_canonical.get(allergy_norm)
            if canonical:
                allergy_groups.add(canonical)
            else:
                # Fallback: use allergy string directly for substring matching
                allergy_groups.add(allergy_norm)

        # Check each ingredient (resolved via synonyms) against allergy groups
        flagged: list[str] = []
        for ing in ing_lower:
            # Resolve ingredient to canonical
            matched_canonical = None
            for syn, canonical in synonym_to_canonical.items():
                if syn in ing:
                    matched_canonical = canonical
                    break

            if matched_canonical and matched_canonical in allergy_groups:
                flagged.append(ing)
            elif any(allergy in ing for allergy in allergy_groups):
                flagged.append(ing)

        if flagged:
            report.flags.append(SafetyFlag(
                severity="warning",
                message=(
                    f"Allergen alert: {', '.join(flagged)} may trigger a reaction "
                    f"based on reported allergies ({', '.join(allergies)}). "
                    "Consider alternatives or patch-test first."
                ),
                affected_ingredients=flagged,
            ))
            report.modified_contraindications.append(
                f"ALLERGEN: Avoid {', '.join(flagged)} — patient allergy reported"
            )

    def _check_severity_escalation(
        self,
        profile: dict,
        report: SafetyReport,
    ) -> None:
        """Flag cases where a dermatologist referral is strongly recommended."""
        acne_severity = profile.get("acne_severity", "none")
        concerns = [c.lower() for c in profile.get("concerns", [])]

        if acne_severity in ("severe", "cystic"):
            report.flags.append(SafetyFlag(
                severity="warning",
                message=(
                    "Severe or cystic acne detected. OTC actives may provide "
                    "limited results. Strongly recommend dermatologist consultation "
                    "for prescription options (adapalene, tretinoin, isotretinoin)."
                ),
            ))

        if any("psoriasis" in c or "lupus" in c for c in concerns):
            report.flags.append(SafetyFlag(
                severity="warning",
                message=(
                    "Autoimmune skin conditions detected (psoriasis/lupus). "
                    "Dermatologist management is essential — skincare alone is insufficient."
                ),
            ))

    def _check_medication_interactions(
        self,
        ingredients: list[str],
        profile: dict,
        report: SafetyReport,
    ) -> None:
        """Check topical ingredients against systemic medications using taxonomy rules."""
        medications = [m.lower() for m in profile.get("medications", [])]
        if not medications:
            return

        ing_lower = [i.lower() for i in ingredients]

        # ── Check against taxonomy drug_interactions ──────────────────────
        for rule in _INTERACTIONS.get("drug_interactions", []):
            drug_name = rule.get("drug", "").lower()
            avoid_list = [a.lower() for a in rule.get("avoid", [])]
            severity = rule.get("severity", "info")
            reason = rule.get("reason", "")

            # Check if patient is on this drug
            if not any(drug_name in med for med in medications):
                continue

            # Check if any avoid-listed ingredient is in the regimen
            flagged = []
            for avoid_ing in avoid_list:
                if any(avoid_ing.replace("_", " ") in i for i in ing_lower):
                    flagged.append(avoid_ing)

            if flagged:
                report.flags.append(SafetyFlag(
                    severity=severity,
                    message=f"{drug_name} interaction: avoid {', '.join(flagged)}. {reason}",
                    affected_ingredients=flagged,
                ))
                if severity == "warning":
                    report.modified_contraindications.append(
                        f"DRUG INTERACTION: {drug_name} — avoid {', '.join(flagged)}. {reason}"
                    )

    # ── New safety checks ─────────────────────────────────────────────────────

    def _check_phototoxicity(
        self,
        ingredients: list[str],
        regimen,
        report: SafetyReport,
    ) -> None:
        """Flag photosensitizing ingredients and verify SPF is included."""
        ing_lower = {i.lower() for i in ingredients}
        high_risk = {i.lower() for i in _PHOTOTOXICITY.get("high_risk", [])}
        moderate_risk = {i.lower() for i in _PHOTOTOXICITY.get("moderate_risk", [])}

        found_high = [i for i in ing_lower if any(h in i for h in high_risk)]
        found_moderate = [i for i in ing_lower if any(m in i for m in moderate_risk)]

        if not found_high and not found_moderate:
            return

        # Check if SPF is in AM routine
        am_steps = getattr(regimen, "am_routine", [])
        has_spf = any(
            "spf" in getattr(step, "product_type", "").lower()
            or any("spf" in ing.lower() or "sunscreen" in ing.lower()
                   for ing in getattr(step, "active_ingredients", []))
            for step in am_steps
        )

        if found_high and not has_spf:
            report.flags.append(SafetyFlag(
                severity="warning",
                message=(
                    f"Photosensitizing ingredients ({', '.join(found_high)}) require "
                    "daily broad-spectrum SPF ≥30. No sunscreen detected in AM routine."
                ),
                affected_ingredients=list(found_high),
            ))
            report.modified_contraindications.append(
                "PHOTOTOXICITY: Add SPF ≥30 to AM routine — photosensitizing actives present."
            )
        elif found_moderate and not has_spf:
            report.flags.append(SafetyFlag(
                severity="caution",
                message=(
                    f"Mildly photosensitizing ingredients ({', '.join(found_moderate)}). "
                    "SPF ≥30 strongly recommended."
                ),
                affected_ingredients=list(found_moderate),
            ))

    def _check_age_safety(
        self,
        ingredients: list[str],
        profile: dict,
        report: SafetyReport,
    ) -> None:
        """Apply age-tier restrictions from taxonomy."""
        age = profile.get("age")
        if age is None:
            return

        ing_lower = [i.lower() for i in ingredients]

        # Pediatric checks
        pediatric = _AGE_SAFETY.get("pediatric", {})
        if age <= pediatric.get("max_age", 15):
            avoid_list = [a.lower() for a in pediatric.get("avoid", [])]
            flagged = [
                a for a in avoid_list
                if any(a.replace("_", " ") in i for i in ing_lower)
            ]
            if flagged:
                report.flags.append(SafetyFlag(
                    severity="warning",
                    message=(
                        f"Pediatric safety (age {age}): {', '.join(flagged)} "
                        "not recommended for patients ≤15. "
                        + pediatric.get("note", "")
                    ),
                    affected_ingredients=flagged,
                ))
                report.modified_contraindications.append(
                    f"PEDIATRIC: Avoid {', '.join(flagged)} for age {age}"
                )

        # Geriatric checks
        geriatric = _AGE_SAFETY.get("geriatric", {})
        if age >= geriatric.get("min_age", 65):
            caution_list = [c.lower() for c in geriatric.get("caution", [])]
            flagged = [
                c for c in caution_list
                if any(c.replace("_", " ") in i for i in ing_lower)
            ]
            if flagged:
                report.flags.append(SafetyFlag(
                    severity="caution",
                    message=(
                        f"Geriatric safety (age {age}): use caution with "
                        f"{', '.join(flagged)}. {geriatric.get('note', '')}"
                    ),
                    affected_ingredients=flagged,
                ))

    def _check_concentration_limits(
        self,
        regimen,
        report: SafetyReport,
    ) -> None:
        """Validate that recommended concentrations don't exceed OTC maximums."""
        all_steps = (
            getattr(regimen, "am_routine", [])
            + getattr(regimen, "pm_routine", [])
            + getattr(regimen, "weekly_treatments", [])
        )

        for step in all_steps:
            conc_range = getattr(step, "concentration_range", "")
            if not conc_range:
                continue

            for ingredient in getattr(step, "active_ingredients", []):
                ing_key = ingredient.lower().replace(" ", "_")
                limits = _CONCENTRATION_LIMITS.get(ing_key, {})
                otc_max = limits.get("otc_max", "")

                if not otc_max:
                    continue

                # Extract numeric percentage from concentration_range and otc_max
                conc_num = self._extract_percentage(conc_range)
                max_num = self._extract_percentage(otc_max)

                if conc_num is not None and max_num is not None and conc_num > max_num:
                    note = limits.get("note", "")
                    report.flags.append(SafetyFlag(
                        severity="warning",
                        message=(
                            f"Concentration limit: {ingredient} at {conc_range} exceeds "
                            f"OTC maximum of {otc_max}. {note}"
                        ),
                        affected_ingredients=[ingredient],
                    ))
                    report.modified_contraindications.append(
                        f"CONCENTRATION: Reduce {ingredient} to ≤{otc_max}"
                    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collect_ingredients(self, regimen) -> list[str]:
        ingredients = []
        all_steps = (
            getattr(regimen, "am_routine", [])
            + getattr(regimen, "pm_routine", [])
            + getattr(regimen, "weekly_treatments", [])
        )
        for step in all_steps:
            ingredients.extend(getattr(step, "active_ingredients", []))
        return list(set(ingredients))

    @staticmethod
    def _extract_percentage(text: str) -> float | None:
        """Extract the first percentage number from a string like '10%' or '0.5–1%'."""
        import re
        # Match patterns like "10%", "0.5%", "10–15%", "≤2%"
        match = re.search(r"(\d+\.?\d*)%", text)
        if match:
            return float(match.group(1))
        return None
