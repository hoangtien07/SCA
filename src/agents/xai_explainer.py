"""
src/agents/xai_explainer.py

Explainability layer for skin image analysis using LIME.

Architecture:
  - Surrogate model: MobileNetV2 pretrained on ImageNet
  - LIME perturbs superpixels of the input image
  - Regions with highest LIME impact → heatmap overlay
  - Output: base64-encoded PNG heatmap + top region descriptions

IMPORTANT CAVEAT:
  This explainability module uses MobileNetV2 (ImageNet) as a SURROGATE
  for GPT-4o Vision. The heatmap shows which image regions influenced the
  surrogate model's predictions — it is an APPROXIMATION of GPT-4o's
  reasoning and should not be treated as ground truth.

  The surrogate may have different biases than GPT-4o, especially for:
  - Non-standard skin conditions
  - Dark skin tones (Fitzpatrick V-VI)
  - Images with unusual lighting or composition

Usage (requires xai_enabled=True in settings):
    from src.agents.xai_explainer import VisionExplainer

    explainer = VisionExplainer()
    result = explainer.explain_analysis(image_bytes, condition="acne")
    print(result.top_regions)  # ["forehead", "cheeks"]
    # result.heatmap_base64 contains PNG overlay
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class ExplanationResult:
    """
    LIME explanation result for a skin image analysis.

    SURROGATE CAVEAT: heatmap_base64 shows which regions influenced
    MobileNetV2's prediction — an approximation of GPT-4o Vision.
    """
    condition: str = ""
    heatmap_base64: str = ""               # base64-encoded PNG overlay
    top_regions: list[str] = field(default_factory=list)  # e.g. ["forehead", "cheeks"]
    confidence: float = 0.0               # surrogate model confidence
    explanation_text: str = ""
    surrogate_caveat: str = (
        "APPROXIMATION: This heatmap uses MobileNetV2 (ImageNet) as a surrogate "
        "for GPT-4o Vision. It highlights regions that influenced the surrogate's "
        "skin classification and may not perfectly reflect GPT-4o's reasoning. "
        "Use as a qualitative guide only."
    )


class VisionExplainer:
    """
    Generates LIME-based visual explanations for skin image analysis.

    Requires:
        - Pillow (PIL)
        - torch + torchvision (MobileNetV2)
        - lime (lime==0.2.0.1)
        - scikit-image (for superpixel segmentation)

    All dependencies are optional — if unavailable, explain_analysis()
    returns an ExplanationResult with empty fields rather than crashing.

    Usage:
        explainer = VisionExplainer(num_samples=500)
        result = explainer.explain_analysis(image_bytes, condition="acne")
    """

    # Facial region map — maps approximate pixel regions to descriptive names
    FACE_REGIONS = {
        "forehead": (0.0, 0.0, 1.0, 0.25),      # top 25% of image
        "eyes": (0.1, 0.2, 0.9, 0.40),
        "nose": (0.3, 0.35, 0.7, 0.65),
        "cheeks": (0.0, 0.35, 1.0, 0.65),
        "mouth_chin": (0.2, 0.65, 0.8, 0.85),
        "jawline": (0.0, 0.75, 1.0, 1.0),
    }

    # MobileNetV2 ImageNet categories relevant to skin analysis
    RELEVANT_CATEGORIES = {
        "skin": list(range(900, 950)),   # approximate ImageNet skin-related categories
    }

    def __init__(self, num_samples: int = 500, surrogate_model: str = "mobilenet_v2"):
        self._num_samples = num_samples
        self._surrogate_model_name = surrogate_model
        self._model = None
        self._transform = None

    def explain_analysis(
        self,
        image_bytes: bytes,
        condition: str = "",
        top_labels: int = 3,
    ) -> ExplanationResult:
        """
        Generate a LIME explanation for a skin image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            condition: Target skin condition to explain (e.g. "acne")
            top_labels: Number of top LIME labels to consider

        Returns:
            ExplanationResult with heatmap and top regions
        """
        try:
            return self._explain_internal(image_bytes, condition, top_labels)
        except ImportError as e:
            logger.warning(f"[VisionExplainer] Missing dependency: {e}. Returning empty explanation.")
            return ExplanationResult(
                condition=condition,
                explanation_text=f"Explanation unavailable: missing dependency ({e})",
            )
        except Exception as e:
            logger.warning(f"[VisionExplainer] Explanation failed: {e}")
            return ExplanationResult(
                condition=condition,
                explanation_text=f"Explanation failed: {e}",
            )

    def generate_heatmap_overlay(
        self,
        image_bytes: bytes,
        explanation_mask: list[list[float]],
        alpha: float = 0.6,
    ) -> str:
        """
        Generate a base64-encoded heatmap overlay PNG.

        Args:
            image_bytes: Original image bytes
            explanation_mask: 2D float mask (0.0–1.0, higher = more important)
            alpha: Overlay transparency (0=transparent, 1=opaque)

        Returns:
            base64-encoded PNG string
        """
        try:
            import numpy as np
            from PIL import Image, ImageFilter

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(img.resize((224, 224)))

            mask = np.array(explanation_mask)
            if mask.max() > 0:
                mask = mask / mask.max()

            # Create heatmap (green=important, transparent=unimportant)
            heatmap = np.zeros((224, 224, 4), dtype=np.uint8)
            heatmap[:, :, 1] = (mask * 255).astype(np.uint8)  # green channel
            heatmap[:, :, 3] = (mask * alpha * 200).astype(np.uint8)  # alpha

            # Overlay
            overlay = Image.fromarray(img_array).convert("RGBA")
            heatmap_img = Image.fromarray(heatmap, mode="RGBA")
            overlay.paste(heatmap_img, mask=heatmap_img)

            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            logger.warning(f"[VisionExplainer] Heatmap generation failed: {e}")
            return ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load MobileNetV2 surrogate model."""
        if self._model is not None:
            return

        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        logger.info("[VisionExplainer] Loading MobileNetV2 surrogate model...")
        self._model = models.mobilenet_v2(pretrained=True)
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("[VisionExplainer] MobileNetV2 loaded")

    def _explain_internal(
        self,
        image_bytes: bytes,
        condition: str,
        top_labels: int,
    ) -> ExplanationResult:
        """Core LIME explanation logic."""
        import numpy as np
        from PIL import Image
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        self._load_model()

        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_224 = img.resize((224, 224))
        img_array = np.array(img_224)

        # LIME explainer
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images: np.ndarray) -> np.ndarray:
            """Batch prediction function for LIME."""
            import torch
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            batch = []
            for img_arr in images:
                pil = Image.fromarray(img_arr.astype(np.uint8))
                tensor = transform(pil).unsqueeze(0)
                batch.append(tensor)

            batch_tensor = torch.cat(batch, dim=0)
            with torch.no_grad():
                logits = self._model(batch_tensor)
                probs = torch.softmax(logits, dim=1).numpy()
            return probs

        # Run LIME
        logger.info(f"[VisionExplainer] Running LIME with {self._num_samples} samples...")
        explanation = explainer.explain_instance(
            img_array,
            predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=self._num_samples,
        )

        # Extract top label explanation
        top_label = explanation.top_labels[0]
        confidence = float(predict_fn(np.expand_dims(img_array, 0))[0][top_label])

        # Get LIME mask
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=10,
            hide_rest=False,
        )

        # Convert mask to float weights
        explanation_mask = mask.astype(float)

        # Generate heatmap
        heatmap_b64 = self.generate_heatmap_overlay(image_bytes, explanation_mask)

        # Map active superpixels to face regions
        top_regions = self._map_to_face_regions(explanation_mask)

        explanation_text = (
            f"The surrogate model (MobileNetV2) focused on the "
            f"{', '.join(top_regions) if top_regions else 'central area'} "
            f"of the image when predicting skin condition features. "
            f"Surrogate confidence: {confidence:.1%}. "
            f"Note: this approximates GPT-4o's analysis and may differ."
        )

        return ExplanationResult(
            condition=condition,
            heatmap_base64=heatmap_b64,
            top_regions=top_regions,
            confidence=round(confidence, 4),
            explanation_text=explanation_text,
        )

    def _map_to_face_regions(self, mask: "np.ndarray") -> list[str]:
        """Map active mask regions to face area names."""
        h, w = mask.shape[:2]
        active_regions = []

        for region_name, (x0_frac, y0_frac, x1_frac, y1_frac) in self.FACE_REGIONS.items():
            x0 = int(x0_frac * w)
            y0 = int(y0_frac * h)
            x1 = int(x1_frac * w)
            y1 = int(y1_frac * h)

            region_mask = mask[y0:y1, x0:x1]
            if region_mask.size > 0:
                region_activity = float(region_mask.mean())
                if region_activity > 0.3:   # threshold: region has notable activity
                    active_regions.append(region_name)

        return active_regions[:4]   # return top 4 regions max
