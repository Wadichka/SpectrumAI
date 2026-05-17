"""Объяснимость предсказаний (Grad-CAM и аналоги, §6.10 главы 6)."""

from app.interpretation.gradcam import GradCAM1D, compute_gradcam, default_target_layer

__all__ = ["GradCAM1D", "compute_gradcam", "default_target_layer"]
