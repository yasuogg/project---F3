"""UI perturbations for robustness evaluation.

We monkey-patch the env so each reset injects a small CSS jitter via JavaScript.
Implemented as a gym wrapper that adds a Playwright eval after reset.
"""
from __future__ import annotations
import random
import gymnasium as gym


CSS_JITTER_JS = """
(() => {
  const css = document.createElement('style');
  css.textContent = `
    button, input, a, select, label {
      font-size: %(fs)dpx !important;
      letter-spacing: %(ls).2fpx !important;
      transform: translateY(%(ty)dpx);
    }
    body { filter: contrast(%(c).2f) brightness(%(b).2f); }
  `;
  document.head.appendChild(css);
})();
"""


class CSSJitterWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int = 0):
        super().__init__(env)
        self.rng = random.Random(seed)

    def _inject(self):
        try:
            page = self.env.unwrapped.page  # browsergym exposes the Playwright Page
            js = CSS_JITTER_JS % dict(
                fs=self.rng.randint(12, 20),
                ls=self.rng.uniform(-0.5, 1.5),
                ty=self.rng.randint(-3, 3),
                c=self.rng.uniform(0.85, 1.15),
                b=self.rng.uniform(0.9, 1.1),
            )
            page.evaluate(js)
        except Exception:
            pass

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._inject()
        return out
