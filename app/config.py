from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "configs"
        self.models = self._load_yaml("models.yaml").get("models", {})
        self.eval_tasks = self._load_yaml("eval_tasks.yaml")
        self.runtime = self._load_yaml("runtime.yaml").get("runtime", {})

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

