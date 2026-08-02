from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from pi05_libero_repro.logiv.model import GroundAction


DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "logiv"
    / "prompts"
    / "pi05-subtasks-v1.json"
)


class PromptConfigurationError(ValueError):
    pass


class SubtaskPromptRenderer:
    """Render a grounded action, never the original multi-stage instruction."""

    def __init__(self, path: Path | str = DEFAULT_PROMPT_PATH) -> None:
        self.path = Path(path)
        raw = self.path.read_bytes()
        payload = json.loads(raw)
        self.prompt_version = str(payload["prompt_version"])
        self.templates: Mapping[str, str] = payload["templates"]
        self.overrides: Mapping[str, str] = payload.get("action_overrides", {})
        self.frontier_overrides: Mapping[str, str] = payload.get(
            "frontier_action_overrides", {}
        )
        self.phase_overrides: Mapping[str, Mapping[str, str]] = payload.get(
            "action_phase_overrides", {}
        )
        self.labels: Mapping[str, str] = payload.get("labels", {})
        self.suffix = str(payload["suffix"])
        self.config_hash = hashlib.sha256(raw).hexdigest()
        if not self.prompt_version or not self.suffix:
            raise PromptConfigurationError("prompt version and suffix must be nonempty")

    def _label(self, symbol: str) -> str:
        return self.labels.get(symbol, symbol.replace("_", " "))

    def _object_label(self, object_id: str, source_id: str | None) -> str:
        if object_id.startswith("moka_pot") and source_id is not None:
            if "right" in source_id:
                return "right moka pot"
            if "left" in source_id:
                return "left moka pot"
        return self._label(object_id)

    def render(self, action: GroundAction) -> str:
        override = self.overrides.get(action.pddl())
        if override is not None:
            return override
        try:
            template = self.templates[action.schema]
        except KeyError as error:
            raise PromptConfigurationError(
                f"no prompt template for action schema {action.schema}"
            ) from error

        arguments = action.arguments
        source_id = arguments[1] if action.schema in {
            "pick",
            "place-on",
            "place-in",
            "place-relative",
        } else None
        values = {
            "object": self._object_label(arguments[0], source_id) if arguments else "object",
            "source": self._label(source_id) if source_id is not None else "source",
            "target": self._label(arguments[2] if action.schema.startswith("place-") and not action.schema.startswith("place-held") else arguments[1]) if len(arguments) > 1 else "target",
            "access": self._label(arguments[-1]) if arguments else "access",
            "device": self._label(arguments[0]) if arguments else "device",
        }
        try:
            stage = template.format(**values)
        except (IndexError, KeyError) as error:
            raise PromptConfigurationError(
                f"invalid template for action schema {action.schema}"
            ) from error
        return f"{stage} {self.suffix}"

    def render_phase(self, action: GroundAction, phase: str) -> str:
        phases = self.phase_overrides.get(action.pddl(), {})
        if phase in phases:
            return str(phases[phase])
        return self.render(action)

    def render_frontier(self, action: GroundAction) -> str:
        """Render a shared policy instruction for a concurrently ready frontier."""
        return self.frontier_overrides.get(action.pddl(), self.render(action))

    def has_phase(self, action: GroundAction, phase: str) -> bool:
        return phase in self.phase_overrides.get(action.pddl(), {})
