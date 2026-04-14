#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json

import imageio.v2 as imageio
import numpy as np

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package is required for AgiBot judged eval") from exc


class FinalFrameJudge:
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        buf = io.BytesIO()
        imageio.imwrite(buf, frame, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def judge(
        self,
        task: str,
        initial_frame: np.ndarray,
        real_final_frame: np.ndarray,
        predicted_final_frame: np.ndarray,
    ) -> dict:
        prompt = (
            "You are judging robot task completion from visual evidence.\n"
            "You will see three images:\n"
            "1. initial real observation\n"
            "2. real final observation from the demonstration (reference goal state)\n"
            "3. predicted final frame from the model rollout\n\n"
            f"Task: {task}\n\n"
            "Return strict JSON only with schema:\n"
            "{\"task_progress\": number, \"rule_success\": boolean, \"reason\": string}\n\n"
            "Rules:\n"
            "- task_progress must be in [0,1].\n"
            "- rule_success means: the final state satisfies the task goal, yes or no.\n"
            "- Judge the model's predicted final frame against the task goal, using the real final frame only as a helpful reference for the intended successful end state.\n"
            "- Be slightly lenient to small pose offsets if the task intent is clearly satisfied.\n"
            "- 1.0 means the predicted final frame clearly satisfies the task.\n"
            "- 0.7-0.9 means the task is almost fully achieved and should count as success in a lenient setting.\n"
            "- 0.4-0.6 means partial completion or the key object reached the target region but final relation is not fully convincing.\n"
            "- 0.0-0.3 means the task goal is not achieved.\n"
            "- If the final state visibly satisfies the task goal, set rule_success=true even if task_progress is below 1.0.\n"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful robot-evaluation judge. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Initial real observation:"},
                        {"type": "image_url", "image_url": {"url": self._frame_to_data_url(initial_frame)}},
                        {"type": "text", "text": "Real final observation from the demonstration:"},
                        {"type": "image_url", "image_url": {"url": self._frame_to_data_url(real_final_frame)}},
                        {"type": "text", "text": "Predicted final frame from the model rollout:"},
                        {"type": "image_url", "image_url": {"url": self._frame_to_data_url(predicted_final_frame)}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            start = content.find("{")
            end = content.rfind("}")
            payload = json.loads(content[start : end + 1] if start != -1 and end != -1 and end > start else content)
        except Exception:
            payload = {"task_progress": 0.0, "rule_success": False, "reason": f"non_json_response: {content[:200]}"}
        progress = float(np.clip(float(payload.get("task_progress", 0.0)), 0.0, 1.0))
        rule_success = bool(payload.get("rule_success", False))
        return {"task_progress": progress, "rule_success": rule_success, "reason": str(payload.get("reason", ""))}
