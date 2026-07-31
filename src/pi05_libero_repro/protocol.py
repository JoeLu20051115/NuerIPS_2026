from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, List, Tuple

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MODEL_IMAGE_SIZE = 224


class EpisodeInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    done: bool
    check_success: bool
    steps: int
    inference_requests: int
    first_frame: np.ndarray
    replay_frames: List[np.ndarray]
    actions: List[np.ndarray]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise EpisodeInvalid(f"quaternion shape: {quat.shape} != (4,)")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


def prepare_observation(obs: dict, prompt: str, image_tools: Any) -> Tuple[dict, np.ndarray]:
    main_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    main_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(main_image, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise EpisodeInvalid(f"observation state must be finite shape (8,), got {state.shape}")
    return (
        {
            "observation/image": main_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": str(prompt),
        },
        main_image,
    )


def run_episode(
    env: Any,
    client: Any,
    initial_state: np.ndarray,
    prompt: str,
    image_tools: Any,
    max_steps: int = 520,
    wait_steps: int = 10,
    replan_steps: int = 5,
) -> EpisodeOutcome:
    if max_steps <= 0 or wait_steps < 0 or replan_steps <= 0:
        raise ValueError("max_steps and replan_steps must be positive; wait_steps must be nonnegative")

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(wait_steps):
            obs, _, _, _ = env.step(list(LIBERO_DUMMY_ACTION))

        action_plan = deque()
        replay_frames = []
        executed_actions = []
        inference_requests = 0
        done = False

        for _ in range(max_steps):
            element, main_image = prepare_observation(obs, prompt, image_tools)
            replay_frames.append(main_image)

            if not action_plan:
                response = client.infer(element)
                action_chunk = np.asarray(response["actions"])
                inference_requests += 1
                if (
                    action_chunk.ndim != 2
                    or action_chunk.shape[0] < replan_steps
                    or action_chunk.shape[1] < 7
                ):
                    raise EpisodeInvalid(
                        f"action chunk shape {action_chunk.shape} cannot supply {replan_steps}x7 actions"
                    )
                if not np.isfinite(action_chunk).all():
                    raise EpisodeInvalid("action chunk contains non-finite values")
                action_plan.extend(action_chunk[:replan_steps, :7])

            action = np.asarray(action_plan.popleft(), dtype=np.float64)
            executed_actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break

        check_success = bool(env.check_success())
        done = bool(done)
        if done != check_success:
            raise EpisodeInvalid(f"success predicate disagreement: done={done}, check_success={check_success}")
        return EpisodeOutcome(
            success=done,
            done=done,
            check_success=check_success,
            steps=len(executed_actions),
            inference_requests=inference_requests,
            first_frame=replay_frames[0],
            replay_frames=replay_frames,
            actions=executed_actions,
        )
    except EpisodeInvalid:
        raise
    except Exception as error:
        raise EpisodeInvalid(str(error)) from error
