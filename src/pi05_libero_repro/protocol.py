from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Callable, List, Mapping, Tuple

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MODEL_IMAGE_SIZE = 224


class EpisodeInvalid(RuntimeError):
    pass


class BaseActionPrefixHasher:
    """Hash executed Base actions with a stable, binary-safe stream."""

    _DOMAIN = b"LOGIV_BASE_ACTION_PREFIX_V1"

    def __init__(self) -> None:
        self._digest = hashlib.sha256(self._DOMAIN)

    def update_and_hexdigest(self, action: np.ndarray | None) -> str:
        if action is None:
            return self._digest.hexdigest()
        array = np.ascontiguousarray(np.asarray(action))
        dtype = array.dtype.str.encode("utf-8")
        shape = b"".join(int(dimension).to_bytes(8, "big") for dimension in array.shape)
        payload = (
            len(dtype).to_bytes(8, "big")
            + dtype
            + len(array.shape).to_bytes(8, "big")
            + shape
            + array.nbytes.to_bytes(8, "big")
            + array.tobytes(order="C")
        )
        updated = self._digest.copy()
        updated.update(payload)
        self._digest = updated
        return self._digest.hexdigest()


@dataclass(frozen=True)
class ShadowStepContext:
    observation: Mapping[str, Any]
    last_action: np.ndarray | None
    policy_step: int
    base_policy_request_count: int
    active_base_request_index: int | None
    next_base_request_index: int
    active_base_request_envelope_json: str | None
    next_base_replay_envelope_json: str | None
    base_action_response_size: int | None
    base_action_chunk_size: int
    pending_base_action_offset: int
    pending_base_actions: np.ndarray
    base_action_prefix_sha256: str


@dataclass(frozen=True)
class ShadowSettlingContext:
    observation: Mapping[str, Any]
    policy_step: int
    settling_step: int
    settling_steps: int


@dataclass(frozen=True)
class ShadowFailureRecord:
    policy_step: int
    stage: str
    reason: str


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
    intervention_requested: bool = False
    discarded_pending_actions: int = 0
    final_observation: dict[str, Any] | None = None
    shadow_calls: int = 0
    shadow_errors: int = 0
    shadow_failure_records: tuple[ShadowFailureRecord, ...] = ()
    shadow_wall_seconds: float = 0.0
    shadow_parity_valid: bool = True


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
    settling_steps: int = 0,
    *,
    shadow_observer: Callable[[ShadowStepContext], None] | None = None,
    shadow_settling_observer: Callable[[ShadowSettlingContext], None] | None = None,
    request_envelope_reader: Callable[[int, bool], str | None] | None = None,
    capture_replay_frames: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    intervention_monitor: Callable[[dict, np.ndarray, int], bool] | None = None,
    shadow_initialization_guard: Callable[[], None] | None = None,
) -> EpisodeOutcome:
    if max_steps <= 0 or wait_steps < 0 or replan_steps <= 0 or settling_steps < 0:
        raise ValueError(
            "max_steps and replan_steps must be positive; "
            "wait_steps and settling_steps must be nonnegative"
        )

    try:
        done = False
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(wait_steps):
            obs, _, done, _ = env.step(list(LIBERO_DUMMY_ACTION))
            if done:
                break

        action_plan = deque()
        replay_frames = []
        first_frame: np.ndarray | None = None
        executed_actions = []
        inference_requests = 0
        shadow_calls = 0
        shadow_errors = 0
        shadow_failure_records: list[ShadowFailureRecord] = []
        shadow_wall_seconds = 0.0
        shadow_parity_valid = True
        current_response_size: int | None = None
        current_chunk_size = 0
        current_chunk_offset = 0
        current_chunk_dtype = np.dtype(np.float64)
        current_chunk_request_index: int | None = None
        shadow_action_prefix = BaseActionPrefixHasher()
        intervention_requested = False
        discarded_pending_actions = 0

        def record_shadow_failure(
            policy_step: int, failure_stage: str, error: Exception
        ) -> None:
            nonlocal shadow_errors
            shadow_errors += 1
            shadow_failure_records.append(
                ShadowFailureRecord(
                    policy_step=policy_step,
                    stage=failure_stage,
                    reason=type(error).__name__[:128],
                )
            )

        def call_shadow(observation: Mapping[str, Any], action: np.ndarray | None, policy_step: int) -> None:
            nonlocal shadow_calls, shadow_errors, shadow_wall_seconds, shadow_parity_valid
            if shadow_observer is None:
                return
            shadow_calls += 1
            shadow_started = None
            python_rng_state = None
            numpy_rng_state = None
            stage = "RNG_CAPTURE"

            try:
                python_rng_state = random.getstate()
                numpy_rng_state = np.random.get_state()
                stage = "CLOCK_START"
                shadow_started = clock()
                stage = "ACTION_HASH"
                copied_action = None if action is None else np.array(action, copy=True)
                action_prefix_sha256 = shadow_action_prefix.update_and_hexdigest(copied_action)
                stage = "INPUT_COPY"
                copied_observation = copy.deepcopy(observation)
                pending_actions = (
                    np.stack(tuple(action_plan), axis=0)
                    if action_plan
                    else np.empty((0, 7), dtype=current_chunk_dtype)
                )
                stage = "ENVELOPE_READ"
                active_envelope = (
                    None
                    if request_envelope_reader is None or current_chunk_request_index is None
                    else request_envelope_reader(current_chunk_request_index, True)
                )
                next_envelope = (
                    None
                    if request_envelope_reader is None
                    else request_envelope_reader(inference_requests, False)
                )
                stage = "OBSERVER_ESCAPE"
                shadow_observer(
                    ShadowStepContext(
                        observation=copied_observation,
                        last_action=copied_action,
                        policy_step=policy_step,
                        base_policy_request_count=inference_requests,
                        active_base_request_index=current_chunk_request_index,
                        next_base_request_index=inference_requests,
                        active_base_request_envelope_json=active_envelope,
                        next_base_replay_envelope_json=next_envelope,
                        base_action_response_size=current_response_size,
                        base_action_chunk_size=current_chunk_size,
                        pending_base_action_offset=current_chunk_offset,
                        pending_base_actions=np.array(pending_actions, copy=True),
                        base_action_prefix_sha256=action_prefix_sha256,
                    )
                )
            except Exception as error:
                record_shadow_failure(policy_step, stage, error)
            finally:
                try:
                    if shadow_started is not None:
                        shadow_wall_seconds += clock() - shadow_started
                except Exception as error:
                    record_shadow_failure(policy_step, "CLOCK_END", error)
                try:
                    if python_rng_state is not None:
                        random.setstate(python_rng_state)
                except Exception as error:
                    record_shadow_failure(policy_step, "PYTHON_RNG_RESTORE", error)
                    shadow_parity_valid = False
                try:
                    if numpy_rng_state is not None:
                        np.random.set_state(numpy_rng_state)
                except Exception as error:
                    record_shadow_failure(policy_step, "NUMPY_RNG_RESTORE", error)
                    shadow_parity_valid = False

        def call_settling_shadow(
            observation: Mapping[str, Any],
            *,
            policy_step: int,
            settling_step: int,
        ) -> None:
            nonlocal shadow_calls, shadow_wall_seconds, shadow_parity_valid
            if shadow_settling_observer is None:
                return
            shadow_calls += 1
            shadow_started = None
            python_rng_state = None
            numpy_rng_state = None
            stage = "SETTLING_RNG_CAPTURE"
            try:
                python_rng_state = random.getstate()
                numpy_rng_state = np.random.get_state()
                stage = "SETTLING_CLOCK_START"
                shadow_started = clock()
                stage = "SETTLING_INPUT_COPY"
                copied_observation = copy.deepcopy(observation)
                stage = "SETTLING_OBSERVER_ESCAPE"
                shadow_settling_observer(
                    ShadowSettlingContext(
                        observation=copied_observation,
                        policy_step=policy_step,
                        settling_step=settling_step,
                        settling_steps=settling_steps,
                    )
                )
            except Exception as error:
                record_shadow_failure(policy_step, stage, error)
            finally:
                try:
                    if shadow_started is not None:
                        shadow_wall_seconds += clock() - shadow_started
                except Exception as error:
                    record_shadow_failure(policy_step, "SETTLING_CLOCK_END", error)
                try:
                    if python_rng_state is not None:
                        random.setstate(python_rng_state)
                except Exception as error:
                    record_shadow_failure(
                        policy_step, "SETTLING_PYTHON_RNG_RESTORE", error
                    )
                    shadow_parity_valid = False
                try:
                    if numpy_rng_state is not None:
                        np.random.set_state(numpy_rng_state)
                except Exception as error:
                    record_shadow_failure(
                        policy_step, "SETTLING_NUMPY_RNG_RESTORE", error
                    )
                    shadow_parity_valid = False

        if done:
            _, first_frame = prepare_observation(obs, prompt, image_tools)
            if capture_replay_frames:
                replay_frames.append(first_frame)
        else:
            call_shadow(obs, None, 0)
            if shadow_initialization_guard is not None:
                shadow_initialization_guard()

        for _ in range(0 if done else max_steps):
            element, main_image = prepare_observation(obs, prompt, image_tools)
            if first_frame is None:
                first_frame = main_image
            if capture_replay_frames:
                replay_frames.append(main_image)

            if not action_plan:
                current_chunk_request_index = inference_requests
                response = client.infer(element)
                action_chunk = np.asarray(response["actions"])
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
                current_response_size = action_chunk.shape[0]
                action_chunk = action_chunk[:replan_steps, :7]
                action_plan.extend(action_chunk)
                inference_requests += 1
                current_chunk_size = action_chunk.shape[0]
                current_chunk_offset = 0
                current_chunk_dtype = action_chunk.dtype

            action = np.asarray(action_plan.popleft())
            executed_actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            current_chunk_offset += 1
            if done:
                break
            call_shadow(obs, action, len(executed_actions))
            if intervention_monitor is not None and intervention_monitor(
                obs, action.copy(), len(executed_actions)
            ):
                intervention_requested = True
                discarded_pending_actions = len(action_plan)
                action_plan.clear()
                break

        done = bool(done)
        for settling_index in range(
            0 if done or intervention_requested else settling_steps
        ):
            settling_observation, _, settling_done, _ = env.step(
                list(LIBERO_DUMMY_ACTION)
            )
            obs = settling_observation
            if settling_done:
                done = True
                break
            _, settling_frame = prepare_observation(
                settling_observation, prompt, image_tools
            )
            if capture_replay_frames:
                replay_frames.append(settling_frame)
            call_settling_shadow(
                settling_observation,
                policy_step=len(executed_actions),
                settling_step=settling_index + 1,
            )
        check_success = bool(env.check_success())
        return EpisodeOutcome(
            success=done or check_success,
            done=done,
            check_success=check_success,
            steps=len(executed_actions),
            inference_requests=inference_requests,
            first_frame=first_frame,
            replay_frames=replay_frames,
            actions=executed_actions,
            intervention_requested=intervention_requested,
            discarded_pending_actions=discarded_pending_actions,
            final_observation=obs,
            shadow_calls=shadow_calls,
            shadow_errors=shadow_errors,
            shadow_failure_records=tuple(shadow_failure_records),
            shadow_wall_seconds=shadow_wall_seconds,
            shadow_parity_valid=shadow_parity_valid,
        )
    except EpisodeInvalid:
        raise
    except Exception as error:
        raise EpisodeInvalid(str(error)) from error
