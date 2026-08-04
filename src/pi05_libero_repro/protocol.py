from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import random
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
class BaseRequestEnvelopeV1:
    episode_seed: int
    inference_index: int
    policy_client_config_sha256: str

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "episode_seed": self.episode_seed,
                "inference_index": self.inference_index,
                "policy_client_config_sha256": self.policy_client_config_sha256,
                "version": "BaseRequestEnvelopeV1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def derive_episode_seed(
    namespace: str, *, master_seed: int, task_id: int, episode_idx: int
) -> int:
    """Derive a stable, namespace-separated uint32 seed for one episode."""

    if not namespace:
        raise ValueError("seed namespace must be nonempty")
    if any(not isinstance(value, int) for value in (master_seed, task_id, episode_idx)):
        raise TypeError("master_seed, task_id, and episode_idx must be integers")
    if task_id < 0 or episode_idx < 0:
        raise ValueError("task_id and episode_idx must be nonnegative")
    payload = f"LOGIV-{namespace}-seed-v1:{master_seed}:{task_id}:{episode_idx}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_episode_runtime(env: Any, episode_seed: int) -> None:
    """Reset all evaluator-side RNGs before an episode reset."""

    if not isinstance(episode_seed, int) or not 0 <= episode_seed < 2**32:
        raise ValueError("episode_seed must fit uint32")
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    env.seed(episode_seed)


class EpisodeSeededClient:
    """Attach a deterministic episode-local RNG envelope to policy requests."""

    def __init__(
        self,
        client: Any,
        *,
        episode_seed: int,
        policy_client_config_sha256: str | None = None,
    ) -> None:
        if episode_seed < 0 or episode_seed >= 2**32:
            raise ValueError("episode_seed must fit uint32")
        self._client = client
        self.episode_seed = int(episode_seed)
        self.inference_index = 0
        self.policy_client_config_sha256 = policy_client_config_sha256
        self._issued_request_envelopes: list[str] = []
        self._acknowledged_request_indexes: set[int] = set()

    @property
    def issued_request_envelopes(self) -> tuple[str, ...]:
        return tuple(self._issued_request_envelopes)

    def _render_request_envelope(self, index: int) -> str | None:
        if self.policy_client_config_sha256 is None:
            return None
        return BaseRequestEnvelopeV1(
            episode_seed=self.episode_seed,
            inference_index=index,
            policy_client_config_sha256=self.policy_client_config_sha256,
        ).canonical_json()

    def request_envelope_reader(self, index: int, require_issued: bool) -> str | None:
        if not isinstance(index, int) or index < 0:
            return None
        if not require_issued:
            return self._render_request_envelope(index)
        if (
            index >= len(self._issued_request_envelopes)
            or index not in self._acknowledged_request_indexes
        ):
            return None
        return self._issued_request_envelopes[index]

    def infer(self, element: dict) -> dict:
        request = dict(element)
        request["__logiv_episode_seed__"] = self.episode_seed
        request["__logiv_inference_index__"] = self.inference_index
        envelope = self._render_request_envelope(self.inference_index)
        if envelope is not None:
            self._issued_request_envelopes.append(envelope)
        result = self._client.infer(request)
        expected = {
            "episode_seed": self.episode_seed,
            "inference_index": self.inference_index,
        }
        if result.get("__logiv_rng__") != expected:
            raise EpisodeInvalid("policy server did not honor episode RNG envelope")
        if envelope is not None:
            self._acknowledged_request_indexes.add(self.inference_index)
        self.inference_index += 1
        return result


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
    request_envelope_reader: Callable[[int, bool], str | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EpisodeOutcome:
    if max_steps <= 0 or wait_steps < 0 or replan_steps <= 0 or settling_steps < 0:
        raise ValueError(
            "max_steps and replan_steps must be positive; "
            "wait_steps and settling_steps must be nonnegative"
        )

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(wait_steps):
            obs, _, _, _ = env.step(list(LIBERO_DUMMY_ACTION))

        action_plan = deque()
        replay_frames = []
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
        done = False

        def call_shadow(observation: Mapping[str, Any], action: np.ndarray | None, policy_step: int) -> None:
            nonlocal shadow_calls, shadow_errors, shadow_wall_seconds, shadow_parity_valid
            if shadow_observer is None:
                return
            shadow_calls += 1
            shadow_started = None
            python_rng_state = None
            numpy_rng_state = None
            stage = "RNG_CAPTURE"

            def record_failure(failure_stage: str, error: Exception) -> None:
                nonlocal shadow_errors
                shadow_errors += 1
                shadow_failure_records.append(
                    ShadowFailureRecord(
                        policy_step=policy_step,
                        stage=failure_stage,
                        reason=type(error).__name__[:128],
                    )
                )

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
                record_failure(stage, error)
            finally:
                try:
                    if shadow_started is not None:
                        shadow_wall_seconds += clock() - shadow_started
                except Exception as error:
                    record_failure("CLOCK_END", error)
                try:
                    if python_rng_state is not None:
                        random.setstate(python_rng_state)
                except Exception as error:
                    record_failure("PYTHON_RNG_RESTORE", error)
                    shadow_parity_valid = False
                try:
                    if numpy_rng_state is not None:
                        np.random.set_state(numpy_rng_state)
                except Exception as error:
                    record_failure("NUMPY_RNG_RESTORE", error)
                    shadow_parity_valid = False

        call_shadow(obs, None, 0)

        for _ in range(max_steps):
            element, main_image = prepare_observation(obs, prompt, image_tools)
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
            call_shadow(obs, action, len(executed_actions))
            if done:
                break

        done = bool(done)
        for _ in range(settling_steps):
            settling_observation, _, _, _ = env.step(list(LIBERO_DUMMY_ACTION))
            _, settling_frame = prepare_observation(
                settling_observation, prompt, image_tools
            )
            replay_frames.append(settling_frame)
        check_success = bool(env.check_success())
        if settling_steps == 0 and done != check_success:
            raise EpisodeInvalid(f"success predicate disagreement: done={done}, check_success={check_success}")
        return EpisodeOutcome(
            success=check_success,
            done=done,
            check_success=check_success,
            steps=len(executed_actions),
            inference_requests=inference_requests,
            first_frame=replay_frames[0],
            replay_frames=replay_frames,
            actions=executed_actions,
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
