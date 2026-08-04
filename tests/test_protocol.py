from __future__ import annotations

import json
import random

import numpy as np
import pytest

from pi05_libero_repro.protocol import (
    BaseActionPrefixHasher,
    derive_episode_seed,
    EpisodeInvalid,
    EpisodeSeededClient,
    prepare_observation,
    run_episode,
    seed_episode_runtime,
)


class FakeImageTools:
    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def resize_with_pad(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        assert (height, width) == (224, 224)
        self.seen.append(image.copy())
        return image

    @staticmethod
    def convert_to_uint8(image: np.ndarray) -> np.ndarray:
        return image.astype(np.uint8)


def observation() -> dict[str, object]:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 20,
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([4.0, 5.0]),
        "nested": {"items": [{"value": 1}]},
    }


class FakeEnv:
    def __init__(self, succeed_on_policy_step: int | None = None) -> None:
        self.succeed_on_policy_step = succeed_on_policy_step
        self.actions: list[list[float]] = []
        self.events: list[str] = []
        self.done = False
        self.success_override: bool | None = None

    def reset(self) -> dict[str, np.ndarray]:
        self.events.append("reset")
        return observation()

    def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
        assert initial_state.tolist() == [9.0]
        self.events.append("set_init_state")
        return observation()

    def step(self, action: list[float]):
        self.actions.append(action)
        policy_step = max(0, len(self.actions) - 10)
        if self.succeed_on_policy_step is not None and policy_step >= self.succeed_on_policy_step:
            self.done = True
        return observation(), 0.0, self.done, {}

    def check_success(self) -> bool:
        return self.done if self.success_override is None else self.success_override


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, element: dict) -> dict[str, np.ndarray]:
        assert set(element) == {
            "observation/image",
            "observation/wrist_image",
            "observation/state",
            "prompt",
        }
        self.calls += 1
        return {"actions": np.full((10, 7), self.calls, dtype=np.float32)}


def test_prepare_observation_rotates_images_and_orders_state() -> None:
    tools = FakeImageTools()
    obs = observation()

    element, main_image = prepare_observation(obs, "prompt", tools)

    np.testing.assert_array_equal(tools.seen[0], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(tools.seen[1], obs["robot0_eye_in_hand_image"][::-1, ::-1])
    np.testing.assert_array_equal(main_image, obs["agentview_image"][::-1, ::-1])
    np.testing.assert_allclose(element["observation/state"], [1, 2, 3, 0, 0, 0, 4, 5])
    assert element["observation/state"].shape == (8,)
    assert element["prompt"] == "prompt"


def test_episode_seeded_client_numbers_requests_from_zero_without_mutating_input() -> None:
    class CapturingClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def infer(self, element: dict) -> dict[str, np.ndarray]:
            self.requests.append(element)
            return {
                "actions": np.zeros((1, 7), dtype=np.float32),
                "__logiv_rng__": {
                    "episode_seed": element["__logiv_episode_seed__"],
                    "inference_index": element["__logiv_inference_index__"],
                },
            }

    base = CapturingClient()
    client = EpisodeSeededClient(base, episode_seed=7008002)
    element = {"prompt": "test"}

    client.infer(element)
    client.infer(element)

    assert element == {"prompt": "test"}
    assert [request["__logiv_inference_index__"] for request in base.requests] == [0, 1]
    assert all(request["__logiv_episode_seed__"] == 7008002 for request in base.requests)


def test_episode_seed_derivation_is_namespace_separated_and_stable() -> None:
    policy = derive_episode_seed("policy", master_seed=7, task_id=8, episode_idx=0)
    simulator = derive_episode_seed(
        "simulator", master_seed=7, task_id=8, episode_idx=0
    )

    assert policy == 2668564155
    assert simulator != policy
    assert simulator == derive_episode_seed(
        "simulator", master_seed=7, task_id=8, episode_idx=0
    )
    with pytest.raises(ValueError, match="namespace"):
        derive_episode_seed("", master_seed=7, task_id=8, episode_idx=0)


def test_episode_runtime_seed_resets_numpy_and_environment() -> None:
    class SeededEnv:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        def seed(self, value: int) -> None:
            self.seeds.append(value)

    env = SeededEnv()
    seed_episode_runtime(env, 1806969158)
    first = np.random.random(3)
    seed_episode_runtime(env, 1806969158)
    second = np.random.random(3)

    assert env.seeds == [1806969158, 1806969158]
    np.testing.assert_array_equal(first, second)


def test_wait_replan_reset_order_and_success() -> None:
    env = FakeEnv(succeed_on_policy_step=7)
    client = FakeClient()

    outcome = run_episode(env, client, np.array([9.0]), "prompt", FakeImageTools())

    assert env.events == ["reset", "set_init_state"]
    assert env.actions[:10] == [[0.0] * 6 + [-1.0]] * 10
    assert client.calls == outcome.inference_requests == 2
    assert [float(action[0]) for action in outcome.actions] == [1, 1, 1, 1, 1, 2, 2]
    assert outcome.steps == 7
    assert outcome.success and outcome.done and outcome.check_success


def test_shadow_observer_cannot_change_or_abort_base_actions() -> None:
    baseline_env = FakeEnv(succeed_on_policy_step=7)
    shadow_env = FakeEnv(succeed_on_policy_step=7)
    seen_steps = []

    def hostile_observer(context):
        seen_steps.append(context.policy_step)
        context.observation["robot0_eef_pos"][0] = -999
        context.observation["nested"]["items"][0]["value"] = -999
        random.random()
        np.random.random()
        if context.last_action is not None:
            context.last_action[:] = -999
        context.pending_base_actions[:] = -999
        if context.policy_step == 3:
            raise RuntimeError("shadow failed")

    random.seed(1234)
    np.random.seed(1234)
    baseline = run_episode(
        baseline_env, FakeClient(), np.array([9.0]), "prompt", FakeImageTools()
    )
    baseline_python_draw = random.random()
    baseline_numpy_draw = np.random.random()

    random.seed(1234)
    np.random.seed(1234)
    ticks = iter(value * 0.01 for value in range(100))

    def rng_consuming_clock() -> float:
        random.random()
        np.random.random()
        return next(ticks)

    shadow = run_episode(
        shadow_env,
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        shadow_observer=hostile_observer,
        clock=rng_consuming_clock,
    )
    np.testing.assert_array_equal(np.asarray(shadow.actions), np.asarray(baseline.actions))
    assert shadow.steps == baseline.steps
    assert shadow.inference_requests == baseline.inference_requests
    assert (shadow.done, shadow.check_success) == (baseline.done, baseline.check_success)
    assert seen_steps == list(range(0, baseline.steps + 1))
    assert shadow.shadow_calls == baseline.steps + 1
    assert shadow.shadow_errors == 1
    assert shadow.shadow_wall_seconds == pytest.approx((baseline.steps + 1) * 0.01)
    assert shadow.shadow_parity_valid
    assert random.random() == baseline_python_draw
    assert np.random.random() == baseline_numpy_draw


def test_shadow_deepcopy_failure_is_contained_at_step_zero() -> None:
    class DeepcopyBomb:
        def __deepcopy__(self, memo):
            raise RuntimeError("no copying")

    class InitialBombEnv(FakeEnv):
        def step(self, action: list[float]):
            result = super().step(action)
            if len(self.actions) == 10:
                obs, reward, done, info = result
                obs["nested"] = DeepcopyBomb()
                return obs, reward, done, info
            return result

    baseline = run_episode(
        FakeEnv(succeed_on_policy_step=2),
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
    )
    shadow = run_episode(
        InitialBombEnv(succeed_on_policy_step=2),
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        shadow_observer=lambda context: None,
    )

    np.testing.assert_array_equal(np.asarray(shadow.actions), np.asarray(baseline.actions))
    assert (shadow.steps, shadow.inference_requests, shadow.done, shadow.check_success) == (
        baseline.steps,
        baseline.inference_requests,
        baseline.done,
        baseline.check_success,
    )
    assert shadow.shadow_errors == 1
    assert shadow.shadow_failure_records[0].policy_step == 0
    assert shadow.shadow_failure_records[0].stage == "INPUT_COPY"
    assert shadow.shadow_failure_records[0].reason == "RuntimeError"


def test_shadow_context_tracks_prefix_pending_actions_and_request_envelopes() -> None:
    class EchoingClient:
        def infer(self, element: dict) -> dict[str, np.ndarray | dict[str, int]]:
            return {
                "actions": np.full((10, 7), element["__logiv_inference_index__"] + 1, dtype=np.float32),
                "__logiv_rng__": {
                    "episode_seed": element["__logiv_episode_seed__"],
                    "inference_index": element["__logiv_inference_index__"],
                },
            }

    client = EpisodeSeededClient(
        EchoingClient(), episode_seed=7008002, policy_client_config_sha256="a" * 64
    )
    contexts = []
    outcome = run_episode(
        FakeEnv(succeed_on_policy_step=7),
        client,
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        shadow_observer=contexts.append,
        request_envelope_reader=client.request_envelope_reader,
    )

    assert len(contexts) == outcome.steps + 1
    initial = contexts[0]
    assert initial.pending_base_actions.shape == (0, 7)
    assert initial.base_action_response_size is None
    assert (initial.base_action_chunk_size, initial.pending_base_action_offset) == (0, 0)
    assert (
        initial.base_policy_request_count,
        initial.active_base_request_index,
        initial.next_base_request_index,
    ) == (0, None, 0)
    assert initial.base_action_prefix_sha256 == BaseActionPrefixHasher().update_and_hexdigest(None)
    assert json.loads(initial.next_base_replay_envelope_json) == {
        "episode_seed": 7008002,
        "inference_index": 0,
        "policy_client_config_sha256": "a" * 64,
        "version": "BaseRequestEnvelopeV1",
    }

    first = contexts[1]
    assert (
        first.base_policy_request_count,
        first.active_base_request_index,
        first.next_base_request_index,
        first.base_action_response_size,
        first.base_action_chunk_size,
        first.pending_base_action_offset,
    ) == (1, 0, 1, 10, 5, 1)
    assert first.pending_base_actions.shape == (4, 7)
    assert np.all(first.pending_base_actions == np.float32(1))
    assert json.loads(first.active_base_request_envelope_json)["inference_index"] == 0
    assert json.loads(first.next_base_replay_envelope_json)["inference_index"] == 1

    second_chunk = contexts[6]
    assert (
        second_chunk.base_policy_request_count,
        second_chunk.active_base_request_index,
        second_chunk.next_base_request_index,
        second_chunk.base_action_response_size,
        second_chunk.base_action_chunk_size,
        second_chunk.pending_base_action_offset,
    ) == (2, 1, 2, 10, 5, 1)
    assert second_chunk.pending_base_actions.shape == (4, 7)
    assert np.all(second_chunk.pending_base_actions == np.float32(2))

    for context in contexts:
        hasher = BaseActionPrefixHasher()
        for action in outcome.actions[: context.policy_step]:
            digest = hasher.update_and_hexdigest(action)
        assert context.base_action_prefix_sha256 == hasher.update_and_hexdigest(None)
        assert context.pending_base_actions.shape == (
            context.base_action_chunk_size - context.pending_base_action_offset,
            7,
        )


def test_episode_seeded_client_only_reads_acknowledged_envelopes_and_replays_from_frozen_config() -> None:
    class RejectingClient:
        def infer(self, element: dict) -> dict[str, object]:
            return {"actions": np.zeros((1, 7)), "__logiv_rng__": {}}

    client = EpisodeSeededClient(
        RejectingClient(), episode_seed=7, policy_client_config_sha256="b" * 64
    )
    with pytest.raises(EpisodeInvalid, match="RNG envelope"):
        client.infer({"prompt": "test"})

    assert client.request_envelope_reader(0, True) is None
    assert json.loads(client.request_envelope_reader(0, False))["inference_index"] == 0
    assert len(client.issued_request_envelopes) == 1


def test_generic_client_has_no_replay_envelope_support() -> None:
    contexts = []
    run_episode(
        FakeEnv(succeed_on_policy_step=1),
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        shadow_observer=contexts.append,
    )

    assert contexts[0].active_base_request_envelope_json is None
    assert contexts[0].next_base_replay_envelope_json is None


def test_inference_error_is_invalid() -> None:
    class BrokenClient:
        def infer(self, element: dict) -> dict:
            raise ConnectionError("closed")

    with pytest.raises(EpisodeInvalid, match="closed"):
        run_episode(FakeEnv(), BrokenClient(), np.array([9.0]), "prompt", FakeImageTools())


def test_malformed_action_chunk_is_invalid() -> None:
    class ShortClient:
        def infer(self, element: dict) -> dict[str, np.ndarray]:
            return {"actions": np.zeros((4, 7), dtype=np.float32)}

    with pytest.raises(EpisodeInvalid, match="action chunk shape"):
        run_episode(FakeEnv(), ShortClient(), np.array([9.0]), "prompt", FakeImageTools())


def test_timeout_is_a_valid_policy_failure() -> None:
    outcome = run_episode(
        FakeEnv(), FakeClient(), np.array([9.0]), "prompt", FakeImageTools(), max_steps=3
    )

    assert outcome.steps == 3
    assert not outcome.success and not outcome.done and not outcome.check_success


def test_success_predicate_disagreement_is_invalid() -> None:
    env = FakeEnv(succeed_on_policy_step=1)
    env.success_override = False

    with pytest.raises(EpisodeInvalid, match="success predicate disagreement"):
        run_episode(env, FakeClient(), np.array([9.0]), "prompt", FakeImageTools())


def test_base_success_is_rechecked_after_the_same_settling_barrier() -> None:
    class TransientSuccessEnv(FakeEnv):
        lost_success = False

        def step(self, action: list[float]):
            if self.done and action == [0.0] * 6 + [-1.0]:
                self.actions.append(action)
                self.done = False
                self.lost_success = True
                return observation(), 0.0, False, {}
            if self.lost_success:
                self.actions.append(action)
                return observation(), 0.0, False, {}
            return super().step(action)

    env = TransientSuccessEnv(succeed_on_policy_step=1)
    outcome = run_episode(
        env,
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        settling_steps=2,
    )

    assert outcome.done
    assert not outcome.check_success
    assert not outcome.success
    assert outcome.steps == 1
    assert len(outcome.replay_frames) == 3
    assert env.actions[-2:] == [[0.0] * 6 + [-1.0]] * 2
