from __future__ import annotations

import numpy as np
import pytest

from pi05_libero_repro.protocol import (
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


def observation() -> dict[str, np.ndarray]:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 20,
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([4.0, 5.0]),
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
