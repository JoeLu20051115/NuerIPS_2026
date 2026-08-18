from __future__ import annotations

import numpy as np
import pytest

from pi05_libero_repro.protocol import (
    BaseActionPrefixHasher,
    EpisodeInvalid,
    prepare_observation,
    run_episode,
)


class ImageTools:
    def resize_with_pad(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        assert (height, width) == (224, 224)
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


class Environment:
    def __init__(self) -> None:
        self.actions: list[list[float]] = []

    def reset(self) -> dict[str, np.ndarray]:
        return observation()

    def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
        assert initial_state.tolist() == [9.0]
        return observation()

    def step(self, action: list[float]):
        self.actions.append(action)
        done = len(self.actions) >= 12
        return observation(), 0.0, done, {}

    def check_success(self) -> bool:
        return len(self.actions) >= 12


class Policy:
    def infer(self, element: dict) -> dict[str, np.ndarray]:
        assert element["prompt"] == "move the object"
        return {"actions": np.ones((10, 7), dtype=np.float32)}


def test_prepare_observation_rotates_images_and_orders_state() -> None:
    obs = observation()
    element, main_image = prepare_observation(obs, "move the object", ImageTools())

    np.testing.assert_array_equal(main_image, obs["agentview_image"][::-1, ::-1])
    np.testing.assert_allclose(element["observation/state"], [1, 2, 3, 0, 0, 0, 4, 5])
    assert element["observation/state"].shape == (8,)


def test_run_episode_executes_policy_and_stops_on_native_completion() -> None:
    outcome = run_episode(
        Environment(),
        Policy(),
        np.array([9.0]),
        "move the object",
        ImageTools(),
        wait_steps=1,
        replan_steps=5,
        settling_steps=0,
    )

    assert outcome.success
    assert outcome.done
    assert outcome.check_success
    assert outcome.inference_requests == 3
    assert outcome.steps == 11


def test_run_episode_rejects_malformed_action_chunks() -> None:
    class BrokenPolicy:
        def infer(self, element: dict) -> dict[str, np.ndarray]:
            return {"actions": np.zeros((2, 6), dtype=np.float32)}

    with pytest.raises(EpisodeInvalid):
        run_episode(
            Environment(),
            BrokenPolicy(),
            np.array([9.0]),
            "move the object",
            ImageTools(),
            wait_steps=1,
            settling_steps=0,
        )


def test_action_prefix_hash_is_stable_and_order_sensitive() -> None:
    first = BaseActionPrefixHasher()
    second = BaseActionPrefixHasher()
    action = np.arange(7, dtype=np.float32)

    assert first.update_and_hexdigest(action) == second.update_and_hexdigest(action.copy())
    assert first.update_and_hexdigest(action + 1) != second.update_and_hexdigest(action)
