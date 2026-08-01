#!/usr/bin/env python3
"""Serve OpenPI with an auditable, episode-local JAX RNG reset protocol."""

from __future__ import annotations

import dataclasses
import logging
import socket
from typing import Any

import jax
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config
import tyro


SEED_KEY = "__logiv_episode_seed__"
INDEX_KEY = "__logiv_inference_index__"
RESPONSE_KEY = "__logiv_rng__"
PROTOCOL_VERSION = 1


class EpisodeSeededPolicy:
    """Reset the wrapped JAX policy at request index zero for each episode."""

    def __init__(self, wrapped: Any) -> None:
        if not hasattr(wrapped, "_rng"):
            raise TypeError("episode RNG protocol requires a JAX OpenPI Policy")
        self._wrapped = wrapped
        self._active_seed: int | None = None
        self._next_index = 0

    def infer(self, observation: dict) -> dict:
        payload = dict(observation)
        has_seed = SEED_KEY in payload
        has_index = INDEX_KEY in payload
        seed = payload.pop(SEED_KEY, None)
        index = payload.pop(INDEX_KEY, None)
        # Preserve the repository's original evaluator path.  It deliberately
        # remains unseeded/continuous and receives no seeded-protocol receipt.
        if not has_seed and not has_index:
            return self._wrapped.infer(payload)
        if has_seed != has_index:
            raise ValueError("incomplete episode RNG envelope")
        if not isinstance(seed, int) or not isinstance(index, int):
            raise ValueError("missing episode RNG envelope")
        if seed < 0 or seed >= 2**32 or index < 0:
            raise ValueError("invalid episode RNG envelope")
        if index == 0:
            self._wrapped._rng = jax.random.key(seed)
            self._active_seed = seed
            self._next_index = 0
            logging.info("Reset policy RNG for episode seed %d", seed)
        if seed != self._active_seed or index != self._next_index:
            raise ValueError(
                f"non-monotonic episode RNG envelope: seed={seed}, index={index}, "
                f"active_seed={self._active_seed}, expected_index={self._next_index}"
            )
        result = self._wrapped.infer(payload)
        result[RESPONSE_KEY] = {"episode_seed": seed, "inference_index": index}
        self._next_index += 1
        return result


@dataclasses.dataclass
class Args:
    port: int = 8000
    policy_config: str = "pi05_libero"
    policy_dir: str = ""


def main(args: Args) -> None:
    if not args.policy_dir:
        raise ValueError("--policy-dir is required")
    trained = policy_config.create_trained_policy(
        config.get_config(args.policy_config), args.policy_dir
    )
    metadata = dict(trained.metadata)
    metadata["logiv_episode_rng_protocol"] = PROTOCOL_VERSION
    hostname = socket.gethostname()
    logging.info("Creating episode-seeded server (host: %s, seed protocol: v%d)", hostname, PROTOCOL_VERSION)
    websocket_policy_server.WebsocketPolicyServer(
        policy=EpisodeSeededPolicy(trained),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
