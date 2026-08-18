#!/usr/bin/env python3
"""Serve an OpenPI policy for LOGIV."""

from __future__ import annotations

import dataclasses

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config
import tyro


@dataclasses.dataclass
class Args:
    port: int = 8000
    policy_config: str = "pi05_libero"
    policy_dir: str = ""


def main(args: Args) -> None:
    if not args.policy_dir:
        raise ValueError("--policy-dir is required")
    policy = policy_config.create_trained_policy(
        config.get_config(args.policy_config), args.policy_dir
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=dict(policy.metadata),
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
