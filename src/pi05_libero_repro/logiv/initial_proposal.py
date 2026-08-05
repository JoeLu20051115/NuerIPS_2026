from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Callable, Generic, TypeVar

from pi05_libero_repro.logiv.model import GoalMode, ProposalPackage


T = TypeVar("T")


class InitialProposalStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class InitialProposalResult(Generic[T]):
    status: InitialProposalStatus
    provider: str
    request_count: int
    elapsed_seconds: float
    package: ProposalPackage | None
    validation: T | None
    reason: str | None

    @property
    def intervention_enabled(self) -> bool:
        return (
            self.status is InitialProposalStatus.ACCEPTED
            and self.package is not None
            and self.validation is not None
        )


def run_initial_proposal(
    provider: Any,
    *,
    provider_name: str,
    task_id: int,
    epoch_id: int,
    goal_mode: GoalMode,
    validator: Callable[[ProposalPackage], T],
    clock: Callable[[], float] = time.perf_counter,
) -> InitialProposalResult[T]:
    started = clock()
    try:
        package = provider.propose(task_id, epoch_id, goal_mode)
        validation = validator(package)
    except Exception as error:
        return InitialProposalResult(
            status=InitialProposalStatus.REJECTED,
            provider=provider_name,
            request_count=1,
            elapsed_seconds=clock() - started,
            package=None,
            validation=None,
            reason=f"{type(error).__name__}: {error}",
        )
    return InitialProposalResult(
        status=InitialProposalStatus.ACCEPTED,
        provider=provider_name,
        request_count=1,
        elapsed_seconds=clock() - started,
        package=package,
        validation=validation,
        reason=None,
    )
