from pi05_libero_repro.logiv.initial_proposal import (
    InitialProposalStatus,
    run_initial_proposal,
)
from pi05_libero_repro.logiv.model import GoalMode


class Provider:
    provider = "scripted-vlm-v1"

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = 0

    def propose(self, task_id, epoch_id, goal_mode):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.value


def test_provider_failure_is_a_fail_open_rejection():
    provider = Provider(error=RuntimeError("offline"))
    result = run_initial_proposal(
        provider,
        provider_name=provider.provider,
        task_id=8,
        epoch_id=0,
        goal_mode=GoalMode.METADATA_ASSISTED,
        validator=lambda package: package,
        clock=iter((10.0, 10.25)).__next__,
    )
    assert result.status is InitialProposalStatus.REJECTED
    assert result.package is None and result.validation is None
    assert result.request_count == 1
    assert result.elapsed_seconds == 0.25
    assert result.reason == "RuntimeError: offline"
    assert not result.intervention_enabled


def test_validator_failure_discards_the_uncertified_package():
    provider = Provider(value=object())

    def reject(package):
        raise ValueError("VAL rejected")

    result = run_initial_proposal(
        provider,
        provider_name=provider.provider,
        task_id=8,
        epoch_id=0,
        goal_mode=GoalMode.METADATA_ASSISTED,
        validator=reject,
    )
    assert result.status is InitialProposalStatus.REJECTED
    assert result.package is None and result.validation is None
    assert result.reason == "ValueError: VAL rejected"


def test_accepted_result_keeps_package_and_certification():
    package = object()
    certificate = object()
    result = run_initial_proposal(
        Provider(value=package),
        provider_name="scripted-vlm-v1",
        task_id=8,
        epoch_id=0,
        goal_mode=GoalMode.METADATA_ASSISTED,
        validator=lambda value: certificate,
    )
    assert result.status is InitialProposalStatus.ACCEPTED
    assert result.package is package
    assert result.validation is certificate
    assert result.request_count == 1
    assert result.reason is None
    assert result.intervention_enabled
