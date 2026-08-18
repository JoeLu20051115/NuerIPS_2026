from __future__ import annotations

from pathlib import Path

import scripts.eval_logiv_libero as evaluator_script


def test_evaluator_has_no_gpt4o_planning_or_repair_runtime_path() -> None:
    source = Path(evaluator_script.__file__).read_text()

    assert "gpt4o_planning" not in source
    assert "Gpt4oProposalProvider" not in source
    assert "Gpt4oRepairOperator" not in source
    assert "gpt-4o-certified-dag-remainder" not in source
    assert "gpt4o_client=episode_gpt4o_client" in source
    assert "_symbolic_record_accounting(" in source
