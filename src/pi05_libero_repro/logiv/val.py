from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, FrozenSet, Sequence, Tuple

from pi05_libero_repro.logiv.domain import (
    DomainError,
    FixedDomain,
    render_domain_pddl,
    render_problem_pddl,
    validate_state,
)
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    GroundAction,
    TaskProblem,
)


WRAPPER_VERSION = "logiv-val-wrapper-v1"
_VALID_MARKER = re.compile(r"^Plan valid$", re.MULTILINE)
_INVALID_MARKERS = (
    "Plan invalid",
    "Goal not satisfied",
    "Plan failed to execute",
)


class SignedTraceStatus(str, Enum):
    VALID = "VALID"
    ACTION_PRECONDITION_FAILURE = "ACTION_PRECONDITION_FAILURE"
    FINAL_GOAL_FAILURE = "FINAL_GOAL_FAILURE"
    PLAN_GROUNDING_INCOMPLETE = "PLAN_GROUNDING_INCOMPLETE"
    PLAN_SCHEMA_ERROR = "PLAN_SCHEMA_ERROR"
    DOMAIN_ERROR = "DOMAIN_ERROR"
    PARSER_ERROR = "PARSER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    VALIDATION_ERROR = "VALIDATION_ERROR"


@dataclass(frozen=True)
class TraceObligation:
    consumer_index: int | None
    consumer: str
    fact: Fact
    positive: bool


@dataclass(frozen=True)
class SignedTraceResult:
    status: SignedTraceStatus
    final_true: FrozenSet[Fact]
    final_false: FrozenSet[Fact]
    obligations: Tuple[TraceObligation, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PlanCertificate:
    certificate_hash: str
    component_hashes: Tuple[Tuple[str, str], ...]
    context: ContextEnvelope
    val_binary: str
    val_binary_sha256: str
    val_version: str
    wrapper_version: str
    timeout_seconds: float
    forbidden_retry_keys: Tuple[str, ...]
    retry_ledger_version: int


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    trace: SignedTraceResult
    certificate: PlanCertificate | None
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    reason: str | None = None


def _trace_result(
    status: SignedTraceStatus,
    true_facts: set[Fact],
    false_facts: set[Fact],
    *,
    obligations: Sequence[TraceObligation] = (),
    reason: str | None = None,
) -> SignedTraceResult:
    return SignedTraceResult(
        status=status,
        final_true=frozenset(true_facts),
        final_false=frozenset(false_facts),
        obligations=tuple(obligations),
        reason=reason,
    )


def run_signed_trace(
    problem: TaskProblem,
    plan: Sequence[GroundAction],
    domain: FixedDomain | None = None,
) -> SignedTraceResult:
    domain = domain or FixedDomain()
    true_facts = set(problem.initial_state)
    false_facts = set(problem.initial_false)
    try:
        validate_state(problem, frozenset(true_facts))
    except DomainError as error:
        return _trace_result(
            SignedTraceStatus.DOMAIN_ERROR, true_facts, false_facts, reason=str(error)
        )
    if true_facts & false_facts:
        return _trace_result(
            SignedTraceStatus.DOMAIN_ERROR,
            true_facts,
            false_facts,
            reason="initial TRUE/FALSE conflict",
        )

    for index, action in enumerate(plan):
        try:
            expected = domain.ground(problem, action.schema, action.arguments)
        except DomainError as error:
            return _trace_result(
                SignedTraceStatus.PLAN_SCHEMA_ERROR,
                true_facts,
                false_facts,
                reason=str(error),
            )
        if expected != action:
            return _trace_result(
                SignedTraceStatus.PLAN_SCHEMA_ERROR,
                true_facts,
                false_facts,
                reason=f"ground action does not match fixed schema at index {index}",
            )

        unknown_positive = action.preconditions - true_facts - false_facts
        unknown_negative = action.negative_preconditions - true_facts - false_facts
        if unknown_positive or unknown_negative:
            unknown = sorted(unknown_positive | unknown_negative)
            return _trace_result(
                SignedTraceStatus.PLAN_GROUNDING_INCOMPLETE,
                true_facts,
                false_facts,
                reason="unknown required literals: " + ", ".join(str(fact) for fact in unknown),
            )

        missing_positive = action.preconditions & false_facts
        violated_negative = action.negative_preconditions & true_facts
        if missing_positive or violated_negative:
            obligations = [
                TraceObligation(index, action.pddl(), fact, True)
                for fact in sorted(missing_positive)
            ]
            obligations.extend(
                TraceObligation(index, action.pddl(), fact, False)
                for fact in sorted(violated_negative)
            )
            return _trace_result(
                SignedTraceStatus.ACTION_PRECONDITION_FAILURE,
                true_facts,
                false_facts,
                obligations=obligations,
            )

        true_facts.difference_update(action.del_effects)
        true_facts.update(action.add_effects)
        false_facts.difference_update(action.add_effects)
        false_facts.update(action.del_effects)
        if true_facts & false_facts:
            return _trace_result(
                SignedTraceStatus.DOMAIN_ERROR,
                true_facts,
                false_facts,
                reason=f"transition created TRUE/FALSE conflict at index {index}",
            )
        try:
            validate_state(problem, frozenset(true_facts))
        except DomainError as error:
            return _trace_result(
                SignedTraceStatus.DOMAIN_ERROR,
                true_facts,
                false_facts,
                reason=str(error),
            )

    unknown_goal = (problem.goal | problem.negative_goal) - true_facts - false_facts
    if unknown_goal:
        return _trace_result(
            SignedTraceStatus.PLAN_GROUNDING_INCOMPLETE,
            true_facts,
            false_facts,
            reason="unknown goal literals: " + ", ".join(str(fact) for fact in sorted(unknown_goal)),
        )
    missing_goal = problem.goal & false_facts
    violated_negative_goal = problem.negative_goal & true_facts
    if missing_goal or violated_negative_goal:
        obligations = [
            TraceObligation(None, "GOAL", fact, True) for fact in sorted(missing_goal)
        ]
        obligations.extend(
            TraceObligation(None, "GOAL", fact, False)
            for fact in sorted(violated_negative_goal)
        )
        return _trace_result(
            SignedTraceStatus.FINAL_GOAL_FAILURE,
            true_facts,
            false_facts,
            obligations=obligations,
        )
    return _trace_result(SignedTraceStatus.VALID, true_facts, false_facts)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_named(values: Sequence[Tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _plan_text(plan: Sequence[GroundAction]) -> str:
    return "".join(f"{action.pddl()}\n" for action in plan)


def _retry_key_strings(keys: FrozenSet[Any]) -> Tuple[str, ...]:
    return tuple(sorted(_json_bytes(key).decode("utf-8") for key in keys))


def _validate_context(context: ContextEnvelope) -> str | None:
    if context.phase is ContextPhase.PREINSTALL_VAL:
        nullable = (
            context.graph_version,
            context.occurrence_id,
            context.attempt_id,
            context.certificate_hash,
            context.safety_epoch,
        )
        if any(value is not None for value in nullable):
            return "PREINSTALL_VAL requires explicit None parent/attempt/certificate/safety fields"
        return None
    if context.phase is ContextPhase.RECOVERY_VAL:
        if context.graph_version is None or context.certificate_hash is None:
            return "RECOVERY_VAL requires parent graph_version and certificate_hash"
        if any(
            value is not None
            for value in (context.occurrence_id, context.attempt_id, context.safety_epoch)
        ):
            return "RECOVERY_VAL occurrence_id, attempt_id, and safety_epoch must be None"
        return None
    return "VAL context phase must be PREINSTALL_VAL or RECOVERY_VAL"


def _check_sidecar(sidecar: bytes, plan: Sequence[GroundAction]) -> str | None:
    try:
        payload = json.loads(sidecar.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return f"occurrence sidecar parse error: {error}"
    if not isinstance(payload, list) or len(payload) != len(plan):
        return "occurrence sidecar length mismatch"
    seen = set()
    for index, (item, action) in enumerate(zip(payload, plan)):
        if not isinstance(item, dict):
            return f"occurrence sidecar item {index} is not an object"
        occurrence_id = item.get("occurrence_id")
        if not isinstance(occurrence_id, str) or not occurrence_id or occurrence_id in seen:
            return f"invalid or duplicate occurrence_id at index {index}"
        seen.add(occurrence_id)
        if item.get("schema") != action.schema or item.get("arguments") != list(action.arguments):
            return f"occurrence sidecar action mismatch at index {index}"
    return None


def _input_components(
    *,
    problem: TaskProblem,
    plan: Sequence[GroundAction],
    occurrence_sidecar: bytes,
    context: ContextEnvelope,
    val_binary: Path,
    val_binary_sha256: str,
    val_version: str,
    timeout_seconds: float,
    forbidden_retry_keys: FrozenSet[Any],
    retry_ledger_version: int,
) -> list[Tuple[str, bytes]]:
    return [
        ("domain", render_domain_pddl().encode("utf-8")),
        ("problem", render_problem_pddl(problem).encode("utf-8")),
        ("plan", _plan_text(plan).encode("utf-8")),
        ("occurrence_sidecar", occurrence_sidecar),
        ("context", _json_bytes(context.payload())),
        ("val_binary_path", str(val_binary.resolve()).encode("utf-8")),
        ("val_binary_sha256", val_binary_sha256.encode("ascii")),
        ("val_version", val_version.encode("utf-8")),
        ("wrapper_version", WRAPPER_VERSION.encode("ascii")),
        (
            "val_argv_template",
            _json_bytes(["{binary}", "{domain.pddl}", "{problem.pddl}", "{plan}"]),
        ),
        ("timeout_seconds", repr(float(timeout_seconds)).encode("ascii")),
        ("forbidden_retry_keys", _json_bytes(_retry_key_strings(forbidden_retry_keys))),
        ("retry_ledger_version", str(retry_ledger_version).encode("ascii")),
    ]


class ValWrapper:
    def __init__(
        self,
        val_binary: Path | str,
        *,
        timeout_seconds: float,
        val_version: str | None = None,
    ) -> None:
        self.val_binary = Path(val_binary)
        self.timeout_seconds = float(timeout_seconds)
        self.val_version_override = val_version

    def _binary_metadata(self) -> tuple[str, str]:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        binary_hash = _sha256(self.val_binary.read_bytes())
        if self.val_version_override is not None:
            return binary_hash, self.val_version_override
        process = subprocess.run(
            [str(self.val_binary), "-h"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(self.timeout_seconds, 1.0),
            check=False,
        )
        output = process.stdout + "\n" + process.stderr
        match = re.search(r"^Version [^\n]+", output, re.MULTILINE)
        if process.returncode != 0 or match is None:
            raise RuntimeError("cannot identify VAL binary version")
        return binary_hash, match.group(0)

    def validate(
        self,
        problem: TaskProblem,
        plan: Sequence[GroundAction],
        occurrence_sidecar: bytes,
        context: ContextEnvelope,
        *,
        forbidden_retry_keys: FrozenSet[Any] = frozenset(),
        retry_ledger_version: int = 0,
    ) -> ValidationResult:
        empty_trace = SignedTraceResult(
            SignedTraceStatus.VALIDATION_ERROR,
            frozenset(problem.initial_state),
            frozenset(problem.initial_false),
        )
        if retry_ledger_version < 0:
            return ValidationResult(
                ValidationStatus.VALIDATION_ERROR,
                empty_trace,
                None,
                reason="retry_ledger_version must be nonnegative",
            )
        context_error = _validate_context(context)
        if context_error:
            return ValidationResult(
                ValidationStatus.VALIDATION_ERROR,
                empty_trace,
                None,
                reason=context_error,
            )
        sidecar_error = _check_sidecar(occurrence_sidecar, plan)
        if sidecar_error:
            return ValidationResult(
                ValidationStatus.VALIDATION_ERROR,
                empty_trace,
                None,
                reason=sidecar_error,
            )
        trace = run_signed_trace(problem, plan)
        if trace.status is not SignedTraceStatus.VALID:
            status = (
                ValidationStatus.INVALID
                if trace.status
                in {
                    SignedTraceStatus.ACTION_PRECONDITION_FAILURE,
                    SignedTraceStatus.FINAL_GOAL_FAILURE,
                }
                else ValidationStatus.VALIDATION_ERROR
            )
            return ValidationResult(status, trace, None, reason=trace.reason)

        try:
            binary_hash, val_version = self._binary_metadata()
            domain_text = render_domain_pddl()
            problem_text = render_problem_pddl(problem)
            plan_text = _plan_text(plan)
            with tempfile.TemporaryDirectory(prefix="logiv-val-") as temporary:
                directory = Path(temporary)
                domain_path = directory / "domain.pddl"
                problem_path = directory / "problem.pddl"
                plan_path = directory / "candidate.plan"
                domain_path.write_text(domain_text, encoding="utf-8")
                problem_path.write_text(problem_text, encoding="utf-8")
                plan_path.write_text(plan_text, encoding="utf-8")
                process = subprocess.run(
                    [
                        str(self.val_binary),
                        str(domain_path),
                        str(problem_path),
                        str(plan_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            return ValidationResult(
                ValidationStatus.VALIDATION_ERROR,
                trace,
                None,
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                reason="VAL timeout",
            )
        except (OSError, ValueError, RuntimeError) as error:
            return ValidationResult(
                ValidationStatus.VALIDATION_ERROR,
                trace,
                None,
                reason=f"VAL process error: {error}",
            )

        output = process.stdout + "\n" + process.stderr
        if process.returncode == 0 and _VALID_MARKER.search(output):
            components = _input_components(
                problem=problem,
                plan=plan,
                occurrence_sidecar=occurrence_sidecar,
                context=context,
                val_binary=self.val_binary,
                val_binary_sha256=binary_hash,
                val_version=val_version,
                timeout_seconds=self.timeout_seconds,
                forbidden_retry_keys=forbidden_retry_keys,
                retry_ledger_version=retry_ledger_version,
            )
            components.append(
                (
                    "validation_output",
                    _json_bytes(
                        {
                            "returncode": process.returncode,
                            "stdout": process.stdout,
                            "stderr": process.stderr,
                        }
                    ),
                )
            )
            component_hashes = tuple((name, _sha256(value)) for name, value in components)
            certificate = PlanCertificate(
                certificate_hash=_digest_named(
                    [(name, digest.encode("ascii")) for name, digest in component_hashes]
                ),
                component_hashes=component_hashes,
                context=context,
                val_binary=str(self.val_binary.resolve()),
                val_binary_sha256=binary_hash,
                val_version=val_version,
                wrapper_version=WRAPPER_VERSION,
                timeout_seconds=self.timeout_seconds,
                forbidden_retry_keys=_retry_key_strings(forbidden_retry_keys),
                retry_ledger_version=retry_ledger_version,
            )
            return ValidationResult(
                ValidationStatus.VALID,
                trace,
                certificate,
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
            )
        if process.returncode != 0 and any(marker in output for marker in _INVALID_MARKERS):
            return ValidationResult(
                ValidationStatus.INVALID,
                trace,
                None,
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
            )
        return ValidationResult(
            ValidationStatus.VALIDATION_ERROR,
            trace,
            None,
            stdout=process.stdout,
            stderr=process.stderr,
            returncode=process.returncode,
            reason="unrecognized or inconsistent VAL output",
        )


def verify_certificate(
    certificate: PlanCertificate,
    *,
    problem: TaskProblem,
    plan: Sequence[GroundAction],
    occurrence_sidecar: bytes,
    context: ContextEnvelope,
    val_binary: Path | str,
    timeout_seconds: float,
    forbidden_retry_keys: FrozenSet[Any],
    retry_ledger_version: int,
) -> bool:
    try:
        binary = Path(val_binary)
        metadata_wrapper = ValWrapper(binary, timeout_seconds=timeout_seconds)
        binary_hash, actual_version = metadata_wrapper._binary_metadata()
        if binary_hash != certificate.val_binary_sha256:
            return False
        if actual_version != certificate.val_version:
            return False
        components = _input_components(
            problem=problem,
            plan=plan,
            occurrence_sidecar=occurrence_sidecar,
            context=context,
            val_binary=binary,
            val_binary_sha256=binary_hash,
            val_version=certificate.val_version,
            timeout_seconds=timeout_seconds,
            forbidden_retry_keys=forbidden_retry_keys,
            retry_ledger_version=retry_ledger_version,
        )
    except (OSError, DomainError, ValueError, TypeError, subprocess.SubprocessError):
        return False
    expected_names = [name for name, _ in components] + ["validation_output"]
    if [name for name, _ in certificate.component_hashes] != expected_names:
        return False
    stored = dict(certificate.component_hashes)
    for name, value in components:
        if stored.get(name) != _sha256(value):
            return False
    if certificate.context != context:
        return False
    if certificate.val_binary != str(binary.resolve()):
        return False
    if certificate.wrapper_version != WRAPPER_VERSION:
        return False
    if certificate.timeout_seconds != float(timeout_seconds):
        return False
    if certificate.forbidden_retry_keys != _retry_key_strings(forbidden_retry_keys):
        return False
    if certificate.retry_ledger_version != retry_ledger_version:
        return False
    output_hash = stored.get("validation_output", "")
    if not re.fullmatch(r"[0-9a-f]{64}", output_hash):
        return False
    expected_certificate_hash = _digest_named(
        [(name, digest.encode("ascii")) for name, digest in certificate.component_hashes]
    )
    return certificate.certificate_hash == expected_certificate_hash
