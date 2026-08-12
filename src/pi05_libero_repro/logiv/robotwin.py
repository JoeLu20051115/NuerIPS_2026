from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class TruthValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RobotwinStage:
    fact: str
    policy_prompt: str
    observer_question: str
    transient: bool = False

    def __post_init__(self) -> None:
        if not self.fact or not self.policy_prompt or not self.observer_question:
            raise ValueError("RoboTwin stage fields must be nonempty")


@dataclass(frozen=True)
class RobotwinTask:
    name: str
    stages: tuple[RobotwinStage, ...]

    @property
    def goal_fact(self) -> str:
        return self.stages[-1].fact


def _stage(
    fact: str, prompt: str, question: str, *, transient: bool = False
) -> RobotwinStage:
    return RobotwinStage(fact, prompt, question, transient)


ROBOTWIN_TASKS: dict[str, RobotwinTask] = {
    "handover_block": RobotwinTask(
        "handover_block",
        (
            _stage(
                "block-held-by-left",
                "Use the left arm to grasp the red block and move it to the center handover area. Keep holding the block with the left gripper.",
                "Is the red block securely held by the left gripper near the center handover area?",
                transient=True,
            ),
            _stage(
                "block-held-by-right",
                "Use the right arm to take the red block from the left gripper. Release the left gripper only after the right gripper securely holds the block.",
                "Has the red block been transferred so that the right gripper holds it and the left gripper has released it?",
                transient=True,
            ),
            _stage(
                "block-on-blue-pad",
                "Using the right arm, place the red block centered on the blue pad, open the right gripper, and withdraw both arms.",
                "Is the red block centered on the blue pad with the right gripper open?",
            ),
        ),
    ),
    "open_microwave": RobotwinTask(
        "open_microwave",
        (
            _stage(
                "handle-grasped",
                "Reach the microwave door handle with the nearest arm and close the gripper firmly around the handle. Do not release it.",
                "Is one gripper visibly closed around the microwave door handle?",
                transient=True,
            ),
            _stage(
                "microwave-open",
                "Keep holding the microwave handle and pull the door fully open along its hinge path, then release the handle.",
                "Is the microwave door visibly open by more than halfway?",
            ),
        ),
    ),
    "place_dual_shoes": RobotwinTask(
        "place_dual_shoes",
        (
            _stage(
                "first-shoe-in-box",
                "Pick up the shoe on the left side and place it inside the shoe box with its toe pointing left. Open the gripper after placement.",
                "Is one shoe fully inside the shoe box with its toe pointing left?",
            ),
            _stage(
                "both-shoes-in-box",
                "Pick up the remaining shoe and place it inside the shoe box beside the first shoe, with its toe pointing left. Release it and withdraw both arms.",
                "Are both shoes fully inside the shoe box with both toes pointing left and both grippers released?",
            ),
        ),
    ),
    "stamp_seal": RobotwinTask(
        "stamp_seal",
        (
            _stage(
                "seal-grasped",
                "Grasp the seal upright with the nearest arm and lift it clear of the table. Keep holding it.",
                "Is the seal visibly held upright by a gripper above the table?",
                transient=True,
            ),
            _stage(
                "seal-on-target",
                "Move the held seal directly over the colored target, press it firmly down on the target, then release the seal centered on the target and withdraw the arm.",
                "Is the seal centered on the colored target with the gripper released?",
            ),
        ),
    ),
    "blocks_ranking_size": RobotwinTask(
        "blocks_ranking_size",
        (
            _stage(
                "small-block-right",
                "Move the smallest block to the right side of the center row and release it.",
                "Is the smallest block placed at the right side of the horizontal center row?",
            ),
            _stage(
                "medium-block-center",
                "Move the medium-sized block to the middle of the same horizontal center row, immediately left of the smallest block, and release it.",
                "Are the medium and smallest blocks aligned in a horizontal center row with medium immediately left of small?",
            ),
            _stage(
                "blocks-ranked-large-to-small",
                "Move the largest block to the left of the medium block so all three blocks form a close horizontal row ordered large, medium, small from left to right. Release and withdraw both arms.",
                "Are all three blocks closely aligned at the table center and ordered large, medium, small from left to right with both grippers released?",
            ),
        ),
    ),
    "move_can_pot": RobotwinTask(
        "move_can_pot",
        (
            _stage(
                "can-beside-pot",
                "Pick up the can and place it upright beside the pot on the same side of the table as its starting position. Release the can close to the pot and withdraw both arms.",
                "Is the can upright directly beside the pot, resting on the table, with both grippers released?",
            ),
        ),
    ),
    "turn_switch": RobotwinTask(
        "turn_switch",
        (
            _stage(
                "switch-activated",
                "Use the nearest arm to press and fully activate the switch, then withdraw the arm.",
                "Is the switch visibly moved fully into its activated position?",
            ),
        ),
    ),
    "stack_blocks_three": RobotwinTask(
        "stack_blocks_three",
        (
            _stage(
                "red-base-centered",
                "Move the red block to the table center as the bottom block of the stack, release it, and lift the gripper away.",
                "Is the red block resting at the table center as a clear stack base?",
            ),
            _stage(
                "green-on-red",
                "Pick up the green block and place it centered directly on top of the red block. Release it gently and lift the gripper away.",
                "Is the green block stably centered directly on top of the red block?",
            ),
            _stage(
                "blue-on-green",
                "Pick up the blue block and place it centered directly on top of the green block. Release it gently and withdraw both arms.",
                "Are the three blocks stably stacked red on the bottom, green in the middle, and blue on top with both grippers released?",
            ),
        ),
    ),
    "stack_bowls_three": RobotwinTask(
        "stack_bowls_three",
        (
            _stage(
                "bottom-bowl-centered",
                "Move one bowl to the table center to form the bottom of the bowl stack, release it, and lift the gripper away.",
                "Is one bowl resting upright at the table center as the stack base?",
            ),
            _stage(
                "two-bowls-stacked",
                "Place a second bowl centered inside and above the bottom bowl, release it gently, and lift the gripper away.",
                "Are two bowls visibly nested and centered as a stable stack?",
            ),
            _stage(
                "three-bowls-stacked",
                "Place the remaining bowl centered inside and above the two stacked bowls, release it gently, and withdraw both arms.",
                "Are all three bowls visibly nested in one stable centered stack with both grippers released?",
            ),
        ),
    ),
    "beat_block_hammer": RobotwinTask(
        "beat_block_hammer",
        (
            _stage(
                "hammer-grasped",
                "Grasp the hammer firmly by its handle with the nearest arm and lift it above the table. Keep holding it.",
                "Is the hammer visibly held by its handle in a gripper above the table?",
                transient=True,
            ),
            _stage(
                "block-struck",
                "Using the held hammer, align the hammer head over the block and strike the block firmly. Keep control of the hammer after contact.",
                "Is the hammer head touching or just completing a clear strike on the block?",
            ),
        ),
    ),
}


CANONICAL_POLICY_PROMPTS = {
    "handover_block": "Pass the red block from the left hand to the right hand, then place the red block onto the blue pad using the right arm.",
    "open_microwave": "Pull the handle to open the microwave door.",
    "place_dual_shoes": "Put both shoes into the shoe box with their tips pointing left.",
    "stamp_seal": "Pick up the seal and press it firmly onto the colored target.",
    "blocks_ranking_size": "Move the large, medium, and small blocks to the table center and align them in order from large to small, left to right.",
    "move_can_pot": "Pick up the can and move it next to the pot.",
    "turn_switch": "Press the switch using the nearest arm.",
    "stack_blocks_three": "Move the red, green, and blue blocks to the table center, then stack blue on green and green on red.",
    "stack_bowls_three": "Stack all three bowls together one by one.",
    "beat_block_hammer": "Pick up the hammer and strike the block.",
}


RECOVERY_POLICY_PROMPTS = {
    "handover_block": (
        "Use the left arm to grab the red block.",
        "Hand the red block to the right arm.",
        "Place the red block onto the blue pad using the right arm.",
    ),
    "open_microwave": (
        "Grab the handle and open the microwave door.",
        "Pull the microwave door open with the left arm.",
    ),
    "place_dual_shoes": (
        "Place the left shoe into the shoe box, tip left.",
        "Place two shoes into the shoe box, tips left.",
    ),
    "stamp_seal": (
        "Grab the seal using the nearest arm.",
        "Press the held seal onto the colored target and release it.",
    ),
    "blocks_ranking_size": (
        "Move the smallest block to the center-right.",
        "Move the medium block to the center.",
        "Arrange the blocks largest to smallest, left to right.",
    ),
    "move_can_pot": (
        "Pick up the can and set it beside the pot.",
    ),
    "turn_switch": (
        "Locate and press the switch using the nearest arm.",
    ),
    "stack_blocks_three": (
        "Move the red block to the table center.",
        "Place the green block centered on the red block and release it.",
        "Place the blue block centered on the green block and release it.",
    ),
    "stack_bowls_three": (
        "Place the bottom bowl at the table center.",
        "Place the second bowl centered inside the first bowl.",
        "Place the third bowl centered inside the second bowl.",
    ),
    "beat_block_hammer": (
        "Pick up the hammer.",
        "Use the hammer to strike the block.",
    ),
}


# These tasks' frozen instructions identify a randomized object variant, target,
# or arm. Keep that scene binding during repair; PDDL still chooses the active
# stage and the shorter repair horizon supplies the local receding-horizon edit.
SCENE_BOUND_REPAIR_TASKS = frozenset(
    {
        "place_dual_shoes",
        "stamp_seal",
        "move_can_pot",
        "turn_switch",
        "beat_block_hammer",
    }
)


def bind_canonical_policy_prompt(task: RobotwinTask) -> RobotwinTask:
    prompt = CANONICAL_POLICY_PROMPTS[task.name]
    return RobotwinTask(
        task.name,
        tuple(
            RobotwinStage(
                stage.fact,
                prompt,
                stage.observer_question,
                stage.transient,
            )
            for stage in task.stages
        ),
    )


@dataclass(frozen=True)
class RobotwinPlanAction:
    stage_index: int
    fact: str
    pddl_name: str
    policy_prompt: str


@dataclass(frozen=True)
class RobotwinCertifiedPlan:
    actions: tuple[RobotwinPlanAction, ...]
    valid: bool
    certificate_sha256: str
    domain_pddl: str
    problem_pddl: str
    plan_pddl: str
    val_stdout: str
    val_stderr: str


def _and(atoms: Sequence[str]) -> str:
    return "(and)" if not atoms else f"(and {' '.join(atoms)})"


class RobotwinPddlPlanner:
    """Bounded linear PDDL planner for a registered RoboTwin causal DAG."""

    def __init__(self, val_binary: str | Path, *, timeout_seconds: float = 5.0) -> None:
        self.val_binary = Path(val_binary)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @staticmethod
    def _domain(task: RobotwinTask) -> str:
        predicates = "\n".join(f"    ({stage.fact})" for stage in task.stages)
        actions = []
        for index, stage in enumerate(task.stages):
            preconditions = [] if index == 0 else [f"({task.stages[index - 1].fact})"]
            actions.append(
                "\n".join(
                    (
                        f"  (:action achieve-stage-{index}",
                        "    :parameters ()",
                        f"    :precondition {_and(preconditions)}",
                        f"    :effect ({stage.fact})",
                        "  )",
                    )
                )
            )
        return "\n".join(
            (
                f"(define (domain logiv-{task.name})",
                "  (:requirements :strips)",
                "  (:predicates",
                predicates,
                "  )",
                *actions,
                ")",
                "",
            )
        )

    @staticmethod
    def _problem(task: RobotwinTask, true_facts: set[str]) -> str:
        initial = " ".join(
            f"({stage.fact})" for stage in task.stages if stage.fact in true_facts
        )
        return "\n".join(
            (
                f"(define (problem logiv-{task.name}-episode)",
                f"  (:domain logiv-{task.name})",
                f"  (:init {initial})" if initial else "  (:init)",
                f"  (:goal ({task.goal_fact}))",
                ")",
                "",
            )
        )

    def plan(
        self, task: RobotwinTask, facts: Mapping[str, TruthValue]
    ) -> RobotwinCertifiedPlan:
        known_names = {stage.fact for stage in task.stages}
        if not set(facts) <= known_names:
            raise ValueError("facts contain names outside the registered task universe")
        confirmed = [
            index
            for index, stage in enumerate(task.stages)
            if facts.get(stage.fact) is TruthValue.TRUE
        ]
        # A later causal milestone directly entails its registered predecessors.
        # This prevents a conservative observer from sending the policy back to
        # disturb already completed geometry when an earlier fact is occluded.
        prefix = max(confirmed, default=-1) + 1
        actions = tuple(
            RobotwinPlanAction(
                stage_index=index,
                fact=stage.fact,
                pddl_name=f"achieve-stage-{index}",
                policy_prompt=stage.policy_prompt,
            )
            for index, stage in enumerate(task.stages)
            if index >= prefix
        )
        true_facts = {stage.fact for stage in task.stages[:prefix]}
        domain = self._domain(task)
        problem = self._problem(task, true_facts)
        plan_text = "".join(f"({action.pddl_name})\n" for action in actions)
        with tempfile.TemporaryDirectory(prefix="logiv-robotwin-val-") as directory:
            root = Path(directory)
            domain_path = root / "domain.pddl"
            problem_path = root / "problem.pddl"
            plan_path = root / "candidate.plan"
            domain_path.write_text(domain, encoding="utf-8")
            problem_path.write_text(problem, encoding="utf-8")
            plan_path.write_text(plan_text, encoding="utf-8")
            result = subprocess.run(
                [str(self.val_binary), str(domain_path), str(problem_path), str(plan_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        valid = result.returncode == 0 and "Plan valid" in result.stdout
        payload = {
            "domain": domain,
            "problem": problem,
            "plan": plan_text,
            "val_binary_sha256": hashlib.sha256(self.val_binary.read_bytes()).hexdigest(),
            "val_stdout": result.stdout,
            "valid": valid,
        }
        certificate = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return RobotwinCertifiedPlan(
            actions=actions,
            valid=valid,
            certificate_sha256=certificate,
            domain_pddl=domain,
            problem_pddl=problem,
            plan_pddl=plan_text,
            val_stdout=result.stdout,
            val_stderr=result.stderr,
        )


def extract_robotwin_images(observation: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    try:
        cameras = observation["observation"]
        images = tuple(
            np.asarray(cameras[name]["rgb"])
            for name in ("head_camera", "right_camera", "left_camera")
        )
    except (KeyError, TypeError) as error:
        raise ValueError("RoboTwin observation is missing required RGB cameras") from error
    for image in images:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RoboTwin RGB camera must be HxWx3 uint8")
    return images


class RobotwinFactGrounder:
    """GPT-4o adapter restricted to registered visual facts."""

    SYSTEM = (
        "You are a conservative visual fact observer for a robot simulation. "
        "Only classify the registered facts from the three synchronized camera views. "
        "Use UNKNOWN when occlusion or ambiguity prevents direct confirmation. "
        "Do not plan, recommend, rank, or describe robot actions."
    )

    def __init__(self, client: Any) -> None:
        self.client = client
        self._initial_images: tuple[np.ndarray, ...] | None = None

    def observe(
        self, task: RobotwinTask, observation: Mapping[str, Any], *, epoch: int
    ) -> dict[str, TruthValue]:
        names = [stage.fact for stage in task.stages]
        definitions = "\n".join(
            f"- {stage.fact}: {stage.observer_question}" for stage in task.stages
        )
        schema = {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": names},
                            "value": {
                                "type": "string",
                                "enum": [value.value for value in TruthValue],
                            },
                        },
                        "required": ["name", "value"],
                        "additionalProperties": False,
                    },
                    "minItems": len(names),
                    "maxItems": len(names),
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
        }
        current_images = extract_robotwin_images(observation)
        if epoch == 0 or self._initial_images is None:
            self._initial_images = tuple(image.copy() for image in current_images)
            images = current_images
            comparison = "These three images are the episode initial views."
        else:
            images = self._initial_images + current_images
            comparison = (
                "The first three images are the initial head/right/left views; "
                "the last three are the current synchronized head/right/left views. "
                "Classify facts for the current views using visible change from initial."
            )
        response = self.client.complete_json(
            purpose="state_gate",
            system=self.SYSTEM,
            text=(
                f"Task: {task.name}. Observation epoch: {epoch}.\n"
                f"{comparison}\n"
                "Return exactly one classification for every registered fact.\n"
                + definitions
            ),
            images=images,
            schema_name="robotwin_visual_facts",
            schema=schema,
        )
        values = {name: TruthValue.UNKNOWN for name in names}
        records = response.get("facts")
        if not isinstance(records, list):
            return values
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            if name not in values or name in seen:
                continue
            seen.add(name)
            try:
                values[name] = TruthValue(record.get("value"))
            except (TypeError, ValueError):
                values[name] = TruthValue.UNKNOWN
        return values


@dataclass(frozen=True)
class RobotwinControllerEvent:
    epoch: int
    active_stage_index: int | None
    facts: Mapping[str, TruthValue]
    plan: RobotwinCertifiedPlan
    control_mode: str = "REPAIR"


@dataclass(frozen=True)
class RobotwinEpisodeOutcome:
    success: bool
    reason: str
    events: tuple[RobotwinControllerEvent, ...]
    dispatches: int


class RobotwinEpisodeController:
    """Receding-horizon LOGIV controller with VLM facts and PDDL-only repair."""

    def __init__(
        self,
        task: RobotwinTask,
        planner: RobotwinPddlPlanner,
        grounder: Any,
        *,
        max_dispatches: int = 64,
        base_stall_observations: int = 2,
        stage_stall_observations: Sequence[int] | None = None,
        min_base_dispatches: int = 0,
    ) -> None:
        self.task = task
        self.planner = planner
        self.grounder = grounder
        self.max_dispatches = int(max_dispatches)
        self.base_stall_observations = int(base_stall_observations)
        self.min_base_dispatches = int(min_base_dispatches)
        self.stage_stall_observations = (
            tuple(int(value) for value in stage_stall_observations)
            if stage_stall_observations is not None
            else None
        )
        if self.max_dispatches <= 0:
            raise ValueError("max_dispatches must be positive")
        if self.base_stall_observations <= 0:
            raise ValueError("base_stall_observations must be positive")
        if self.min_base_dispatches < 0:
            raise ValueError("min_base_dispatches must be nonnegative")
        if self.stage_stall_observations is not None:
            if len(self.stage_stall_observations) != len(self.task.stages):
                raise ValueError("stage stall observations must match task stages")
            if any(value <= 0 for value in self.stage_stall_observations):
                raise ValueError("stage stall observations must be positive")

    def _stall_threshold(self, stage_index: int) -> int:
        if self.stage_stall_observations is None:
            return self.base_stall_observations
        return self.stage_stall_observations[stage_index]

    def run(
        self,
        *,
        initial_observation: Any,
        dispatch: Callable[[str], Any],
        dispatch_with_mode: Callable[[str, str], Any] | None = None,
        dispatch_with_context: Callable[[str, str, int | None], Any] | None = None,
        native_success: Callable[[], bool],
        budget_exhausted: Callable[[], bool],
        base_prompt: str | None = None,
    ) -> RobotwinEpisodeOutcome:
        events: list[RobotwinControllerEvent] = []
        observation = initial_observation
        dispatches = 0
        latched_true: set[str] = set()
        control_mode = "BASE_MONITORED" if base_prompt is not None else "REPAIR"
        previous_frontier: int | None = None
        unchanged_false_observations = 0
        visual_goal_native_conflicts = 0
        latched_false_observations: dict[str, int] = {}
        for epoch in range(self.max_dispatches + 1):
            observed = self.grounder.observe(self.task, observation, epoch=epoch)
            latched_true.update(
                name for name, value in observed.items() if value is TruthValue.TRUE
            )
            # A completed node is latched against one-frame VLM flicker. During
            # repair, two consecutive visible FALSE observations prove that its
            # effect no longer holds, so PDDL may reopen that node. A later TRUE
            # node still entails its registered prefix in the planner.
            for stage in self.task.stages:
                if stage.fact == self.task.goal_fact:
                    continue
                value = observed.get(stage.fact, TruthValue.UNKNOWN)
                if value is TruthValue.TRUE or control_mode != "REPAIR":
                    latched_false_observations[stage.fact] = 0
                elif value is TruthValue.FALSE and stage.fact in latched_true:
                    count = latched_false_observations.get(stage.fact, 0) + 1
                    latched_false_observations[stage.fact] = count
                    if count >= 2:
                        latched_true.discard(stage.fact)
                else:
                    latched_false_observations[stage.fact] = 0
            facts = {
                name: TruthValue.TRUE if name in latched_true else value
                for name, value in observed.items()
            }
            succeeded = native_success()
            if facts.get(self.task.goal_fact) is TruthValue.TRUE and not succeeded:
                visual_goal_native_conflicts += 1
            else:
                visual_goal_native_conflicts = 0
            goal_conflict_repair = base_prompt is not None and (
                control_mode == "REPAIR"
                or (
                    dispatches >= self.min_base_dispatches
                    and visual_goal_native_conflicts
                    >= self.base_stall_observations
                )
            ) and facts.get(self.task.goal_fact) is TruthValue.TRUE and not succeeded
            if goal_conflict_repair:
                # The benchmark evaluator is authoritative for terminal success.
                # A persistent visual false positive must not erase the remaining
                # goal from the Current Problem used by PDDL.
                latched_true.discard(self.task.goal_fact)
                facts[self.task.goal_fact] = TruthValue.FALSE
            plan = self.planner.plan(self.task, facts)
            if not plan.valid:
                return RobotwinEpisodeOutcome(
                    False, "VAL_REJECTED_PLAN", tuple(events), dispatches
                )
            active = plan.actions[0].stage_index if plan.actions else None
            if goal_conflict_repair:
                control_mode = "REPAIR"
            if (
                not succeeded
                and control_mode == "BASE_MONITORED"
                and active is not None
            ):
                active_fact = self.task.stages[active].fact
                if active != previous_frontier:
                    previous_frontier = active
                    unchanged_false_observations = 0
                elif facts.get(active_fact) is TruthValue.FALSE:
                    unchanged_false_observations += 1
                else:
                    unchanged_false_observations = 0
                if (
                    dispatches >= self.min_base_dispatches
                    and unchanged_false_observations
                    >= self._stall_threshold(active)
                ):
                    control_mode = "REPAIR"
            events.append(
                RobotwinControllerEvent(
                    epoch, active, facts, plan, control_mode=control_mode
                )
            )
            if succeeded:
                return RobotwinEpisodeOutcome(
                    True, "NATIVE_SUCCESS", tuple(events), dispatches
                )
            if budget_exhausted() or dispatches >= self.max_dispatches:
                return RobotwinEpisodeOutcome(
                    False, "ACTION_BUDGET_EXHAUSTED", tuple(events), dispatches
                )
            if control_mode == "BASE_MONITORED":
                assert base_prompt is not None
                prompt = base_prompt
            elif active is None:
                if base_prompt is None:
                    return RobotwinEpisodeOutcome(
                        False,
                        "VISUAL_GOAL_WITHOUT_NATIVE_SUCCESS",
                        tuple(events),
                        dispatches,
                    )
                prompt = base_prompt
            elif base_prompt is None:
                prompt = self.task.stages[active].policy_prompt
            elif self.task.name in SCENE_BOUND_REPAIR_TASKS:
                prompt = base_prompt
            else:
                prompt = RECOVERY_POLICY_PROMPTS[self.task.name][active]
            if dispatch_with_context is not None:
                observation = dispatch_with_context(prompt, control_mode, active)
            elif dispatch_with_mode is None:
                observation = dispatch(prompt)
            else:
                observation = dispatch_with_mode(prompt, control_mode)
            dispatches += 1
        raise AssertionError("controller loop exceeded its explicit dispatch bound")
