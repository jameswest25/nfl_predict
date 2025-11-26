#!/usr/bin/env python3
"""
codex_meta_loop.py

Hybrid orchestrator for the `codex` CLI that runs a multi-prompt, multi-phase loop:

1. Prompt 1 (chat): Conceptual analysis (data realism, leakage, incomplete/wrong implementations).
2. Prompt 2 (chat): Rich, detailed plan (no edits).
3. Prompt 3 (exec): Agent-style implementation of the plan, up to N iterations (default 10).
4. Prompt 4 (exec): Single audit pass to check if plan is fully implemented.

If audit finds gaps:
- Treat audit output as new "analysis",
- Build a new plan,
- Repeat implementation loop (max 10),
- Audit once again, etc.

When a plan + audit confirm that "The plan is completely implemented",
the meta-cycle ends and the script starts over at Prompt 1.
"""

import subprocess
import time
from pathlib import Path
import textwrap
from typing import Tuple, Optional
from shutil import which

# ---------------- CONFIG ----------------

# Where your repo lives; "." = current directory
PROJECT_DIR = Path(".").resolve()

# How long to sleep between full meta-cycles (seconds)
SLEEP_BETWEEN_CYCLES = 30

# Model to use for chat steps (analysis/plan).
# Adjust to whatever models your `codex` CLI supports.
CHAT_MODEL = "gpt-4.1-mini"

# Command for codex CLI (if installed via brew, usually just "codex")
CODEX_CMD = "codex"

# Max number of implementation iterations (Prompt 3) per plan
MAX_IMPLEMENTATION_ITERATIONS = 10
MAX_VALIDATION_ITERATIONS = 5
VALIDATION_SUCCESS_SENTINEL = "PIPELINE VALIDATION COMPLETE"

# Command used to run the full NFL pipeline end-to-end for regression testing
PIPELINE_VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"
PIPELINE_ENTRYPOINT = PROJECT_DIR / "main.py"
PIPELINE_EVAL_COMMAND = f"cd {PROJECT_DIR} && {PIPELINE_VENV_PYTHON} {PIPELINE_ENTRYPOINT}"

# Primary metric artifacts to inspect after each pipeline run
METRICS_ROOT = PROJECT_DIR / "output" / "metrics"
PRIMARY_TARGET = "anytime_td"
PRIMARY_FAMILY = "xgboost"
PRIMARY_METRIC_FILENAME = "metrics.yaml"
PRIMARY_IMPORTANCE_FILENAME = "feature_importance.json"
CUTOFF_SUMMARY_FILE = METRICS_ROOT / "cutoff_backtest_summary.csv"

# ----------------------------------------


def run_codex_chat(prompt: str) -> str:
    """
    Call `codex chat` with the given prompt and return the output as a string.
    Adjust flags (-m, etc.) if your codex version uses a different interface.
    """
    _ensure_codex_cli_available()
    print("\n[run_codex_chat] Starting chat call...\n")
    proc = subprocess.Popen(
        [CODEX_CMD, "chat", "-m", CHAT_MODEL],
        cwd=str(PROJECT_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate(prompt)
    print("[run_codex_chat] Finished. Exit code:", proc.returncode)
    if proc.returncode != 0:
        raise RuntimeError(f"'codex chat' exited with {proc.returncode}:\n{out}")
    print("---- Chat output (truncated) ----")
    print(out[:4000])
    print("-------- end chat output --------\n")
    return out


def run_codex_exec(prompt: str) -> str:
    """
    Call `codex exec` with the given prompt and return the full output as a string.
    In exec mode, Codex can act as an agent in the repo (read/edit files, run commands, etc.).
    """
    _ensure_codex_cli_available()
    print("\n[run_codex_exec] Starting exec call...\n")
    proc = subprocess.Popen(
        [CODEX_CMD, "exec"],
        cwd=str(PROJECT_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate(prompt)
    print("[run_codex_exec] Finished. Exit code:", proc.returncode)
    if proc.returncode != 0:
        raise RuntimeError(f"'codex exec' exited with {proc.returncode}:\n{out}")
    print("---- Exec output (truncated) ----")
    print(out[:4000])
    print("-------- end exec output --------\n")
    return out


def _ensure_codex_cli_available() -> None:
    """Verify that the codex CLI is installed and on PATH."""
    if which(CODEX_CMD) is None:
        raise RuntimeError(
            f"Unable to find '{CODEX_CMD}' on PATH. "
            "Install the Codex CLI or update CODEX_CMD in codex_meta_loop.py."
        )


def _latest_metric_run_dir(target: str = PRIMARY_TARGET, family: str = PRIMARY_FAMILY) -> Optional[Path]:
    """Return the most recent metrics directory for the given model target/family."""
    base = METRICS_ROOT / target / family
    if not base.exists():
        return None

    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def build_model_validation_instructions() -> str:
    """
    Provide concrete steps for validating whether code changes improved the model.
    """
    latest_dir = _latest_metric_run_dir()

    if latest_dir is not None:
        baseline_metrics_path = latest_dir / PRIMARY_METRIC_FILENAME
        baseline_importance_path = latest_dir / PRIMARY_IMPORTANCE_FILENAME
    else:
        placeholder_dir = METRICS_ROOT / PRIMARY_TARGET / PRIMARY_FAMILY / "<timestamp>"
        baseline_metrics_path = placeholder_dir / PRIMARY_METRIC_FILENAME
        baseline_importance_path = placeholder_dir / PRIMARY_IMPORTANCE_FILENAME

    latest_root_hint = METRICS_ROOT / PRIMARY_TARGET / PRIMARY_FAMILY

    instructions = [
        "- Capture a baseline snapshot before running any new code:",
        f"  * Record `auc`, `pr_auc`, `brier_score`, and `precision_at_thresh` from "
        f"`{baseline_metrics_path}`.",
        f"  * Note the current feature-importance distribution from `{baseline_importance_path}`.",
        f"- Run `{PIPELINE_EVAL_COMMAND}` to rebuild datasets, retrain models, and regenerate predictions.",
        "- After the run completes, identify the newest timestamped directory under "
        f"`{latest_root_hint}` and repeat the metric collection.",
        "  * Compare before/after metrics and explicitly call out improvements or regressions.",
        "  * Re-review the refreshed `feature_importance.json` for any unexpected signal shifts.",
        f"- Inspect `cutoff_backtest_summary.csv` (e.g., `{CUTOFF_SUMMARY_FILE}`) to confirm horizon-level "
        "hit rates and calibration remain acceptable.",
        "- Only declare success if the post-run metrics improve (higher AUC/PR AUC, lower Brier/log loss) "
        "or, at minimum, hold steady with a justified explanation. Any regression must be reported."
    ]

    return "\n".join(instructions)


def build_prompt_validation(plan: str, failure_notes: Optional[str] = None) -> str:
    """
    Prompt dedicated to ensuring the pipeline command runs successfully and produces metrics.
    """
    validation_text = build_model_validation_instructions()
    validation_block = textwrap.indent(validation_text, "    ")

    prompt = textwrap.dedent(f"""
    You are now in the validation loop. Your job is to make sure the full pipeline command runs
    successfully and produces refreshed metrics artifacts.

    Plan context (for reference only, do NOT re-hash it unless needed):

    \"\"\"PLAN_START
    {plan}
    PLAN_END\"\"\"

    Validation requirements:
{validation_block}

    Rules for this loop:
    - You may inspect and edit code, rerun commands, or add logging as needed to fix any issues uncovered
      while running `{PIPELINE_EVAL_COMMAND}`.
    - Keep commits incremental. If you end up making larger fixes, describe them clearly.
    - After each attempt, report status using `PROGRESS:`. If the run fails, capture stack traces / log
      pointers so the next attempt knows what to fix.
    - Once you have successfully run the pipeline, collected the new metrics, and compared them against
      the previous baseline, end your response with the exact line:
          {VALIDATION_SUCCESS_SENTINEL}
      Include the usual summary before that sentinel line.
    - Do NOT output the sentinel unless the metrics truly exist and have been evaluated.
    """).strip()

    if failure_notes:
        prompt += textwrap.dedent(f"""

        Here is the status from the previous validation attempt:

        \"\"\"VALIDATION_STATUS
        {failure_notes}
        VALIDATION_STATUS_END\"\"\"

        Use it to avoid repeating the same debugging steps.
        """).rstrip()

    return prompt + "\n"


# ---------- PHASE PROMPTS ----------

def build_prompt_1() -> str:
    """
    Prompt 1: Conceptual analysis, data realism, leakage, incomplete/incorrect implementations,
    features that don't work, and awareness of hallucinated/legacy cruft.
    Also describes the NFL anytime-TD modeling project explicitly.
    """
    return textwrap.dedent("""
    Context about this project:

    - This is a project that takes NFL play-by-play data and uses it for modeling so that we can
      predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
      in an upcoming game.

    - The goal is to build features and models that are as close as possible to the underlying
      football reality that produces touchdowns: play-calling tendencies, player usage, game state,
      defensive matchups, injuries, roles, red-zone behavior, etc.

    - Training featurization must conceptually and logically match prediction/inference featurization.
      Anything that can only be known in hindsight at inference time (future data, downstream labels,
      or derived artifacts that use future information) is a form of leakage and must be eliminated.

    - Over time, previous model runs and refactors may have left behind:
        * partially-implemented ideas,
        * experimental code paths,
        * hallucinated features,
        * or confusing / inconsistent logic.
      DO NOT assume that all existing code, features, configs, or comments are intentional or correct
      just because they exist. Treat any piece of code or configuration that does not clearly make
      sense in the context of the project as a candidate for cleanup, simplification, or removal.

    Your task in this step:

    Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
    and let me know:

    1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
       predicting anytime TDs in a way that matches how football is actually played.
    2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
       on the field. The goal of getting closer to reality is entirely so that the model is more
       accurate and metrics like AUC or other evaluation metrics improve.
    3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
    4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
    5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
       after the prediction cut-off point, including label-derived features).
    6. Any underlying data sources or features that appear to not be working at all, or are effectively
       noise / dead weight.
    7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
       or features that don't actually exist in the real data or problem domain.

    You should:

    - Be concrete and specific in your findings.
    - Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
      reworked, instead of assuming it must be intentional.
    - Focus on how each issue you find ultimately affects model realism and predictive performance.
    """)


def build_prompt_2(analysis: str) -> str:
    """
    Prompt 2: Rich, detailed plan based on analysis. No edits, just planning.
    """
    return textwrap.dedent(f"""
    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    \"\"\"ANALYSIS_START
    {analysis}
    ANALYSIS_END\"\"\"

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.
    """)


def build_prompt_3(plan: str, progress_notes: Optional[str] = None) -> str:
    """
    Prompt 3: Implement the plan as an agent using codex exec.
    This will be run multiple times (up to MAX_IMPLEMENTATION_ITERATIONS), feeding back progress_notes.
    Includes Git commit/push instructions and revert strategy.
    """
    base = textwrap.dedent(f"""
    Okay, great. Please work like an agent and implement the following plan in this repository.

    Here is the plan:

    \"\"\"PLAN_START
    {plan}
    PLAN_END\"\"\"

    Repository + Git requirements:

    - This repo is under git. At the very beginning of THIS RUN, before editing any files:
        1) Run `git status` to inspect the current state.
        2) If there are uncommitted changes from previous runs, stage and commit them with a concise
           message like `codex: iteration checkpoint` (or a slightly more descriptive variant).
        3) If a remote named `origin` exists and authentication allows, run `git push` so that the
           current state is saved remotely. If push fails due to auth or remote issues, continue with
           local commits only, but do NOT delete history.
        4) Only after ensuring there is a clean commit of the current state should you begin making
           new edits in this run.

    - During this run:
        * Make coherent, incremental commits as you reach logical checkpoints.
        * If you realize that your changes have badly broken the project and you cannot fix them
          cleanly within this run, you may revert to the last good commit (for example, using
          `git reset --hard HEAD` or `git checkout .`), then proceed more conservatively.

    Your tools allow you to:
    - Inspect files in this repo.
    - Edit files.
    - Run shell commands/tests as needed (e.g. project-specific tests, evaluations, or scripts).
    - Use git commands to create commits and, if possible, push them.

    Your goal in THIS RUN:
    - Implement as much of the plan as you reasonably can with high quality.
    - Prioritize correctness, alignment with football reality, and improved model performance
      over speed.
    - Run whatever tests or checks are appropriate to validate your changes.

    At the end of THIS RUN:
    - Give a concise update on your progress.
    - If the ENTIRE plan is fully implemented and validated, include a line with EXACTLY:
        The plan is completely implemented
      (case and spacing exactly as written).
    - If the plan is NOT fully implemented, include a concise progress summary starting with:
        PROGRESS:
      followed by a short description of what you accomplished and what remains.

    Take your time and implement the richest / most complete solution for the pieces you touch in this run.
    """).strip()

    if progress_notes:
        base += textwrap.dedent(f"""

        Here is the progress summary from the last implementation run:

        \"\"\"PROGRESS_FROM_PREVIOUS_RUN
        {progress_notes}
        PROGRESS_FROM_PREVIOUS_RUN_END\"\"\"

        Use this to avoid repeating work and to focus on remaining parts of the plan.
        """).rstrip()

    return base + "\n"


def build_prompt_4(plan: str) -> str:
    """
    Prompt 4: Deep audit to see if the plan is actually fully implemented.
    If fully implemented, Codex should respond ONLY with the sentinel line.
    Includes Git instructions but forbids edits.
    """
    validation_text = build_model_validation_instructions()
    validation_block = textwrap.indent(validation_text, "        ")

    return textwrap.dedent(f"""
    Okay, great.

    Please do a deep dive and audit whether the following plan is completely implemented in this repository
    or if any gaps or incomplete implementations remain.

    Here is the plan:

    \"\"\"PLAN_START
    {plan}
    PLAN_END\"\"\"

    Git / repository behavior for this audit step:

    - This is an audit-only step. You may:
        * Run `git status`, `git log`, and other read-only git commands to inspect history and state.
        * Run tests, evaluation scripts, or other checks to assess completeness and correctness.

    - You must NOT:
        * Edit source files in this step.
        * Stage, commit, reset, or amend git history.
        * Apply any code changes, even minor ones.

    Your audit should:

    - Thoroughly inspect the current code, data flow, feature engineering, and modeling.
    - Check if each step and idea in the plan is fully implemented and correct.
    - Look for:
        * Gaps where something is only partially done.
        * Places where the implementation deviates from what the plan intended.
        * Any remaining data leakage issues the plan was supposed to address.
        * Any conceptual misalignments that the plan was supposed to fix but did not.

    If you find ANY gaps, incomplete implementations, or deviations from the plan:
    - Do NOT make edits in this step.
    - Instead, describe the gaps clearly and concisely so that we can build a new plan for them.

    Model validation + success criteria:
{validation_block}

    If you find that there are NO gaps and the plan is fully implemented:
    - Respond ONLY with this exact line (no extra text, no explanation):

        The plan is completely implemented
    """)


# ---------- PHASE RUNNERS ----------

def phase_1_analysis() -> str:
    prompt = build_prompt_1()
    return run_codex_chat(prompt)


def phase_2_plan(analysis: str) -> str:
    prompt = build_prompt_2(analysis)
    return run_codex_chat(prompt)


def phase_3_implement(plan: str, max_iterations: int = MAX_IMPLEMENTATION_ITERATIONS) -> Tuple[bool, str]:
    """
    Loop Prompt 3 using `codex exec` until either:
      - the agent outputs "The plan is completely implemented", OR
      - max_iterations is reached.

    Returns:
      (is_complete: bool, last_output_or_progress: str)
    """
    progress_notes: Optional[str] = None

    for iteration in range(1, max_iterations + 1):
        print(f"[phase_3_implement] Iteration {iteration}/{max_iterations}")

        prompt = build_prompt_3(plan, progress_notes=progress_notes)
        output = run_codex_exec(prompt)

        # Success sentinel
        if "The plan is completely implemented" in output:
            print("[phase_3_implement] Implementation complete.")
            return True, output

        # Try to extract a PROGRESS: section for the next iteration.
        idx = output.rfind("PROGRESS:")
        if idx != -1:
            progress_notes = output[idx:].strip()
            print("[phase_3_implement] Updated progress notes.")
        else:
            # Fallback: store tail as fuzzy progress
            progress_notes = output[-2000:]
            print("[phase_3_implement] No PROGRESS marker found; using output tail.")

    print("[phase_3_implement] Reached max iterations without completion.")
    return False, progress_notes or ""


def phase_pipeline_validation(plan: str, max_iterations: int = MAX_VALIDATION_ITERATIONS) -> Tuple[bool, str]:
    """
    Loop dedicated to running the pipeline and ensuring metrics are generated.
    """
    failure_notes: Optional[str] = None

    for iteration in range(1, max_iterations + 1):
        print(f"[phase_pipeline_validation] Attempt {iteration}/{max_iterations}")

        prompt = build_prompt_validation(plan, failure_notes=failure_notes)
        output = run_codex_exec(prompt)

        if VALIDATION_SUCCESS_SENTINEL in output:
            print("[phase_pipeline_validation] Pipeline run completed successfully.")
            return True, output

        idx = output.rfind("PROGRESS:")
        if idx != -1:
            failure_notes = output[idx:].strip()
            print("[phase_pipeline_validation] Captured failure notes for next attempt.")
        else:
            failure_notes = output[-2000:]
            print("[phase_pipeline_validation] No PROGRESS marker found; storing output tail.")

    print("[phase_pipeline_validation] Validation loop exhausted attempts without success.")
    return False, failure_notes or ""


def phase_4_audit(plan: str) -> Tuple[bool, str]:
    """
    Run audit ONCE. Never loops.
    Returns (plan_complete, gaps_or_text).

    If plan_complete is True, gaps_or_text is empty.
    If False, gaps_or_text contains the audit description of remaining gaps.
    """
    print("[phase_4_audit] Running a single audit pass…")

    prompt = build_prompt_4(plan)
    output = run_codex_exec(prompt).strip()

    if output == "The plan is completely implemented":
        print("[phase_4_audit] Verified full implementation.")
        return True, ""

    print("[phase_4_audit] Gaps found (audit does not loop).")
    return False, output


# ---------- TOP-LEVEL LOOP ----------

def run_one_full_cycle():
    """
    One full meta-cycle:

    1. Prompt 1: Analysis
    2. Prompt 2: Plan
    3. Prompt 3: Implement (loop up to MAX_IMPLEMENTATION_ITERATIONS)
    4. Validation loop: ensure main.py pipeline run succeeds and metrics are refreshed
    5. Prompt 4: Audit (run exactly once)
       - If gaps remain, treat audit output as a new "analysis" and go back through plan + implementation
         until an audit finally passes (or you stop the script).
    """
    print("\n==================== NEW META-CYCLE ====================\n")

    # PHASE 1: conceptual analysis
    analysis = phase_1_analysis()

    # PHASE 2: plan based on analysis
    plan = phase_2_plan(analysis)

    while True:
        # PHASE 3: implement plan (up to MAX_IMPLEMENTATION_ITERATIONS)
        implemented, progress_or_output = phase_3_implement(plan, max_iterations=MAX_IMPLEMENTATION_ITERATIONS)

        if not implemented:
            print("[run_one_full_cycle] Implementation loop hit iteration limit without full completion.")

        # VALIDATION LOOP
        validation_ok, validation_notes = phase_pipeline_validation(plan, max_iterations=MAX_VALIDATION_ITERATIONS)
        if not validation_ok:
            print("[run_one_full_cycle] Pipeline validation failed; using validation notes as new analysis.")
            analysis = validation_notes
            plan = phase_2_plan(analysis)
            continue

        # PHASE 4: single audit run
        plan_complete, gaps = phase_4_audit(plan)

        if plan_complete:
            print("[run_one_full_cycle] Plan fully implemented and audited.")
            return

        # If audit reveals gaps OR implementation timed out:
        print("[run_one_full_cycle] Gaps remain after implementation + audit.")
        print("[run_one_full_cycle] Treating audit output as new analysis and rebuilding plan.")

        # Treat audit text as new "analysis" focused on the remaining gaps
        analysis = gaps
        plan = phase_2_plan(analysis)
        # Loop back: new plan -> new implementation loop (max 10) -> audit once


def main():
    print(f"[codex_meta_loop] Project directory: {PROJECT_DIR}")
    print(f"[codex_meta_loop] Using codex CLI command: {CODEX_CMD}")
    print(f"[codex_meta_loop] Chat model: {CHAT_MODEL}")
    print(f"[codex_meta_loop] Max implementation iterations per plan: {MAX_IMPLEMENTATION_ITERATIONS}")
    print()

    while True:
        try:
            run_one_full_cycle()
        except KeyboardInterrupt:
            print("\n[codex_meta_loop] Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"[codex_meta_loop] ERROR during meta-cycle: {e}")

        print(f"[codex_meta_loop] Sleeping {SLEEP_BETWEEN_CYCLES} seconds before next meta-cycle...\n")
        time.sleep(SLEEP_BETWEEN_CYCLES)


if __name__ == "__main__":
    main()

