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

from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
import textwrap
from typing import Tuple, Optional
from shutil import which

# ---------------- CONFIG ----------------

PROJECT_DIR = Path(".").resolve()
SLEEP_BETWEEN_CYCLES = 30

CHAT_MODEL = "gpt-5.1"
CODEX_CMD = "codex"

MAX_IMPLEMENTATION_ITERATIONS = 10
MAX_VALIDATION_ITERATIONS = 5
VALIDATION_SUCCESS_SENTINEL = "PIPELINE VALIDATION COMPLETE"

PIPELINE_VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"
PIPELINE_ENTRYPOINT = PROJECT_DIR / "main.py"
PIPELINE_EVAL_COMMAND = f"cd {PROJECT_DIR} && {PIPELINE_VENV_PYTHON} {PIPELINE_ENTRYPOINT}"

METRICS_ROOT = PROJECT_DIR / "output" / "metrics"
PRIMARY_TARGET = "anytime_td"
PRIMARY_FAMILY = "xgboost"
PRIMARY_METRIC_FILENAME = "metrics.yaml"
PRIMARY_IMPORTANCE_FILENAME = "feature_importance.json"
CUTOFF_SUMMARY_FILE = METRICS_ROOT / "cutoff_backtest_summary.csv"

# Streaming/robustness
HEARTBEAT_SECONDS = 20  # print a heartbeat if codex produces no output for this many seconds

# Track current subprocess so Ctrl-C can kill it
_CURRENT_PROC: Optional[subprocess.Popen] = None


# ---------------- UTILITIES ----------------

def _ensure_codex_cli_available() -> None:
    if which(CODEX_CMD) is None:
        raise RuntimeError(
            f"Unable to find '{CODEX_CMD}' on PATH. "
            "Install the Codex CLI or update CODEX_CMD in codex_meta_loop.py."
        )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the whole process group (Codex may spawn children)."""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    # Give it a moment, then hard kill if needed
    t0 = time.time()
    while proc.poll() is None and (time.time() - t0) < 2.0:
        time.sleep(0.05)

    try:
        if proc.poll() is None:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
    except Exception:
        pass


def _codex_cmd_for_exec(*, model: str, sandbox: str, approval: str) -> list[str]:
    """
    IMPORTANT: The Codex CLI treats these as GLOBAL flags, which must be placed BEFORE the subcommand.
    This avoids: "unexpected argument '--ask-for-approval' found".
    """
    # Using short flags:
    # -a = ask-for-approval
    # -s = sandbox
    # -m = model
    #
    # `exec -` means: read the prompt from stdin.
    return [
        CODEX_CMD,
        "-a", approval,
        "-s", sandbox,
        "-m", model,
        "exec",
        "-",
    ]


def _run_codex_streaming(cmd: list[str], prompt: str) -> Tuple[int, str]:
    """
    Run codex with:
      - stdin as a pipe (so we can send prompt then close -> EOF)
      - stdout/stderr connected to a PTY (so Codex thinks it's a terminal and streams output)

    Returns (exit_code, full_output_text).
    """
    global _CURRENT_PROC

    print("\n================ PROMPT SENT TO CODEX ================\n")
    print(prompt)
    print("\n======================================================\n")
    sys.stdout.flush()

    master_fd, slave_fd = pty.openpty()

    # start_new_session=True => proc becomes its own process group (killpg works)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdin=subprocess.PIPE,
        stdout=slave_fd,
        stderr=slave_fd,
        text=True,
        start_new_session=True,
        env={**os.environ, "TERM": os.environ.get("TERM", "xterm-256color")},
    )
    _CURRENT_PROC = proc

    # Parent doesn't need the slave
    os.close(slave_fd)

    # Feed prompt and close stdin so `codex exec -` gets EOF and proceeds
    try:
        if proc.stdin is not None:
            proc.stdin.write(prompt if prompt.endswith("\n") else prompt + "\n")
            proc.stdin.flush()
            proc.stdin.close()
    except Exception:
        pass

    print("\n================ CODEX OUTPUT (streaming) ================\n")
    sys.stdout.flush()

    chunks: list[str] = []
    last_output_time = time.time()

    try:
        while True:
            # If proc exited and there's nothing left to read, break
            if proc.poll() is not None:
                # drain any remaining output quickly
                drained_any = False
                while True:
                    r, _, _ = select.select([master_fd], [], [], 0.0)
                    if not r:
                        break
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    drained_any = True
                    s = data.decode(errors="replace")
                    chunks.append(s)
                    sys.stdout.write(s)
                    sys.stdout.flush()
                if not drained_any:
                    break

            r, _, _ = select.select([master_fd], [], [], 0.2)
            if r:
                data = os.read(master_fd, 4096)
                if not data:
                    break
                s = data.decode(errors="replace")
                chunks.append(s)
                sys.stdout.write(s)
                sys.stdout.flush()
                last_output_time = time.time()
            else:
                # heartbeat
                now = time.time()
                if (now - last_output_time) >= HEARTBEAT_SECONDS:
                    elapsed = int(now - last_output_time)
                    sys.stdout.write(f"\n[codex_meta_loop] (heartbeat) no output from codex for {elapsed}s...\n")
                    sys.stdout.flush()
                    last_output_time = now

    except KeyboardInterrupt:
        # guaranteed kill of codex + children
        sys.stdout.write("\n[codex_meta_loop] KeyboardInterrupt: terminating codex...\n")
        sys.stdout.flush()
        _kill_process_group(proc)
        raise
    finally:
        try:
            os.close(master_fd)
        except Exception:
            pass
        _CURRENT_PROC = None

    rc = proc.wait()
    out = "".join(chunks)

    print("\n================ END CODEX OUTPUT ================\n")
    sys.stdout.flush()
    return rc, out


# ---------------- CODEX WRAPPERS ----------------

def run_codex_chat(prompt: str) -> str:
    """
    We do NOT use `codex chat` because it requires a TTY ("stdout is not a terminal").
    Instead we use `codex exec -` with read-only sandbox for analysis/plan.
    """
    _ensure_codex_cli_available()
    print("\n[run_codex_chat] Starting (exec - read-only) call...\n")
    cmd = _codex_cmd_for_exec(model=CHAT_MODEL, sandbox="read-only", approval="never")
    exit_code, out = _run_codex_streaming(cmd, prompt)
    print("[run_codex_chat] Finished. Exit code:", exit_code)
    if exit_code != 0:
        raise RuntimeError(f"'codex exec' exited with {exit_code}:\n{out}")
    return out


def run_codex_exec(prompt: str, *, sandbox: str = "workspace-write") -> str:
    """
    `codex exec -` agent mode.
    sandbox:
      - workspace-write for implementation/validation (so it can edit & run).
      - read-only if you want strict safety (but then tests/pipeline that write artifacts may fail).
    """
    _ensure_codex_cli_available()
    print(f"\n[run_codex_exec] Starting exec call... (sandbox={sandbox})\n")
    cmd = _codex_cmd_for_exec(model=CHAT_MODEL, sandbox=sandbox, approval="never")
    exit_code, out = _run_codex_streaming(cmd, prompt)
    print("[run_codex_exec] Finished. Exit code:", exit_code)
    if exit_code != 0:
        raise RuntimeError(f"'codex exec' exited with {exit_code}:\n{out}")
    return out


# ---------------- METRICS UTILS ----------------

def _latest_metric_run_dir(target: str = PRIMARY_TARGET, family: str = PRIMARY_FAMILY) -> Optional[Path]:
    base = METRICS_ROOT / target / family
    if not base.exists():
        return None
    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    return max(run_dirs, key=lambda p: p.stat().st_mtime) if run_dirs else None


def build_model_validation_instructions() -> str:
    latest_dir = _latest_metric_run_dir()

    if latest_dir:
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
    validation_text = build_model_validation_instructions()
    validation_block = textwrap.indent(validation_text, "    ")

    prompt = textwrap.dedent(f"""
    You are now in the validation loop. Your job is to make sure the full pipeline command runs
    successfully and produces refreshed metrics artifacts.

    Plan context (for reference only, do NOT re-hash it unless needed):

    <BEGIN_PLAN_START>
    {plan}
    <END_PLAN_END>

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

        <BEGIN_VALIDATION_STATUS>
        {failure_notes}
        <END_VALIDATION_STATUS>

        Use it to avoid repeating the same debugging steps.
        """).rstrip()

    return prompt + "\n"


# ---------- PHASE PROMPTS (UNCHANGED TEXT) ----------

def build_prompt_1() -> str:
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
    return textwrap.dedent(f"""
    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    {analysis}
    <END_ANALYSIS_END>

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
    base = textwrap.dedent(f"""
    Okay, great. Please work like an agent and implement the following plan in this repository.

    Here is the plan:

    <BEGIN_PLAN_START>
    {plan}
    <END_PLAN_END>

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

        <BEGIN_PROGRESS_FROM_PREVIOUS_RUN>
        {progress_notes}
        <END_PROGRESS_FROM_PREVIOUS_RUN>

        Use this to avoid repeating work and to focus on remaining parts of the plan.
        """).rstrip()

    return base + "\n"


def build_prompt_4(plan: str) -> str:
    validation_text = build_model_validation_instructions()
    validation_block = textwrap.indent(validation_text, "        ")

    return textwrap.dedent(f"""
    Okay, great.

    Please do a deep dive and audit whether the following plan is completely implemented in this repository
    or if any gaps or incomplete implementations remain.

    Here is the plan:

    <BEGIN_PLAN_START>
    {plan}
    <END_PLAN_END>

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
    progress_notes: Optional[str] = None

    for iteration in range(1, max_iterations + 1):
        print(f"[phase_3_implement] Iteration {iteration}/{max_iterations}")

        prompt = build_prompt_3(plan, progress_notes=progress_notes)
        output = run_codex_exec(prompt, sandbox="workspace-write")

        if "The plan is completely implemented" in output:
            print("[phase_3_implement] Implementation complete.")
            return True, output

        idx = output.rfind("PROGRESS:")
        if idx != -1:
            progress_notes = output[idx:].strip()
            print("[phase_3_implement] Updated progress notes.")
        else:
            progress_notes = output[-2000:]
            print("[phase_3_implement] No PROGRESS marker found; using output tail.")

    print("[phase_3_implement] Reached max iterations without completion.")
    return False, progress_notes or ""


def phase_pipeline_validation(plan: str, max_iterations: int = MAX_VALIDATION_ITERATIONS) -> Tuple[bool, str]:
    failure_notes: Optional[str] = None

    for iteration in range(1, max_iterations + 1):
        print(f"[phase_pipeline_validation] Attempt {iteration}/{max_iterations}")

        prompt = build_prompt_validation(plan, failure_notes=failure_notes)
        output = run_codex_exec(prompt, sandbox="workspace-write")

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
    print("[phase_4_audit] Running a single audit pass…")

    prompt = build_prompt_4(plan)

    # NOTE: audit may run tests/scripts that write metrics/artifacts. Using workspace-write is pragmatic.
    # The prompt forbids code edits; sandbox here is about shell-command policy, not "editing discipline".
    output = run_codex_exec(prompt, sandbox="workspace-write").strip()

    if output == "The plan is completely implemented":
        print("[phase_4_audit] Verified full implementation.")
        return True, ""

    print("[phase_4_audit] Gaps found (audit does not loop).")
    return False, output


# ---------- TOP-LEVEL LOOP ----------

def run_one_full_cycle():
    print("\n==================== NEW META-CYCLE ====================\n")

    analysis = phase_1_analysis()
    plan = phase_2_plan(analysis)

    while True:
        implemented, _progress_or_output = phase_3_implement(plan, max_iterations=MAX_IMPLEMENTATION_ITERATIONS)
        if not implemented:
            print("[run_one_full_cycle] Implementation loop hit iteration limit without full completion.")

        validation_ok, validation_notes = phase_pipeline_validation(plan, max_iterations=MAX_VALIDATION_ITERATIONS)
        if not validation_ok:
            print("[run_one_full_cycle] Pipeline validation failed; using validation notes as new analysis.")
            analysis = validation_notes
            plan = phase_2_plan(analysis)
            continue

        plan_complete, gaps = phase_4_audit(plan)
        if plan_complete:
            print("[run_one_full_cycle] Plan fully implemented and audited.")
            return

        print("[run_one_full_cycle] Gaps remain after implementation + audit.")
        print("[run_one_full_cycle] Treating audit output as new analysis and rebuilding plan.")
        analysis = gaps
        plan = phase_2_plan(analysis)


def main():
    _ensure_codex_cli_available()

    print(f"[codex_meta_loop] Project directory: {PROJECT_DIR}")
    print(f"[codex_meta_loop] Using codex CLI command: {CODEX_CMD}")
    print(f"[codex_meta_loop] Chat model: {CHAT_MODEL}")
    print(f"[codex_meta_loop] Max implementation iterations per plan: {MAX_IMPLEMENTATION_ITERATIONS}")
    print()

    # Ensure Ctrl-C kills codex too
    def _sigint_handler(sig, frame):
        global _CURRENT_PROC
        if _CURRENT_PROC is not None and _CURRENT_PROC.poll() is None:
            sys.stdout.write("\n[codex_meta_loop] SIGINT received: terminating codex...\n")
            sys.stdout.flush()
            _kill_process_group(_CURRENT_PROC)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

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
