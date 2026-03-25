# AGENTS.md

## Purpose

This repository extends **Instant Policy** with a **history-aware track module**.
Agents working in this repo must preserve the base graph-diffusion policy semantics while making changes incrementally and verifiably.

The codebase currently contains:

- The original Instant Policy path: `ip/train.py`, `ip/eval.py`, `ip/models/model.py`, `ip/models/graph_rep.py`
- The history-aware extension path: `ip/train_history.py`, `ip/eval_history.py`, `ip/deployment_history.py`, `ip/models/model_history.py`, `ip/models/graph_rep_history.py`, `ip/models/track_encoder.py`

Do not assume the history path is a full paper-faithful implementation. It is an extension on top of the original Instant Policy backbone.

## Core Architecture

### Base policy

The base policy is the Instant Policy graph-diffusion model:

- Scene observations are encoded from segmented point clouds
- Demos, current observation, and future action hypotheses are represented in one heterogeneous graph
- Action prediction is done through diffusion over future gripper/action nodes

Core files:

- `ip/models/model.py`
- `ip/models/diffusion.py`
- `ip/models/graph_rep.py`

### History-aware extension

The history-aware path adds `track` nodes to the original graph:

- Current history is represented by `current_track_seq`, `current_track_valid`, `current_track_age_sec`
- These are encoded into `track_node_embds`
- Track nodes are injected into the graph alongside `scene` and `gripper` nodes

Core files:

- `ip/models/model_history.py`
- `ip/models/graph_rep_history.py`
- `ip/models/graph_rep_haigd.py`
- `ip/models/track_encoder.py`
- `ip/utils/track_buffer.py`

Important: the current implementation only feeds **current-observation history**, not demo-side history.

## Repo Facts Agents Must Preserve

1. The original Instant Policy path must remain runnable unless the task explicitly targets only the history-aware fork.
2. The meaning of action tensors must not change silently.
3. The diffusion label semantics must not change silently.
4. The repo currently relies on relative end-effector transforms for actions.
5. `current_track_*` is part of the history data contract.
6. Training-time history tracks and deployment-time history tracks currently come from different sources:
   - Offline/pseudo-data uses object-state-derived tracks
   - Deployment uses `TrackBuffer`, which is a lightweight online heuristic
7. This train/deploy mismatch is a known risk. Any history-related change must explicitly consider it.

## Data Contract

History-aware samples are expected to include at least:

- `pos_demos`
- `graps_demos`
- `batch_demos`
- `pos_obs`
- `batch_pos_obs`
- `demo_T_w_es`
- `current_grip`
- `actions`
- `actions_grip`
- `T_w_e`
- `current_track_seq`
- `current_track_valid`
- `current_track_age_sec`

Relevant files:

- `ip/utils/data_proc.py`
- `ip/utils/running_dataset.py`
- `ip/utils/running_dataset_history.py`

Do not change saved sample field names without updating all loaders, training code, eval code, and deployment code.

## How To Work In This Repo

### For algorithmic or mathematical tasks

Before editing code, first produce a short implementation spec containing:

- Problem statement
- Current behavior
- Desired behavior
- Tensor shapes
- Loss or objective changes
- Graph topology changes
- Training/inference implications
- Acceptance criteria

Do not jump directly into broad code changes for mathematically unclear tasks.

### For implementation tasks

Make changes in small steps:

1. Data contract
2. Model wiring
3. Training loop
4. Eval/deployment
5. Tests and smoke checks

Avoid large mixed refactors.

### When touching history logic

Always state whether the change affects:

- Offline pseudo-data generation
- Dataset loading
- Graph construction
- Training only
- Inference only
- Deployment only

If a change affects only one of these, call out the resulting mismatch risk.

## Modification Rules

### Allowed by default

- Add or improve tests
- Fix script breakages
- Improve configuration plumbing
- Tighten shape checks
- Add small utility functions
- Improve comments where code is non-obvious

### Be careful with

- `ip/models/diffusion.py`
- `ip/models/diffusion_history.py`
- `ip/models/model.py`
- `ip/models/model_history.py`
- `ip/models/graph_rep.py`
- `ip/models/graph_rep_history.py`
- `ip/utils/data_proc.py`

These files define core semantics. Do not casually rename tensors, reorder dimensions, or change normalization behavior.

### Do not do this unless explicitly required

- Rewrite both base and history paths in one patch
- Change action normalization ranges without updating all related code
- Replace the deployment tracker with a different tracking stack in an incidental task
- Remove compatibility with existing saved `.pt` samples without migration logic
- Introduce broad stylistic refactors unrelated to the requested fix

## Known Gaps And Risks

Agents should be aware of these existing issues:

1. `eval_history.py` and `deployment_history.py` expect `./checkpoints/config.pkl`, but this file may be missing in a local workspace.
2. `deployment_history.py` uses `os.path.exists` and must import `os`.
3. `running_dataset_history.py` may need attention under newer PyTorch defaults around `torch.load(..., weights_only=True)`.
4. Several history-extension config flags are defined but not fully enforced as topology switches.
5. Curriculum dropout scaffolding exists, but scheduling may not currently be stepped during training.
6. Online history tracking is much weaker than the tracker design described in the paper.

Do not hide these issues in summaries if they are relevant to the task.

## Preferred Commands

Use these commands first when validating work:

```bash
python -m compileall ip
```

```bash
python ip/train_history.py --help
```

```bash
python ip/eval_history.py --help
```

If you change dataset serialization or loading, inspect one saved sample and confirm expected keys and shapes.

## Validation Expectations

For code changes, validate at the smallest useful level:

- Syntax check for touched Python files
- Shape/path sanity for changed data flow
- One targeted smoke check for the touched entrypoint when practical

If you cannot run a full validation, state exactly what was not run and why.

## Expected Output Style For Agents

When finishing a task, report:

- What changed
- What was verified
- What remains risky or unverified

For reviews, prioritize:

- Behavioral regressions
- Shape/semantic mismatches
- Training/inference inconsistencies
- Hidden assumptions around history inputs

## File Ownership Heuristics

Use these boundaries when possible:

- Data generation and serialization:
  `ip/generate_pseudo_data.py`, `ip/generate_pseudo_data_new.py`, `ip/utils/data_proc.py`, `ip/utils/running_dataset_history.py`
- History representation:
  `ip/models/track_encoder.py`, `ip/models/graph_rep_history.py`, `ip/models/graph_rep_haigd.py`
- Policy and diffusion logic:
  `ip/models/model_history.py`, `ip/models/diffusion_history.py`
- Deployment and rollout:
  `ip/deployment_history.py`, `ip/utils/rl_bench_utils_history.py`, `ip/utils/track_buffer.py`

Minimize cross-boundary edits unless the task truly requires them.
