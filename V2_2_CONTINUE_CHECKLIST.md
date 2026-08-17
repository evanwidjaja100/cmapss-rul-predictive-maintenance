# V2.2 Cleanup — Continue Checklist

Picked up on 2026-08-17 after the "V2.2 final cleanup" work session.
Master plan: `C_MAPSS_V2_2_FINAL_CLEANUP_AGENT_PLAN.md`.
Tracker: `V2_2_FINAL_CLEANUP_PLAN.md` (I-1..I-17; most rows DONE).

## Already done (do not redo)

- I-2/I-3/I-6/I-7/I-9/I-10/I-11/I-12/I-15/I-5/I-13/I-14/I-16 DONE.
- Clean-checkout simulation (temp tree, CI command): **134 passed, 11 skipped,
  9 deselected, 0 failed**. Local full suite: **154 passed**. Local artifact-free:
  **145 passed, 9 deselected**. README/PROJECT_SPEC/report now carry these.
- All recomputations done: freeze (provenance metadata), posthoc, conformal,
  sensitivity (prefix-only), error analysis, config regeneration.
- `tests/test_v2_2_cleanup.py` (19 tests incl. 4 needs_artifacts falsification
  tests) green. CI yml verified: runs `-m "not needs_artifacts"` — matches README.

## Remaining steps (in order)

1. **I-1 — track `experiments/v2_2`**: add `!experiments/v2_2/` to `.gitignore`
   (after `experiments/*`); run `git status --short` and confirm the v2_2 CSV/JSON
   audit files appear. Do NOT commit `data/`, `models/`, or `.opencode/` state.
2. **I-8 — FD004 re-freeze (optional)**: script is config-driven now
   (`resolve_model_config`, dynamic `fd004_gru_w{window}_{loss}_cond{variant}.keras`);
   posthoc reads window/loss from YAML. Existing `models/v2_2/fd004_gru_w45_huber_condC.keras`
   already matches — re-freeze only if you want fresh metadata.
3. **I-4 — verify freeze metadata**: check `models/v2_2/fd001_final_fit_metadata.json`
   (or wherever the re-freeze wrote it) contains git_commit/git_is_dirty/git_diff_hash/
   source_tree_hash. CHANGELOG already documents historical dirty-worktree provenance.
4. **I-17 — mark DONE**: CI parity verified (ci.yml line 21 == README command).
5. **Final test runs** (numbers already recorded, re-run for confidence):
   `.venv\Scripts\python.exe -m pytest` (expect 154 passed).
6. **Update tracker**: set I-1 DONE (after commit), I-17 DONE, I-8 note.
7. **Commit** (logical commits or one):
   `git add` configs/, scripts/, src/, tests/, README.md, PROJECT_SPEC.md,
   CHANGELOG.md, reports/, experiments/v2_2/, .gitignore,
   V2_2_FINAL_CLEANUP_PLAN.md, C_MAPSS_V2_2_FINAL_CLEANUP_AGENT_PLAN.md.
   Style: short summary line (see `git log --oneline -5`).
8. **Final report** (`reports/v2_2_final_report.md` or new file per master plan
   §27): sections 27.1–27.12 (what was checked, real test counts, recomputed
   numbers, provenance disclosure, remaining limitations). End with
   `GOAL_COMPLETE`.
9. **Exit checklist**: walk master plan §26 (40 items), evidence per item.

## Key numbers to cite

- FD001 post-hoc: RMSE 26.2526, MAE 21.2347, R² 0.6009, NASA 60,963.79.
- CV NASA/engine xgb_w90_d6: 368.0221. Accuracy champion lstm_w60_huber (RMSE 26.19).
- FD004 variant C post-hoc: RMSE 33.6579, R² 0.6189, NASA 1,545,798.5.
- Conformal q: 66.2097 / 44.7955 / 41.4224 (α 0.1/0.2/0.3), 15 engines.
- Sensitivity top: 4, 11, 3, 9, 12, 7, 20 (prefix-only baseline).
- Tests: local 154 / artifact-free 145+9 / clean checkout 134+11+9.

## Environment notes

- Use only `.venv\Scripts\python.exe` (verified to resolve from repo root).
- PowerShell 5.1: no `&&`; no `rg`; use grep tool or `Select-String`.
- Edit tool quirks: it strips leading indentation from the first matched line —
  re-read the file after multi-line edits.
- Clean-checkout recipe: robocopy /XD with FULL PATHS (bare `data` also
  excludes `src/rul_prediction/data`); run pytest with
  `$env:PYTHONPATH = "<tmp>\src"` so the temp src shadows the editable install.