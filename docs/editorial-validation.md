# Editorial validation checkpoint

All sixteen earlier cases have agent-reviewed reference corrections in `config/editorial/regression_cases.json`. They distinguish invented observations, study scope, causal inference, policy opinions, source attribution and unproven policy effects. These are development labels, not independent truth certification or human calibration.

The editor retains confidence thresholds of 0.8 for evidence and 0.9 for changed voice. It recognizes a narrow firsthand-framing correction by exact remaining words, keeps turn metadata and cut boundaries immutable, rejects control characters, and allows one checkpointed repair attempt. Rejected proposals retain originals and stop before speech. Unchanged speaker wording needs no probabilistic voice judgment.

## Live results

| Test | Scope | Accepted edits | Turns requiring review | Mechanical acceptance |
| --- | --- | ---: | ---: | --- |
| Earlier v3 baseline | Fresh author generation | 4 | 16 | Passed |
| v4 experiment | Fresh Gemini generation | 0 | 22 | Passed |
| v5 experiment | Fresh OpenAI generation | 1 | 32 | Passed |
| v6 guard replay | Saved proposals, live JEV validation | 2 | 21 | Passed |

The experiments did not establish better unattended acceptance. OpenAI drafting was not retained as the default. The safeguards fix specific failure modes; they are not a claim that overall editorial quality is solved. The replay is explicitly distinguished from fresh generation.

The two accepted replay edits survived one isolated full-context polish pass unchanged. All 38 turns were processed, metadata stayed identical, and unresolved originals were never approved or voiced. The isolated `polish_validation` artifacts cannot unlock the production pipeline.

77 offline tests and Ruff pass. Tests cover semantic voice changes, fabricated/overconfident forms, citation leaks, control characters, stale evidence, bounded repair/recovery and gates stopping paid downstream work. The existing user-approved school-phone episode remains unchanged. Further production calibration and source-scoped drafting work remain necessary.

The cumulative local test reservation ledger reached $23.6622 under the user-authorized $25 cap. Provider billing is separate. Generated run artifacts and credentials are excluded from Git.

GitHub CI also passed on Python 3.11 and 3.14 for the published source commit: [verified test run](https://github.com/CyberSecDef/crosstalk/actions/runs/37408283271).
