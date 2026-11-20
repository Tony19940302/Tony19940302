---
name: code-review
description: Independently review BattleTracker changes and assess their validation evidence.
---

# code-review

1. Read the applicable requirements and inspect the actual diff and test results.
2. Prioritize incorrect resource attribution, duplicate or missing counts, replay inconsistency, and data leakage. Check malformed input and unknown states.
3. Report each actionable finding with evidence, severity, location, and reproduction. Report residual risk and missing coverage when review finds no defects.
4. Re-run affected checks after fixes and record evidence in the review report.
