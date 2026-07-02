# NUS RA Task — RB + SA Answer-or-Abstain Audit

This is my submission for the take-home study on the RB + SA answer-or-abstain system. The brief and the project summary I was given are included here as `Next-Stage-Brief.docx` and `message (3).txt`; the actual write-up is in `report.md`.

`sa-module-v2/` is the codebase I was given, with one addition of my own: a fourth test condition, `with_memory_sa_scrambled`, added to both `run_selfaware.py` and `run_falseqa.py`, along with a paired significance test (McNemar) and a fuller per-condition breakdown in the saved results. The idea and the reasoning behind it are explained in the report; in short, it checks whether the system's reported gains actually come from retrieving *relevant* past failures, or whether any failure-lesson-shaped content in the prompt produces much the same effect. `memory_bank/`, `results/`, and `checkpoints/` under `sa-module-v2/` hold the actual output of running this — a full-scale reproduction of the original SelfAware and FalseQA experiments plus the new condition, not a toy run.

`Results/` is everything that was shared with me alongside the code: the reported summaries for all four benchmarks, the checkpoints from the Gemini and Qwen3 runs, and the underlying datasets. A fair amount of what's in there — the KUQ and UnknownBench numbers, the Gemini/Qwen3 results, a few benchmark runs that never made it into the README — has no accompanying source code, which is itself one of the findings in the report.

To reproduce the new experiment, see the "How to reproduce" section of `report.md`; it needs a GPU and about an hour for the full-scale run, or a few minutes at reduced scale.

I used Claude (Claude Code) through this exercise, for reading and cross-referencing the code and result files, running literature searches, implementing the new test condition, running the experiments, and drafting the report. That's noted again at the end of `report.md`. The findings and the decisions about what to check and how to read the results are mine.
