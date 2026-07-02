# Report: RB + SA Answer-or-Abstain System

**Saisab Sadhu**

---

In this report we cover `sa-module-v2/`, the only part of the shared materials with runnable code. KUQ, UnknownBench, and the Gemini/Qwen3 runs are discussed from their result files only as no source code for those was shared.

---

## 1. What the code does

SA is a Self-Awareness layer built on top of ReasoningBank (RB), an existing agent-memory framework. The setup is straightforward: during training, each question goes to Qwen2.5-7B-Instruct once at temperature 0, with a system prompt already telling it to hedge on subjective or unanswerable questions. The response then gets labelled by a deterministic function — roughly forty hedge phrases crossed against the dataset's own `answerable` flag, producing one of four outcomes: `CORRECT`, `KNOWLEDGE_BOUNDARY_MISSED` (attempted something it should have declined), `FALSE_IDK` (declined something it should have answered), or `FACTUAL_ERROR` (attempted, wrong). The model never judges itself here; this is entirely rule-based.

Each episode produces two memories: a generic RB strategy note, and an SA lesson written with a prompt specific to the failure type. The failure type is handed to the extraction prompt directly rather than inferred. Episodes accumulate in a low-level store, and every twenty of them the system groups by failure type and a free-text domain tag the model generated, promoting any recurring pattern - three or more times into a standing guideline that gets injected into every future prompt.

At test time, the same held-out questions run under three cumulative conditions: bare baseline, `with_memory` (top-3 RB notes by embedding similarity), and `with_memory_sa` (top-2 RB notes, top-2 retrieved SA failure episodes, and all active guidelines). Each response is scored with the same detector, and abstention F1 is the headline metric — where "should have abstained" is the positive class. `run_falseqa.py` mirrors this almost exactly, built around false-premise questions instead.

---

## 2. Pipeline: 

Data arrives pre-split into train and test JSON. There is no code to regenerate that split. Training -- ask, label, extract, store - is the only point where ground truth touches the system. Testing then repeats across the three conditions on the same fixed set and aggregates into accuracy, precision, recall, F1, and a Wilson interval.

Two things to be noted. First, the same keyword detector both generates the training signal and scores the test outcome. There is no independent judge anywhere in the loop. Second, F1 here scores only the abstain-or-answer decision — it says nothing about whether attempted answers are actually correct. The headline metric is narrower than it looks.

---

## 3. Assumptions

The system rests on several assumptions that do not all hold cleanly.

It assumes the forty-phrase keyword list reliably distinguishes genuine uncertainty from confident hedging and trusts the same detector as both evaluation ground truth and training signal. It assumes comparing `with_memory` against `with_memory_sa` isolates SA's contribution, which is not quite right given that retrieval depth differs between the two conditions (more on this in §4.3). It also assumes a single F1 at one operating point is sufficient to call a model better calibrated. The selective-prediction literature [Selective Question Answering under Domain Shift](https://aclanthology.org/2020.acl-main.503/) (Kamath et al., ACL 2020) generally prefers a risk-coverage curve for exactly this reason. And it assumes the train/test split and the unanswerable label set are clean, which §4.6 shows does not fully hold.

---

## 4. Problems, weak points, methodological risks

### 4.1 The gain looks like the model simply hedging more, not discriminating better

This is the most consequential finding. Per-item SelfAware counts across conditions (CORRECT / FACTUAL\_ERROR / FALSE\_IDK / KNOWLEDGE\_BOUNDARY\_MISSED):

- Baseline: 184 / 162 / 57 / 97
- RB: 190 / 130 / 110 / 70
- RB+SA: 206 / 99 / 141 / 54

FALSE\_IDK nearly triples. Overall abstention rises from roughly 23% to 48%. The same pattern appears across all four headline datasets: answerable-side accuracy collapses as F1 rises — 83.5%→59.2% on SelfAware, 62.2%→50.0% on FalseQA, 82.7%→60.4% on KUQ, 94.0%→58.9% on UnknownBench. The two largest claimed gains sit exactly where baseline recall on the abstain class was weakest; where indiscriminate over-abstention pays off most in F1 terms.

A likely mechanism: the lessons and guidelines injected into the prompt are phrased using the same hedge vocabulary the detector looks for, so a primed model can satisfy the scorer by echoing surface form rather than reasoning correctly §6a and §6b test this directly.

### 4.2 The RB vs. RB+SA comparison is not a clean ablation

`with_memory` retrieves three RB strategies. `with_memory_sa` retrieves two. The reported SA uplift therefore conflates SA's contribution with a reduction in RB retrieval depth. These are not separated anywhere in the current setup.

Also the Gemini checkpoints folder includes a FalseQA result: F1 falling from 0.443 to 0.391 under RB+SA. This is not in the README also

### 4.3 Label quality issues in SelfAware

The SelfAware unanswerable set is mostly genuine subjective or philosophical questions, but includes corrupted GSM8K word problems — one switches from "bears" to "the dogs" mid-sentence — that are unanswerable only because the sentence does not parse.

### 4.4 Smaller issues

- Setup instructions do not reproduce as mentioned: flask vs. llama-server naming inconsistencies, a `simple_server.py` and a `download_selfaware` module that both do not exist.
- The factual-correctness check uses a loose substring match. It does not touch the headline F1 but corrupts which lessons get extracted. Also, every episode's confidence is hardcoded to 0.5, yet `SelfModel` computes a calibration gap from it regardless — meaninglessly.
- Nine failure types are declared in `memory.py`; but only four are ever produced

---

## 5. Proposed robustness check

The open question is whether the RB+SA gain comes from the relevance of retrieved failure lessons — the system's actual claim or whether any failure-lesson-shaped content produces the same shift because the real effect is general hedge-priming.

The check adds a fourth condition: `with_memory_sa_scrambled`. Identical to `with_memory_sa` in every respect — same RB depth, same guidelines — except SA episodic retrieval is queried with a different, unrelated test question, paired through a fixed derangement so no question retrieves its own lessons. One variable changes. If the two conditions are statistically indistinguishable under a paired test, that points at presence rather than relevance. If `with_memory_sa` wins clearly, that supports the claimed mechanism.

I considered a wider design (a naive "hedge more" control, a matched-abstention-rate comparison, a full risk–coverage curve) but kept this focused deliberately, in line with what the brief asked for.

---

## 6. What I implemented

Both run scripts now have a `build_derangement` function and a `with_memory_sa_scrambled` branch, plus a paired continuity-corrected `mcnemar_test`, appropriate since every condition runs on the same ordered test set. Results now report abstention rate per condition and McNemar test between the real and scrambled SA conditions, and against baseline for context. `run_selfaware.py` also gained `--train-limit` / `--test-limit` flags defaulting to the paper's own 300/500.

No memory-bank checkpoint was shared, so this rebuilds RB/SA memory from scratch — which doubles as an independent reproduction check before the new condition is added. Since the setup instructions did not run as-is, I served `vllm serve Qwen/Qwen2.5-7B-Instruct` directly with `--served-model-name` matching what `llm_client.py` expects, so the existing client needed no changes.

### 6a. SelfAware results

Full scale, train=300, test=500, memory rebuilt from scratch (Used an H200)

| Condition | F1 | Precision | Recall | Answerable Acc | Unanswerable Acc |
|---|---|---|---|---|---|
| baseline | 0.427 | 0.492 | 0.377 | 82.7% | 37.7% |
| RB only | 0.502 | 0.455 | 0.558 | 70.2% | 55.8% |
| RB+SA | 0.524 | 0.442 | 0.643 | 63.9% | 64.3% |
| RB+SA, scrambled | 0.495 | 0.422 | 0.597 | 63.6% | 59.7% |

McNemar: RB+SA vs. scrambled p=0.428 (not significant); RB+SA vs. baseline p=0.045 (significant); RB vs. baseline p=0.203 (not significant).

The from-scratch numbers sit close to the reported 0.425 / 0.483 / 0.506, so the effect direction reproduces. But RB+SA beats the scrambled placebo by only 0.029 F1, which is not distinguishable from noise at n=500. RB+SA against baseline is significant, but that is consistent with general content priming rather than relevant retrieval. But one caveat, scrambling only touched per-episode retrieval. The twelve standing guidelines were identical in both conditions, so this does not rule out the guidelines carrying real weight.

### 6b. FalseQA results

Full scale, train=300, test=300:

| Condition | F1 | Precision | Recall | False-premise Acc | True-premise Acc |
|---|---|---|---|---|---|
| baseline | 0.673 | 0.621 | 0.735 | 73.5% | 62.8% |
| RB only | 0.676 | 0.534 | 0.919 | 91.9% | 33.5% |
| RB+SA | 0.729 | 0.584 | 0.971 | 97.1% | 42.7% |
| RB+SA, scrambled | 0.726 | 0.575 | 0.985 | 98.5% | 39.6% |

McNemar: RB+SA vs. scrambled p=0.719 (not significant); RB+SA vs. baseline p=1.000, again not distinguishable from doing nothing at all; RB vs. baseline p=0.017, the only comparison that clears significance.

Across both runnable datasets, the claim that relevant retrieval drives the gain does not survive direct testing. A real, modest effect of adding memory-derived content in general does appear to be present but again relevance to the specific question does not seem to be what matters.

---

## 7. How to reproduce

```bash
cd sa-module-v2
mkdir -p data memory_bank results
cp <path-to>/selfaware_train.json <path-to>/selfaware_test.json data/

CUDA_VISIBLE_DEVICES=<free_gpu> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --port 8080 \
  --dtype bfloat16 --gpu-memory-utilization 0.25 --max-model-len 4096

# Full scale, all four conditions including scrambled
python3 src/run_selfaware.py --train-limit 300 --test-limit 500

# Original three conditions only
python3 src/run_selfaware.py --skip-scrambled

# FalseQA (limits set in __main__, default 300/300)
cp <path-to>/falseqa_train.json <path-to>/falseqa_test.json data/
python3 src/run_falseqa.py
```

---

## 8. A complementary research direction

The core loop here is fragile, the only signal for whether the model expressed uncertainty is a lexical match on surface text, and that same signal generates the lessons fed back into future prompts. The scrambling experiments above suggest this loop is being exploited, the model learns to echo hedge vocabulary rather than reason about whether to abstain.

One way to break out of that loop is to use a model-internal uncertainty signal that does not depend on what the model says out loud. Semantic entropy (Farquhar, S., Kossen, J., Kuhn, L. et al. Detecting hallucinations in large language models using semantic entropy. Nature 630, 625–630 (2024). https://doi.org/10.1038/s41586-024-07421-0) samples several generations and computes entropy over semantically clustered answers rather than surface wording - directly measuring whether the model is generating consistent answers or spreading probability across different meanings. Semantic Entropy Probes also approximate this cheaply from a single generation's hidden states, without the cost of multiple samples.

Concretely, this would let you check whether the lexical detector agrees with semantic entropy on the same responses, replace the hardcoded confidence placeholder (currently always 0.5) with a real uncertainty estimate, and eventually move from a single F1 operating point to a proper risk-coverage curve showing the tradeoff between abstention rate and answer accuracy across thresholds.

---

## 9. Questions I would want to clarify

- Has an internal ablation ever compared retrieval-relevant against retrieval-irrelevant SA content? since my own placebo could not statistically distinguish the two on either runnable dataset. Do KUQ and UnknownBench tell a different story?
- Is abstain-class F1 really the right target, or would something that penalises over-abstention symmetrically like a risk-coverage curve, or accuracy at a fixed abstention rate; kind of better capture what "more trustworthy" is meant to mean here?

---

## Regarding AI tool usage

I used Claude throughout (implementing the ablation, running the experiments, and drafting this report). Every finding was independently verified against the actual files and logs. The choice of what to check and how to interpret the results was mine.
