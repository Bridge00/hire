# HIRE: Explanation-Based Code Evaluation Research

This repository studies whether LLM-generated explanations can make code-correctness evaluation more accurate, especially when the evaluator does **not** directly execute the implementation.

The central research tension is:

- Query-aware explanations often achieve strong F1, but can be overly permissive and approve incorrect code.
- Conservative methods reduce false positives, but frequently reject correct code.
- More explanation detail is useful only when it exposes a decision that changes observable behavior; generic detail often makes evaluation worse.

The project is experimental. Most results are cached under `.cache/`; preserve existing caches and create a new method folder for every new variant.

## Repository map

| Location | Purpose |
|---|---|
| `run_eval_normal.py` | Main local evaluation entry point. |
| `src/eval_class.py` | Evaluation pipelines, routing, cache handling, dialogue flows, and research variants. |
| `utils/prompts.py` | Prompts for explainers, judges, specification extraction, appeals, and refinement. |
| `data/` | Benchmark records and source-code variants. |
| `calculate_metrics.py` | Metrics from cached evaluation outputs. |
| `execution_logs/` | Local experiment logs. |
| `.cache/` | Cached model calls and final evaluations. Do not overwrite a completed method without explicit intent. |

## Data and labels

The usual evaluation setup runs both source types:

- `canonical_solution`: expected correct.
- `incorrect_solution`: expected incorrect.

The standard runner excludes rows with either of these fields:

- `incorrect_solution_old`
- `wrong_canonical`

Metrics use:

- **TPR**: canonical solutions approved.
- **FPR**: incorrect solutions approved. Lower is particularly important for review settings.
- **FNR**: canonical solutions rejected.
- **F1** and **accuracy** as aggregate summaries.

## Core methods

### Query-aware L3

The explanation model sees the problem statement while explaining code. This usually gives the best general F1, but can create a correctness bias: the explanation frames buggy code as if it fulfills the task.

Primary cache family:

```text
hire_explainer_obj_alignment_checker_query_aware_lambda_L{level}
```

Available multi-level experiments commonly use `L1`, `L3`, `L5`, `L8`, and `L10`.

### Non-query-aware explanation + style transfer

The raw explainer describes only the code. A separate style-transfer stage then maps that description into task terminology before the final alignment judge sees it.

```text
hire_explainer_obj_alignment_checker_style_transfer_lambda_L3
```

This is a key baseline: it tends to reduce FPR relative to query-aware L3 while retaining relatively strong F1.

### Sequential style-transfer dialogue

The judge can ask the code-aware explainer for clarification, up to three turns.

```text
dialogue_hire_explainer_obj_style_transfer
```

This is more conservative than one-shot style transfer. It generally lowers FPR but raises FNR. In practice, the judge often decides at turn 0, so it should not be described as deeply deliberative unless the trace shows follow-up turns.

Run it with:

```powershell
python run_eval_normal.py `
  --dataset humaneval_py `
  --eval_source canonical_solution incorrect_solution `
  --eval_model gpt-4o-mini `
  --explainer_model gpt-4o-mini `
  --eval_prompt hire_explainer_obj_style_transfer `
  --mode dialogue `
  --lambda_val 3 `
  --num_workers 15
```

### Behavior comparison

`behavior_comparison_no_rc` is an independent code-evaluation baseline. It is highly conservative: low FPR, but high FNR. It is useful when false positives are much more costly than false negatives.

### Majority voting across lambda levels

For query-aware alignment checker levels `L1`, `L3`, `L5`, `L8`, and `L10`, a strict 3-of-5 approval vote improved F1 over individual levels on HE-PY, HE-CPP, and HE-Go while modestly lowering FPR relative to L3.

Do not mix cache families in an ensemble. All votes must use the same prompt family, source type, model, and filtered task set.

## Research variants tried

### Requirements/specifications mapping

The project separates preconditions from functional requirements:

```text
Problem -> structured specification
       -> preconditions, functional obligations, output requirements, ambiguities
```

Important rule: preconditions define valid evaluation inputs; they are not requirements that code must validate.

Variants include `--requirements_only_judges`, `--specifications_only_judges`, specification-aware dialogue, one-by-one verification, and requirements appeals.

These methods reduce some invented-invalid-input objections, but do not fully solve semantic interpretation failures. Appeal models can also under-specify a genuine requirement and incorrectly dismiss a real defect.

### Reconstruction / compare / update (RCU)

The intended loop is:

```text
code -> explanation -> reconstructed code -> behavioral comparison
     -> patch only missing implementation facts -> repeat
```

Several RCU variants were tested, including specification-blind versions and fact-coverage variants. They did not outperform the one-shot style-transfer baseline. Common failures were comparator hallucinations, syntactic rather than semantic discrepancies, malformed patches, and added detail giving the final judge more ways to misread a correct explanation.

### Code-only refine before/after style transfer

The tested pipeline was:

```text
raw explanation -> code-only self-refine -> style transfer
                -> code-only post-style audit -> final judge
```

It can repair a genuine style-transfer corruption. Example: a correct explanation of a function over **words** was changed to **letters** by style transfer; a post-style code audit restored the correct unit and recovered the verdict.

However, full HE-PY evaluation was harmful: it produced many new false positives and false negatives. A code-faithful rewrite can be a worse task explanation, because it replaces semantic descriptions with low-level mechanics.

### Dynamic detail appendix

This is a narrower alternative to rewriting explanations. It appends only code-grounded, behavior-changing facts, such as:

- strict versus non-strict comparisons;
- loop/range/slice endpoints;
- tie-breaking on equal values;
- negated predicates;
- early returns;
- output construction/slicing/formatting;
- rounding or epsilon adjustments.

The initial targeted trial recovered only 3 of 21 HE-PY style-transfer false positives. The likely improvement is to make details **contrastive** (e.g., explicitly state what happens at equality) rather than merely list operators.

### Requirement interrogation (current promising direction)

The intended design deliberately separates task semantics from code behavior:

```text
problem -> requirements + valid-input domain
requirement -> neutral behavior probe
code + probe only -> what does the implementation do?
requirement + behavior answer -> satisfied / violated / uncertain
```

The code-behavior model should not see the expected outcome or full problem statement. The verifier should reject only on an explicit violation over the valid-input domain. A prototype verifier exists; end-to-end experiment wiring is still incomplete and should be finished before relying on this method.

## Key empirical findings

On the standard filtered sets, representative results were:

| Dataset | Method | F1 | FPR |
|---|---|---:|---:|
| HE-PY | Query-aware L3 | 79.7 | 25.6 |
| HE-PY | One-shot style transfer L3 | 78.3 | 16.8 |
| HE-PY | Sequential style-transfer dialogue | 69.1 | 13.6 |
| HE-PY | `behavior_comparison_no_rc` | 67.3 | 7.2 |
| HE-CPP | Query-aware L3 | 71.3 | 26.6 |
| HE-CPP | Sequential style-transfer dialogue | 51.3 | 12.1 |
| HE-Go | Query-aware L3 | 70.3 | 45.1 |
| HE-Go | Sequential style-transfer dialogue | 59.2 | 26.2 |

Interpretation:

- Query-aware explanations optimize F1 but can approve incorrect code too often.
- Dialogue and behavior comparison are useful conservative controls, not universal replacements.
- Adding generic detail does not solve false positives.
- The critical missing information is often a specific behavioral boundary, not a fuller narrative.

## Recurring failure modes

1. **Requirement hallucination**: evaluator invents validation, edge-case, efficiency, or implementation obligations.
2. **Requirement under-specification**: evaluator/appeal model weakens explicit semantics such as inclusivity, ordering, ties, or examples.
3. **Semantic-equivalence failure**: equivalent descriptions are treated as contradictory, e.g. “all digits odd” versus “no even digits.”
4. **Correct mechanics, poor task framing**: a code-faithful explanation may be less task-compatible after refinement.
5. **Persuasive explanation of wrong code**: a coherent summary can conceal the one branch, comparator, tie, or slice that causes a defect.
6. **Excess detail**: creates irrelevant objections and more opportunities for the final judge to fail correct code.

## Guidance for future agents

1. Start from the one-shot style-transfer L3 baseline unless testing a specific hypothesis.
2. Always compare FPR and FNR, not only F1.
3. Preserve dirty worktree changes and never overwrite an existing experimental method folder.
4. Use a distinct cache/prompt suffix for each new ablation.
5. Apply the standard dataset exclusions before reporting metrics.
6. When inspecting an error, distinguish:
   - code comprehension failure;
   - explanation corruption;
   - requirement interpretation failure;
   - final-judge inconsistency.
7. Prefer selective, observable, contrastive implementation facts over generic explanation expansion.
8. Do not treat out-of-domain inputs as defects unless the specification explicitly requires handling them.

