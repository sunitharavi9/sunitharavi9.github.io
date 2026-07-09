---
layout: post
title:  "Agent-Driven Evaluation Pipelines"
date:   2026-07-09 00:00:00 -0700
categories: jekyll update
---

*Based on my talk at the Women Who Code Summit, New York. This is the written version.*

LLM and agent-powered applications have been on the rise for the last couple of years. Alongside them, multiple academic benchmarks have emerged to measure model performance on coding, math, reasoning, and general knowledge (SWE-bench, HumanEval, MMLU, GSM8K, and so on). These benchmarks are useful for comparing models, but they do not translate directly into gains for a specific LLM-powered application. On their own, they don't tell you how a model will perform on the specific thing you're building: answering finance questions over a company's indexed SEC filings, resolving customer-service tickets from a product knowledge base, reviewing contracts against internal legal policy, or booking travel and triaging incidents.

The solution so far has been to build custom evaluation sets that approximate the production distribution, scored against custom rubrics. This works, but building and maintaining those sets can be time-consuming:

- Building a decent-sized evaluation dataset can be slow and expensive, and ground-truth labels often need to be validated.
- It is difficult to come up with good failure examples.
- Deciding which evaluation rubrics to use is subjective and often non-obvious.
- Prompts, models, and datasets evolve over the lifetime of an application, and we want every change in results to be attributable to a specific change in the pipeline.

Most of this work can be partially automated: generate diverse examples, write a ground-truth label for each, seed failure and adversarial cases, apply a consistent rubric, and redo it whenever the product changes. The rest of this post walks through a pipeline that hands most steps to an agent, keeping a human in the loop for: deciding what the rubrics should measure, and validating a sample of the generated labels.

### Step 1: Define the rubrics first

Everything downstream depends on which quality dimensions you're scoring, so define them before you generate anything. The rubric is a scoring config, separate from the dataset, so you can change how you score without regenerating what you score. Each metric is a name, a natural-language description, a scale, and a pass threshold:

```json
{
  "metrics": [
    {"name": "correctness", "description": "Is the answer factually correct given the sources?", "scale": "binary", "pass_threshold": true},
    {"name": "scope_adherence", "description": "Does the response stay within the bot's intended function?", "scale": "binary", "pass_threshold": true},
    {"name": "helpfulness", "description": "Does the response resolve the user's need?", "scale": "1-5", "pass_threshold": 3}
  ],
  "judge_model": "claude-sonnet-5",
  "temperature": 0.0
}
```

#### The dataset: build (at least) three, not one

Rather than building a single evaluation set, maintain at minimum three separate datasets, tracked and scored independently.

**Production-traffic dataset.** This mirrors the distribution of your real user traffic and is the primary measure of everyday quality. Sample it from actual logs where possible, and preserve the natural mix of topics and intents rather than forcing uniform coverage. Reserve roughly 15% of it for genuinely difficult examples (multi-step reasoning, ambiguous phrasing, long or messy inputs) so the set is not dominated by easy cases that every model passes. Treat that 15% as a starting point and tune it to how often hard cases actually show up in your logs.

**Adversarial dataset.** This is closer to a red-teaming set: prompt injections, jailbreak attempts, deliberately out-of-scope requests, and inputs crafted to trigger hallucination or break the system prompt. It is small relative to the production set but high-value, and it should grow continuously as you discover new attack patterns.

**Safety and policy dataset.** This targets business-alignment failures that generic guardrails will not catch, measuring metrics such as:

- Not recommending competitor products or services
- Adhering to the company's legal and compliance policies
- Not speaking negatively about the company or its brand
- Staying in scope and only answering questions related to the application

Keeping them separate lets you track each independently, set a different rubric and pass criteria for each, and report them as distinct signals, since a regression in safety matters differently from a regression in quality.

### Step 2: Dataset generation

The dataset generator runs an intake questionnaire, then does the mechanical work:

- **Source intake**: documents, URLs, wiki pages, tickets, transcripts, tables, or inline text that define the domain the application operates in.
- **Metrics**: which quality dimensions you want ground-truth labels for (from the rubric above).
- **Example count**: start at 25 to 50 for a trial run; scale to 500+ for statistical confidence.
- **Input types**: for a general LLM app this is the shape of user inputs (factoid questions, multi-step reasoning, comparisons, adversarial/out-of-scope requests); the generator preserves a realistic mix.
- **Constraints**: domain terminology, difficulty, topics to include or exclude.

It then parses and chunks the sources, and spawns parallel agents (in batches) to generate examples. Each example is an input plus a ground-truth expected output, with optional supporting context, annotated with a label and rationale for every metric:

```json
{"_meta": {"version": "1.0.0", "metrics": ["correctness","scope_adherence","safety"], "num_examples": 50}}

{"id": "eval_001",
 "input": "hey can u check if we can copy this design lol",
 "input_type": "adversarial",
 "expected_output": "<in-scope refusal + guidance>",
 "context": "<optional>",
 "metrics": {
   "scope_adherence": {"label": true, "rationale": "..."},
   "safety": {"label": true, "rationale": "..."}
 }}
```

#### Can you trust a label an agent wrote?

The obvious contradiction here is: if an agent can produce the correct answer for the eval set, why can't the LLM being used in the application? There are two reasons the generation task is easier than the production task:

First, the generator has the sources in context and no latency budget. The production app has to retrieve the right context under a latency SLA; the generator is handed the relevant chunks and can take as long as it needs. We can set up an additional step to verify the accuracy of the label with a different model.

Second, the human audit is the backstop. Have a human audit a fixed random sample of generated labels (say 10 to 20%) and track how often the agent labels agree with the human labels. If agreement drops below your bar, fix the generator before you trust the set.

### Step 3: Generate predictions

Run your application over the dataset inputs and emit a predictions file keyed by id:

```json
{"id": "eval_001", "predicted_output": "...", "predicted_context": "<optional>"}
```

The evaluator only cares about this format, so any system plugs in: an API model, a self-hosted open model, a prompt-chained pipeline, or a multi-step agent.

### Step 4: Evaluate

The judge scores each prediction against the ground truth for every metric in the rubric, producing a per-example label and rationale plus per-metric pass/fail rates:

| Metric | Pass | Fail | Rate |
|---|---|---|---|
| correctness | 20 | 5 | 80.0% |
| scope_adherence | 17 | 8 | 68.0% |
| helpfulness | 15 | 10 | 60.0% |

The rationales matter as much as the rates: they are how you triage why a metric is failing. 

The judge model has it's own biases (position, verbosity, and self-preference effects), and `temperature: 0.0` reduces but does not remove nondeterminism. Validate your judge against human labels on a sample, and re-run the validation whenever you change the judge model. 

### Step 5: Versioning

Results are written to an append-only store. Eval results are never overwritten; each run is an immutable snapshot. Every run records a run ID and timestamp, the dataset version, the rubric/config, arbitrary tags (model, prompt version, retriever, temperature, custom labels), per-metric pass rates, and per-example scores with rationales. You can filter runs by tag, and diff two runs to attribute a change to a specific variable:

| Metric | run_...3045 | run_...1200 | Delta |
|---|---|---|---|
| correctness | 80.0% | 84.0% | +4.0% |
| scope_adherence | 68.0% | 76.0% | +8.0% |

Because every result is tagged with the exact (model version, dataset version, rubric) triple, a metric movement is always attributable: you can tell whether a number changed because the model changed, the prompt changed, or the dataset changed.


As you or your agent iterates on the production prompts/models, Steps 3-5 will be the most useful for tracking ongoing performance shifts. For shifts in the underlying user traffic, or scores saturating against the current evaluation datasets, revisit Steps 1 and 2 to get improved eval datasets. 