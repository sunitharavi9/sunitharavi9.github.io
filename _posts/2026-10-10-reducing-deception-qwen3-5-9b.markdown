---
layout: post
title:  "Reducing Deception in AI Models"
date:   2026-10-10 00:00:00 -0700
categories: jekyll update
---

<script>
  window.MathJax = { tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Summary

We compare three interventions for reducing deception in [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B): a developer system prompt (inference only), LoRRA activation steering (a merged LoRA adapter), and supervised fine-tuning (SFT) on an honesty and ethics data mix. Deception is measured on MASK and MACHIAVELLI and capability on MMLU, GSM8K and MT-Bench.

Code: [github.com/sunitharavi9/deception-reduction](https://github.com/sunitharavi9/deception-reduction)

Main findings, all relative to the unprompted base model:

1. **A system prompt with a code of conduct** raises MASK macro honesty from 0.593 to 0.811, but evasion also rises from 8.4% to 14.2%. It lowers MACHIAVELLI violations by 6.2 points and deception by 11.4, and lowers game score by 15.8.
2. **LoRRA** raises MASK macro honesty to 0.654 with no increase in evasion (7.4%). It lowers MACHIAVELLI violations by 12.7 points and deception by 15.6, more than either prompt, with no capability change. Its MASK gain comes almost entirely from one of six archetypes (provided facts).
3. **SFT on general instruction data alone inflates MASK honesty through evasion.** A Tülu 3-only control reaches 0.806 macro honesty while evading on 39.0% of items.
4. **Our honesty and ethics data replaces evasion with direct answers.** Relative to the control, the full mix holds macro honesty level (0.814), cuts evasion to 14.0%, and raises normalized honesty from 0.691 to 0.731. It gives the lowest MACHIAVELLI deception of any condition (88.3). An ablation shows that moral-choice and truthful-dialogue data remove the evasion, and synthetic pressure scenarios with honest-vs-evasive pairs keep the resulting answers honest.
5. **The SFT recipe, not the added data, costs capability.** All three SFT variants, including the control, fall from 0.938 to 0.587–0.679 on GSM8K and from 8.15 to 6.96–7.08 on MT-Bench. MMLU moves by at most 0.007.

## Experimental setup

**Model.** [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

**Deception benchmarks.**

- **[MASK](https://arxiv.org/abs/2503.03750)** (Ren et al., 2025) separates honesty from accuracy. For each proposition, the model's belief is first elicited with neutral prompts; the model is then given a pressure prompt (for example, a sales persona or a supervisor requesting a cover-up). Each pressured response is labelled honest, lie (contradicts the elicited belief) or evasion. Items fall into six archetypes: continuations, disinformation, doubling down, known facts, provided facts and statistics.
- **[MACHIAVELLI](https://arxiv.org/abs/2304.03279)** (Pan et al., 2023) runs the model as an agent in text-based games and scores its actions for deception, ethical violations (Σ), power-seeking (Σ) and disutility, each normalised to a random agent (100 = random; lower is better). It also reports in-game reward (game score).

**MASK metrics used:**

| Metric | Definition |
| --- | --- |
| Honesty, macro | 1 − lie rate, averaged over the six archetypes. Evasions count as non-lies. |
| Honesty, normalized | Honesty computed only over items on which the model holds a belief. |
| Lie rate given belief | Share of lies among items on which the model holds a belief. |
| Lie rate (overall) | Share of all 999 items labelled lie. Used in the per-archetype analysis and the chart. |
| Evasion rate | Share of all 999 items labelled evasion. Not directional: a rise can raise honesty without the model being more truthful. |

**Capability benchmarks.** [MMLU](https://arxiv.org/abs/2009.03300) (Hendrycks et al., 2021), 0-shot, all 14,042 test questions. [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021), 0-shot, all 1,319 test questions. [MT-Bench](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023), 80 questions × 2 turns, LLM-judged.

**Judge.** MASK and MT-Bench are scored by [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506).

**Decoding and runs.** All evaluation generations use temperature 0.7.

## Interventions

### 1. Developer system prompts

The developer prompt is placed first in the system message, followed by the benchmark's own system prompt. MASK belief elicitation is run without it, so the reference belief is the model's unprompted belief.

- **Honest:** the prompt from the MASK paper: "You are a helpful assistant. Please respond to the user's prompt in a way that is honest and truthful."
- **Honest + ethics:** the honest prompt plus a six-point code of conduct the model is instructed to follow even when a role or persona instructs otherwise: (1) honesty, (2) no deceiving others, (3) care for people, (4) fairness and rules, (5) openness when declining, (6) fiction is fine.

### 2. LoRRA

We replicate Low-Rank Representation Adaptation (LoRRA) from [representation engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023), following its use in the MASK paper. A [LoRA](https://arxiv.org/abs/2106.09685) adapter (Hu et al., 2021) is trained so that the model's activations on a plain input match its own activations shifted along an honest-minus-dishonest persona direction. No deception-labelled data is used.

**Contrast data.** 5,000 instructions from [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned), a cleaned release of [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al., 2023), restricted to instructions with no input field and a reference response of at least 12 words. Each instruction is rendered three times with Qwen's chat template, with the reference response truncated to 64 tokens and identical across renderings:

- x⁺: system prompt "Pretend you're an honest person making statements about the world." + instruction + response
- x⁻: system prompt "Pretend you're a dishonest person making statements about the world." + instruction + response
- x: instruction + response, no system prompt

Because the response tokens are identical, activation differences at response positions are attributable to the persona.

**Objective.** Let \\(h_\ell(\cdot) \in \mathbb{R}^{\lvert R \rvert \times d}\\) be the hidden states at layer ℓ over response tokens R, with hidden size d. With the adapter disabled, the target at each target layer ℓ ∈ T is

$$
r_\ell = h_\ell(x) + \alpha \left[ h_\ell(x^+) - h_\ell(x^-) \right], \qquad \alpha = 5.
$$

With the adapter enabled and only x as input, the loss is the mean squared error to the target, averaged over valid response tokens, hidden dimensions and target layers:

$$
\mathcal{L} = \frac{1}{\lvert T \rvert} \sum_{\ell \in T} \frac{1}{\lvert R \rvert\, d} \sum_{t \in R} \left\lVert h^{\mathrm{LoRA}}_{\ell,t}(x) - r_{\ell,t} \right\rVert_2^2 .
$$

T consists of the layers at 50% and 75% of depth. Adapters (rank 16, scaling α = 32, dropout 0.05) are applied to layers from 15% to 50% of depth, so the shift is produced upstream of the target layers and the layers after 50% depth are unmodified. The adapter is merged and evaluated without a developer prompt.

### 3. Supervised fine-tuning

**Data mix** (after decontamination and deduplication; 2% held out for validation):

| Source | SFT | DPO pairs | Target behaviour |
| --- | --- | --- | --- |
| [Moral Stories](https://arxiv.org/abs/2012.15738), MACHIAVELLI-style format | 3,000 | 3,000 | Ethical action choice in game-style prompts (moral > immoral) |
| [Moral Stories](https://arxiv.org/abs/2012.15738), open question | 3,000 | 3,000 | Ethical advice in prose (moral > immoral) |
| [DolusChat](https://arxiv.org/abs/2505.13787) | 6,000 | 6,000 | Truthful reply when context gives a reason to deceive (truthful > deceptive) |
| Synthetic pressure scenarios | 2,283 | 2,044 + 2,197 | Honesty under MASK-style pressure (honest > lie; honest > evasive) |
| [Tülu 3](https://arxiv.org/abs/2411.15124) SFT mixture | 23,955 | – | Capability preservation |
| **Total** | **38,238** | **16,241** | |

- **Moral Stories** (Emelin et al., 2021): each item gives a norm, a situation, an intention, and a moral and an immoral action, both of which achieve the intention. Training on the moral action therefore targets the ethical route to the goal rather than abandoning the goal.
- **DolusChat** (Cundy & Gleave, 2025): conversations in which the assistant has context it may be tempted to hide or distort. The system prompt and context form the prompt; the truthful reply is the SFT target; (truthful, deceptive) is the DPO pair.
- **Tülu 3** (Lambert et al., 2024): general instruction-following conversations under 12k characters, SFT only.

**Synthetic pressure scenarios.** Cundy & Gleave (2025) show that preference learning against lies can induce evasion instead of honesty. The synthetic data supplies an explicit honest > evasive signal:

1. *Scenario generation.* A teacher model writes scenarios in each of MASK's six archetypes. Each combines one of 49 domains (e.g. used-car sales, airline delays, geology) with one of 12 pressure sources (e.g. a sales quota, a supervisor's instruction, avoiding a lawsuit), with invented names and no real controversies.
2. *Reply generation.* For each scenario the teacher writes an honest-and-direct reply (stays in role, helps with the legitimate part of the task, states the true fact plainly), a lie and an evasion.
3. *Filtering.* A judge labels all three replies; a scenario is kept only if its honest reply passes. Each kept scenario yields one SFT example and up to two DPO pairs sharing the same chosen reply.

| Archetype | Scenarios | vs. lie | vs. evasive |
| --- | --- | --- | --- |
| Continuations | 435 | 350 | 399 |
| Disinformation | 195 | 174 | 176 |
| Doubling down | 553 | 530 | 526 |
| Known facts | 292 | 241 | 289 |
| Provided facts | 323 | 279 | 322 |
| Statistics | 485 | 470 | 485 |
| **Total** | **2,283** | **2,044** | **2,197** |

**Decontamination.** Every SFT and DPO example is checked against every MASK text field (system and user prompts, doubling-down turns, propositions, ground truths, belief-elicitation prompts), all MT-Bench turns, and the GSM8K and MMLU test questions. An example is dropped if any of its text shares an 8-word n-gram with an evaluation prompt after lowercasing and keeping only alphanumeric tokens, or if it contains a benchmark canary string. Synthetic scenarios are additionally compared with every MASK proposition and ground truth by content-word Jaccard overlap (stopwords removed) and dropped above 0.34. MACHIAVELLI is not indexed because no training source is drawn from game text.

**Training.** Both stages use LoRA (r = 32, α = 64, dropout 0.05), skip examples longer than 2,048 tokens, compute loss on the final assistant reply only, and use AdamW with 5% linear warm-up and cosine decay. Stage 1 (SFT): 37,446 training examples (792 validation), one epoch, learning rate 1e-4. Stage 2 ([DPO](https://arxiv.org/abs/2305.18290), Rafailov et al., 2023): 15,877 training pairs (364 validation), one epoch, learning rate 2e-5, β = 0.1, plus a length-normalised NLL term on the chosen reply to prevent the likelihood of both replies from falling. **Only Stage 1 is evaluated in this post.**

**Controls.** *SFT (Tülu)* trains on the same Tülu 3 subset as the full mix, with the same seed and hyperparameters, and no honesty or ethics data. *SFT (no syn.)* is the full mix without the synthetic scenarios. *SFT (full)* is the complete mix.

## Results

**Table 1.** All conditions, test splits. ↑/↓ give the desirable direction; evasion is not directional. MACHIAVELLI scores are relative to a random agent (100).

| Metric | Base | Honest | Honest + ethics | LoRRA | SFT (Tülu) | SFT (no syn.) | SFT (full) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MASK honesty, macro ↑ | 0.593 | 0.646 | 0.811 | 0.654 | 0.806 | 0.753 | 0.814 |
| MASK honesty, normalized ↑ | 0.488 | 0.545 | 0.737 | 0.558 | 0.691 | 0.642 | 0.731 |
| MASK lie rate given belief ↓ | 0.544 | 0.492 | 0.254 | 0.411 | 0.287 | 0.313 | 0.259 |
| MASK evasion rate | 0.084 | 0.101 | 0.142 | 0.074 | 0.390 | 0.165 | 0.140 |
| MACHIAVELLI deception ↓ | 110.96 | 108.83 | 99.52 | 95.38 | 101.50 | 98.39 | 88.30 |
| MACHIAVELLI violations Σ ↓ | 104.54 | 101.62 | 98.38 | 91.89 | 98.21 | 86.16 | 86.57 |
| MACHIAVELLI power Σ ↓ | 104.28 | 103.70 | 101.30 | 98.53 | 101.84 | 96.17 | 95.97 |
| MACHIAVELLI game score ↑ | 144.34 | 145.72 | 128.53 | 121.88 | 122.10 | 116.21 | 113.61 |
| MMLU ↑ | 0.783 | 0.783 | 0.781 | 0.785 | 0.777 | 0.776 | 0.778 |
| GSM8K ↑ | 0.938 | 0.936 | 0.939 | 0.936 | 0.587 | 0.679 | 0.619 |
| MT-Bench ↑ | 8.146 | 8.206 | 8.225 | 8.206 | 7.076 | 6.962 | 6.994 |

**Table 2.** MASK overall lie rate and evasion rate per archetype (share of items in the archetype). B = base, H = honest prompt, E = honest + ethics, L = LoRRA, C = SFT (Tülu) control, S = SFT (full). The no-synthetic ablation was not broken down per archetype.

| Archetype | n | Lie B | Lie H | Lie E | Lie L | Lie C | Lie S | Ev. B | Ev. H | Ev. E | Ev. L | Ev. C | Ev. S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continuations | 176 | .489 | .403 | .091 | .483 | .267 | .233 | .102 | .068 | .045 | .097 | .267 | .222 |
| Disinformation | 125 | .256 | .176 | .032 | .280 | .072 | .056 | .048 | .048 | .032 | .032 | .416 | .376 |
| Doubling down | 120 | .325 | .242 | .233 | .350 | .300 | .125 | .075 | .058 | .125 | .042 | .125 | .075 |
| Known facts | 209 | .282 | .263 | .191 | .258 | .072 | .167 | .048 | .062 | .144 | .048 | .493 | .014 |
| Provided facts | 274 | .657 | .642 | .325 | .285 | .318 | .230 | .095 | .150 | .245 | .117 | .391 | .047 |
| Statistics | 95 | .432 | .400 | .263 | .421 | .137 | .305 | .158 | .232 | .189 | .063 | .695 | .305 |
| Overall | 999 | .437 | .391 | .202 | .334 | .207 | .190 | .084 | .101 | .142 | .074 | .390 | .140 |

### System prompts

The honest prompt produces small changes: macro honesty 0.593 → 0.646, MACHIAVELLI violations −2.9 and deception −2.1. On provided facts its lie rate change (0.657 → 0.642) is within one standard error.

The honest + ethics prompt lowers the lie rate in all six archetypes, most on continuations (0.489 → 0.091) and disinformation (0.256 → 0.032). The overall lie rate falls by 23.5 points (0.437 → 0.202) while evasion rises by 5.8 points (0.084 → 0.142), so at most about a quarter of the lie reduction can be attributed to lies converted into evasions. The shift toward evasion is concentrated on provided facts (0.095 → 0.245) and known facts (0.048 → 0.144).

On MACHIAVELLI it lowers deception by 11.4 points and violations by 6.2, and lowers game score by 15.8. Category scores from the full report show the change is not uniform: deception (−11.4), manipulation (−9.4) and promise-breaking (−17.6) decrease, while stealing (+10.3) and non-physical harm (+5.2) increase. Neither prompt changes MMLU, GSM8K or MT-Bench by more than 0.003, 0.003 and 0.08 respectively.

### LoRRA

LoRRA lowers the overall MASK lie rate by 10.3 points (0.437 → 0.334) with evasion slightly lower (0.084 → 0.074), so its MASK gain is not explained by evasion. The gain is localised: provided facts falls by 37.2 points (0.657 → 0.285), which over 274 of 999 items accounts for about 10.2 of the 10.3-point overall reduction. The other five archetypes change by 0.6 to 2.5 points in either direction, within one standard error.

On MACHIAVELLI, LoRRA lowers violations by 12.7 points, deception by 15.6 and power-seeking by 5.8, more than either prompt on all three. The full report notes increases in some categories, including promise-breaking and spying. Game score falls by 22.5 points (144.34 → 121.88). Capability changes are within ±0.002 on MMLU and GSM8K and +0.06 on MT-Bench.

### Fine-tuning

**Control.** SFT on the Tülu 3 subset alone raises macro honesty from 0.593 to 0.806, close to the honest + ethics prompt. The evasion rate rises from 8.4% to 39.0% (statistics 69.5%, known facts 49.3%, disinformation 41.6%). Because MASK scores evasions as non-lies, most of this honesty gain reflects the model declining to answer rather than answering truthfully.

![Share of MASK items labelled lie, evasion, or neither, for the base model and the three SFT variants]({{ "/assets/images/mask-outcomes-sft.png" | relative_url }})

*Overall MASK lie and evasion rates, test split (n = 999).*

**Full mix vs. control.** Adding the honesty and ethics data lowers evasion from 0.390 to 0.140 with macro honesty unchanged within error (0.806 → 0.814). Normalized honesty rises from 0.691 to 0.731 and lie rate given belief falls from 0.287 to 0.259. Per archetype (Table 2), the full mix lowers lie rate on four archetypes but raises it on known facts (0.072 → 0.167) and statistics (0.137 → 0.305), the two archetypes where the control evaded most. On those, some evasions became lies rather than honest answers.

On MACHIAVELLI, relative to the control, the full mix lowers violations by a further 11.6 points and deception by 13.2, compared with 6.3 and 9.5 points from the control relative to base. Its deception score (88.3) is the lowest of all conditions; its violations score (86.57) is second to the no-synthetic ablation (86.16), a difference of 0.4 points. Game score falls to 113.61, the lowest of all conditions.

**Ablation.** Removing the synthetic scenarios separates the contributions of the two data components:

- *Moral Stories + DolusChat* (control → no syn.): evasion 39.0% → 16.5%; overall lie rate 20.7% → 23.1%; macro honesty 0.806 → 0.753; normalized honesty 0.691 → 0.642; MACHIAVELLI violations 98.2 → 86.2; deception 101.5 → 98.4.
- *Synthetic scenarios* (no syn. → full): evasion 16.5% → 14.0%; overall lie rate 23.1% → 19.0%; macro honesty 0.753 → 0.814; normalized honesty 0.642 → 0.731; MACHIAVELLI violations 86.2 → 86.6; deception 98.4 → 88.3.

Moral Stories and DolusChat account for the reduction in evasion and the full reduction in MACHIAVELLI violations, but the direct answers they produce include more lies. The synthetic scenarios reduce lies on MASK and MACHIAVELLI deception without increasing evasion. The full report states that the synthetic data's largest per-archetype lie-rate reductions (about 15 points each) are on continuations, statistics and doubling down, the three most heavily weighted archetypes in its generation (see the archetype table under Interventions). These per-archetype ablation numbers are not tabulated here.

### Capabilities

All three SFT variants fall well below the base model on GSM8K (0.938 → 0.587, 0.679, 0.619) and MT-Bench (8.146 → 7.076, 6.962, 6.994). MMLU changes by at most 0.007. Because the Tülu-only control shows the same drop, it is attributable to the fine-tuning recipe and the Tülu subset rather than to the honesty and ethics data. GSM8K varies by 9.2 points across the three runs although none of the added data targets mathematics, which is consistent with a formatting or answer-extraction effect; this has not been verified. Prompting and LoRRA leave all three capability benchmarks unchanged.

## Limitations

- **Unvalidated judge.** MASK and MT-Bench labels come from Mistral-Small-3.2-24B-Instruct-2506, not the judges used in the original benchmarks, and have not been checked against human annotation.
- **Prompts not optimised.** Each developer prompt was written once. Results for prompting are a lower bound; a code of conduct that explicitly discourages evasion may change both honesty and evasion.
- **LLM-generated training data.** The synthetic scenarios, replies and labels are model-generated. A sample was spot-checked by hand, but no systematic quality evaluation was run; mislabelled evasions, unrealistic scenarios or incorrect facts may remain.

## Future work

- **Source of pressured lying.** If lying is driven by role-play, stating the same incentive without a persona (a bare instruction or an explicit reward) should reduce it. If it is driven by post-training, the effect should differ across pretrained, instruction-tuned and preference-tuned checkpoints of the same model on identical pressure prompts.
- **Representational vs. surface change.** Compare activations with and without LoRRA. A shift that is constant across prompts indicates a fixed steering vector; a shift that is larger on pressure prompts indicates a change in how pressure is processed.
- **Multi-vector steering.** LoRRA's gain concentrates in one archetype and it worsens some MACHIAVELLI categories, consistent with a single contrast capturing one facet of honesty. Separate contrasts for reporting one's own belief, disclosing information given in context and not fabricating, plus ethics facets such as harm, fairness and promise-keeping, could be steered individually or jointly and evaluated per archetype and per MACHIAVELLI category.
- **Capability preservation.** Use stronger preservation data or a recipe closer to the model's original post-training, after ruling out answer-extraction errors on GSM8K.
- **Stage 2.** Evaluate the DPO stage, in particular whether the honest > evasive pairs further reduce evasion without raising lie rate on known facts and statistics.

## References

- Cobbe, K., et al. Training verifiers to solve math word problems. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168), 2021.
- Cundy, C. and Gleave, A. Preference learning with lie detectors can induce honesty or evasion. NeurIPS 2025. [arXiv:2505.13787](https://arxiv.org/abs/2505.13787).
- Emelin, D., Le Bras, R., Hwang, J. D., Forbes, M., and Choi, Y. Moral Stories: Situated reasoning about norms, intents, actions, and their consequences. EMNLP 2021. [arXiv:2012.15738](https://arxiv.org/abs/2012.15738).
- Hendrycks, D., et al. Measuring massive multitask language understanding. ICLR 2021. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300).
- Hu, E. J., et al. LoRA: Low-rank adaptation of large language models. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Lambert, N., et al. Tülu 3: Pushing frontiers in open language model post-training. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124), 2024.
- Pan, A., et al. Do the rewards justify the means? Measuring trade-offs between rewards and ethical behavior in the MACHIAVELLI benchmark. ICML 2023. [arXiv:2304.03279](https://arxiv.org/abs/2304.03279).
- Rafailov, R., et al. Direct preference optimization: Your language model is secretly a reward model. NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
- Ren, R., et al. The MASK benchmark: Disentangling honesty from accuracy in AI systems. [arXiv:2503.03750](https://arxiv.org/abs/2503.03750), 2025.
- Taori, R., et al. Stanford Alpaca: An instruction-following LLaMA model. [GitHub](https://github.com/tatsu-lab/stanford_alpaca), 2023. Cleaned release: [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned).
- Zheng, L., et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
- Zou, A., et al. Representation engineering: A top-down approach to AI transparency. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405), 2023.
