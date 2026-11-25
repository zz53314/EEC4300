# Boolean Experiments: Algorithm Improvements and Results

This directory implements the Boolean expression-discovery task in the HypoSpace project. The experiments use the DeepSeek instruction-following model `deepseek-chat`: given partial Boolean input–output observations, the model proposes expressions that are consistent with the observations. The candidates are then evaluated against the hypothesis space constraints and structural-equivalence rules.

## Experimental Setup

- **Model:** `deepseek-chat`, called through an OpenAI-compatible API with temperature 0.7.
- **Data:** [`datasets/boolean_2var.json`](datasets/boolean_2var.json), containing 35 observation sets. Variables: `x` and `y`. Operators: `AND` and `OR`. Maximum expression depth: 2.
- **Evaluation scale:** Three random seeds: 33550336, 33550337, and 33550338. Each seed samples 30 observation sets and makes 10 model queries per set. Every method uses the corresponding seed and the same dataset.
- **Structural deduplication:** Expressions are deduplicated using the mechanistic keys in the dataset, with commutativity, AND/OR idempotence, and associativity flattening enabled. Distribution, absorption, and De Morgan normalization are not applied.

## Metrics

- **Validity:** Number of candidates that satisfy all hypothesis-space constraints and match every visible observation, divided by the number of queries.
- **Uniqueness / Novelty:** Number of unique expressions within the hypothesis space, deduplicated by mechanistic structure, divided by the number of queries. This measures structural exploration rather than the number of unique valid expressions alone.
- **Recovery:** Number of ground-truth mechanistic structures recovered by the valid unique candidates, divided by the total number of ground truths for the observation set.

## Experimental Procedure

1. **Generate the dataset.** Enumerate expressions allowed by the variables, operators, depth limit, and mechanistic-deduplication rules. Construct combinations of partial observations and enumerate every ground-truth expression consistent with each observation set.
2. **Run the baseline.** For the same sampled observation sets, prompt the original DeepSeek model to generate expressions repeatedly without per-candidate verification feedback. Parse responses and calculate the three metrics.
3. **Improve the algorithm incrementally.** Keep the model, observation sets, query count, and metric definitions unchanged. Add candidate feedback and validated-unique archiving in successive improved versions.
4. **Aggregate and compare results.** For each seed, average the 30 per-observation-set metrics. The table below macro-averages the three seed means with equal weight. Percentages are proportions multiplied by 100; parenthesized values show percentage-point changes relative to baseline.

## Three Improved Algorithms

All three algorithms are inference-time prompting and feedback improvements. They do not retrain or replace the DeepSeek model.

### Improved 1: Valid Unique-Candidate Archive

After each generation, the program checks parsing, variables, operators, constants, maximum depth, and consistency with every observation. Only candidates that pass all checks and have not appeared under the mechanistic structural key are archived. The archive is included in subsequent prompts, and invalid or duplicate candidates receive targeted feedback.

**Core idea:** Separate verified valid structures from raw model output, and encourage the model to generate a different structure on subsequent queries.

### Improved 2: Validity-First Feedback

Improved 2 builds on the validated archive and gives validity priority over novelty. When a candidate is invalid or outside the hypothesis space, feedback explains the mismatch or constraint violation. Structural diversity is requested only after the candidate satisfies all observations and constraints. Duplicate valid structures are also explicitly discouraged.

**Core idea:** Resolve correctness before diversity, avoiding feedback that encourages structurally novel but invalid expressions.

### Improved 3: Complete Observation-Mismatch Diagnostics

Improved 3 retains the validity-first feedback and valid-unique archive, but reports every visible observation mismatch in a candidate instead of only the first mismatch. The model receives a complete list of inputs that the candidate incorrectly evaluates.

**Core idea:** Replace partial, single-mismatch feedback with complete constraint diagnostics to reduce repeated invalid generations.

## Results and Comparison

The table reports the macro-average over three seeds. These values are not recomputed by pooling all 900 queries into a single proportion.

| Method | Validity ↑ | Uniqueness / Novelty ↑ | Recovery ↑ |
|---|---:|---:|---:|
| Baseline | 93.78% | 17.67% | 42.33% |
| Improved 1 | 68.56% (−25.22 pp) | **40.33% (+22.67 pp)** | 74.67% (+32.33 pp) |
| Improved 2 | **80.33% (−13.44 pp)** | 37.89% (+20.22 pp) | **75.96% (+33.63 pp)** |
| Improved 3 | 81.00% (−12.78 pp) | 36.56% (+18.89 pp) | 73.81% (+31.48 pp) |

**Interpretation:** The baseline has the highest validity but repeatedly generates the same answers, resulting in low novelty and recovery. All three improved methods substantially increase novelty and recovery, while validity decreases. This indicates a correctness–diversity tradeoff when the model is explicitly encouraged to explore alternative structures.

Among the three improvements:

- **Improved 1** achieves the highest novelty (40.33%).
- **Improved 2** achieves the highest recovery (75.96%) and the highest validity among the improved methods (80.33%).
- **Improved 3** achieves the highest validity among all improved methods (81.00%), but does not exceed Improved 2 on recovery in this experiment.

> The percentage-point changes compare means across methods. The p-values stored in the JSON files test whether each metric mean differs from zero; they do not test significance between methods.

### Result Files

Every main result below contains 30 samples with 10 queries per sample. Seed-level values correspond to `statistics.{valid_rate,novelty_rate,recovery_rate}.mean` in the JSON files.

| Method | Seed 33550336 | Seed 33550337 | Seed 33550338 |
|---|---|---|---|
| Baseline | [JSON](results/boolean_2var_deepseek-chat_base/boolean_2var_deepseek-chat_seed33550336.json) | [JSON](results/boolean_2var_deepseek-chat_base/boolean_2var_deepseek-chat_seed33550337.json) | [JSON](results/boolean_2var_deepseek-chat_base/boolean_2var_deepseek-chat_seed33550338.json) |
| Improved 1 | [JSON](results/boolean_2var_deepseek-chat_improved_1/boolean_2var_deepseek-chat_improved_seed33550336.json) | [JSON](results/boolean_2var_deepseek-chat_improved_1/boolean_2var_deepseek-chat_improved_seed33550337.json) | [JSON](results/boolean_2var_deepseek-chat_improved_1/boolean_2var_deepseek-chat_improved_seed33550338.json) |
| Improved 2 | [JSON](results/boolean_2var_deepseek-chat_improved_2/boolean_2var_deepseek-chat_improved_2_seed33550336.json) | [JSON](results/boolean_2var_deepseek-chat_improved_2/boolean_2var_deepseek-chat_improved_2_seed33550337.json) | [JSON](results/boolean_2var_deepseek-chat_improved_2/boolean_2var_deepseek-chat_improved_2_seed33550338.json) |
| Improved 3 | [JSON](results/boolean_2var_deepseek-chat_improved_3/boolean_2var_deepseek-chat_improved_3_seed33550336.json) | [JSON](results/boolean_2var_deepseek-chat_improved_3/boolean_2var_deepseek-chat_improved_3_seed33550337.json) | [JSON](results/boolean_2var_deepseek-chat_improved_3/boolean_2var_deepseek-chat_improved_3_seed33550338.json) |

## Directory Structure

- `boolean_dataset.py`: Boolean expressions, observations, and dataset-generation/eumeration logic.
- `boolean_benchmark.py`: Original baseline benchmark.
- `boolean_benchmark_improved_1.py`: Valid unique-candidate archive and candidate feedback.
- `boolean_benchmark_improved_2.py`: Validity-first candidate feedback.
- `boolean_benchmark_improved_3.py`: Candidate feedback listing every observation mismatch.
- `modules/`: LLM interfaces and model definitions.
- `config/`: Model and benchmark configuration.
- `datasets/`: Pre-generated Boolean benchmark datasets.
- `results/`: Baseline and improved-method result JSON files.
