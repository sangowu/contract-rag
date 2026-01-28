# pach_reranked_hybrid_rag Evaluation Report

Auto-generated evaluation report.

- **Experiment**: pach_reranked_hybrid_rag
- **Created**: 2026-01-28 10:17:54
- **Total Samples**: 100

## Overall Metrics

### Retrieval Metrics

| K | Hit@K | MRR@K | Recall@K | Precision@K |
|---|-------|-------|----------|-------------|
| 1 | 0.1100 | 0.1100 | 0.1100 | 0.1100 |
| 3 | 0.2100 | 0.1533 | 0.2100 | 0.0700 |
| 5 | 0.2700 | 0.1673 | 0.2650 | 0.0540 |
| 10 | 0.4100 | 0.1844 | 0.3917 | 0.0410 |

### Answer Quality Metrics

| Metric | Value |
|--------|-------|
| F1 | 0.8431 |
| Exact Match | 0.7500 |
| Precision | 0.8507 |
| Recall | 0.8600 |

### Simplified RAGAS Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| simple_context_coverage | 0.1710 | Context Coverage (≈Faithfulness) |
| simple_answer_context_overlap | 0.0228 | Answer-Context Overlap |
| simple_ground_truth_coverage | 0.1180 | Ground Truth Coverage (≈Context Recall) |

### Latency Metrics

| Stage | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|-------|-----------|----------|----------|----------|
| Retrieval | 839.9 | 856.0 | 1021.4 | 1250.2 |
| Generation | 552.9 | 31.1 | 163.4 | 1151.4 |
| Total | 1392.9 | 914.6 | 1208.7 | 1994.9 |

## Results by Category

### Anti-Assignment

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.3333
- **F1**: 1.0000
- **Exact Match**: 1.0000

#### Example 1

- **Query**: Does this agreement include any 'Anti-Assignment' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: Yes
- **Metrics**: hit@10=1.0000, mrr@10=0.3333, f1=1.0000, em=1.0000

---

### Agreement Date

- **Samples**: 4
- **Hit@10**: 1.0000
- **MRR@10**: 0.7083
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Agreement Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 6/29/06
- **Pred Answer**: 06/29/2006
- **Metrics**: hit@10=1.0000, mrr@10=1.0000, f1=nan, em=0.0000

#### Example 2

- **Query**: What is the Agreement Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 7/1/17
- **Pred Answer**: 07/01/2017
- **Metrics**: hit@10=1.0000, mrr@10=1.0000, f1=nan, em=0.0000

#### Example 3

- **Query**: What is the Agreement Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 3/1/05
- **Pred Answer**: 03/01/2005
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=nan, em=0.0000

---

### Audit Rights

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.5000
- **F1**: 1.0000
- **Exact Match**: 1.0000

#### Example 1

- **Query**: Does this agreement include any 'Audit Rights' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=1.0000, em=1.0000

---

### Governing Law

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 1.0000
- **F1**: nan
- **Exact Match**: 1.0000

#### Example 1

- **Query**: What is the Governing Law of this agreement? Answer with the name of a state or country only.
- **Gold Answer**: New York
- **Pred Answer**: New York
- **Metrics**: hit@10=1.0000, mrr@10=1.0000, f1=nan, em=1.0000

---

### Covenant Not To Sue

- **Samples**: 2
- **Hit@10**: 1.0000
- **MRR@10**: 0.1000
- **F1**: 1.0000
- **Exact Match**: 1.0000

#### Example 1

- **Query**: Does this agreement include any 'Covenant Not To Sue' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.1000, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Covenant Not To Sue' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.1000, f1=1.0000, em=1.0000

---

### Document Name

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.5000
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Document Name in this agreement?
- **Gold Answer**: e-business Hosting Agreement
- **Pred Answer**: The Document Name in this agreement is the e-business Hosting Agreement.
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=nan, em=0.0000

---

### Insurance

- **Samples**: 3
- **Hit@10**: 1.0000
- **MRR@10**: 0.2750
- **F1**: 1.0000
- **Exact Match**: 1.0000

#### Example 1

- **Query**: Does this agreement include any 'Insurance' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Insurance' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.1250, f1=1.0000, em=1.0000

#### Example 3

- **Query**: Does this agreement include any 'Insurance' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.2000, f1=1.0000, em=1.0000

---

### Parties

- **Samples**: 4
- **Hit@10**: 1.0000
- **MRR@10**: 0.6146
- **F1**: 0.6434
- **Exact Match**: 0.0000

#### Example 1

- **Query**: Who are the parties to this agreement? List the entity or individual names separated by semicolons.
- **Gold Answer**: Enterprise Transportation Company ("Carrier"); Enterprise Products  Operating L.P. ("Shipper")
- **Pred Answer**: Shipper; Consignee; Carrier
- **Metrics**: hit@10=1.0000, mrr@10=0.1250, f1=0.3077, em=0.0000

#### Example 2

- **Query**: Who are the parties to this agreement? List the entity or individual names separated by semicolons.
- **Gold Answer**: Bravatek Solutions ("Bravatek"); Sibannac Inc. ("COMPANY")
- **Pred Answer**: Bravatek Solutions, Inc.; Sibannac, Inc.
- **Metrics**: hit@10=1.0000, mrr@10=1.0000, f1=0.7273, em=0.0000

#### Example 3

- **Query**: Who are the parties to this agreement? List the entity or individual names separated by semicolons.
- **Gold Answer**: Dazzle Interactive Network Technologies Co., Ltd. ("Party A"); Shenzhen iDreamSky Technology Co., Ltd. ("Party B")
- **Pred Answer**: Party A; Party B; Shenzhen iDreamSky Technology Co., Ltd.; Dazzle Interactive Network Technologies Co., Ltd.
- **Metrics**: hit@10=1.0000, mrr@10=0.3333, f1=1.0000, em=0.0000

---

### Post-Termination Services

- **Samples**: 3
- **Hit@10**: 1.0000
- **MRR@10**: 0.2454
- **F1**: 1.0000
- **Exact Match**: 1.0000

#### Example 1

- **Query**: Does this agreement include any 'Post-Termination Services' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.1250, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Post-Termination Services' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.1111, f1=1.0000, em=1.0000

#### Example 3

- **Query**: Does this agreement include any 'Post-Termination Services' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=1.0000, em=1.0000

---

### Effective Date

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 1.0000
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Effective Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 4/7/17
- **Pred Answer**: 04/07/2017
- **Metrics**: hit@10=1.0000, mrr@10=1.0000, f1=nan, em=0.0000

---
