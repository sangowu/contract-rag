# vanilla_rag Evaluation Report

Auto-generated evaluation report.

- **Experiment**: vanilla_rag
- **Created**: 2026-01-27 08:19:55
- **Total Samples**: 100

## Overall Metrics

### Retrieval Metrics

| K | Hit@K | MRR@K | Recall@K | Precision@K |
|---|-------|-------|----------|-------------|
| 1 | 0.0200 | 0.0200 | 0.0200 | 0.0200 |
| 3 | 0.0900 | 0.0517 | 0.0900 | 0.0300 |
| 5 | 0.1200 | 0.0577 | 0.1200 | 0.0240 |
| 10 | 0.1700 | 0.0649 | 0.1700 | 0.0170 |

### Answer Quality Metrics

| Metric | Value |
|--------|-------|
| F1 | 0.7493 |
| Exact Match | 0.6900 |
| Precision | 0.7487 |
| Recall | 0.7601 |

### Simplified RAGAS Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| simple_context_coverage | 0.5062 | Context Coverage (≈Faithfulness) |
| simple_answer_context_overlap | 0.0083 | Answer-Context Overlap |
| simple_ground_truth_coverage | 0.4312 | Ground Truth Coverage (≈Context Recall) |

### Latency Metrics

| Stage | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|-------|-----------|----------|----------|----------|
| Retrieval | 29.2 | 24.8 | 27.6 | 57.8 |
| Generation | 563.4 | 42.9 | 128.9 | 1096.0 |
| Total | 592.6 | 68.4 | 152.9 | 1124.7 |

## Results by Category

### Expiration Date

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.5000
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Expiration Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: perpetual
- **Pred Answer**: The expiration date of the agreement cannot be determined from the provided clauses. The date "August 26, 2014" is mentioned, but it is not clearly identified as the expiration date. Without explicit language stating the expiration date, it cannot be confirmed.
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=nan, em=0.0000

---

### Governing Law

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.5000
- **F1**: nan
- **Exact Match**: 1.0000

#### Example 1

- **Query**: What is the Governing Law of this agreement? Answer with the name of a state or country only.
- **Gold Answer**: New York
- **Pred Answer**: New York
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=nan, em=1.0000

---

### Effective Date

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.2000
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Effective Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 4/7/17
- **Pred Answer**: 04/07/2017
- **Metrics**: hit@10=1.0000, mrr@10=0.2000, f1=nan, em=0.0000

---

### Exclusivity

- **Samples**: 2
- **Hit@10**: 1.0000
- **MRR@10**: 0.2667
- **F1**: 1.0000
- **Exact Match**: 1.0000

#### Example 1

- **Query**: Does this agreement include any 'Exclusivity' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.2000, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Exclusivity' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: Yes
- **Metrics**: hit@10=1.0000, mrr@10=0.3333, f1=1.0000, em=1.0000

---

### Document Name

- **Samples**: 1
- **Hit@10**: 1.0000
- **MRR@10**: 0.2000
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Document Name in this agreement?
- **Gold Answer**: e-business Hosting Agreement
- **Pred Answer**: The Document Name in this agreement is "e-business hosting agreement".
- **Metrics**: hit@10=1.0000, mrr@10=0.2000, f1=nan, em=0.0000

---

### Agreement Date

- **Samples**: 4
- **Hit@10**: 0.7500
- **MRR@10**: 0.2440
- **F1**: nan
- **Exact Match**: 0.0000

#### Example 1

- **Query**: What is the Agreement Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 6/29/06
- **Pred Answer**: 06/29/2006
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=nan, em=0.0000

#### Example 2

- **Query**: What is the Agreement Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 7/1/17
- **Pred Answer**: 07/01/2017
- **Metrics**: hit@10=1.0000, mrr@10=0.1429, f1=nan, em=0.0000

#### Example 3

- **Query**: What is the Agreement Date of this agreement? Answer strictly in the format mm/dd/yyyy.
- **Gold Answer**: 3/1/05
- **Pred Answer**: 03/01/2005
- **Metrics**: hit@10=1.0000, mrr@10=0.3333, f1=nan, em=0.0000

---

### Termination For Convenience

- **Samples**: 4
- **Hit@10**: 0.5000
- **MRR@10**: 0.3750
- **F1**: 0.7500
- **Exact Match**: 0.7500

#### Example 1

- **Query**: Does this agreement include any 'Termination For Convenience' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: Yes
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Termination For Convenience' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=0.0000, mrr@10=0.0000, f1=1.0000, em=1.0000

#### Example 3

- **Query**: Does this agreement include any 'Termination For Convenience' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: Yes
- **Metrics**: hit@10=1.0000, mrr@10=1.0000, f1=1.0000, em=1.0000

---

### Competitive Restriction Exception

- **Samples**: 2
- **Hit@10**: 0.5000
- **MRR@10**: 0.2500
- **F1**: 0.5000
- **Exact Match**: 0.5000

#### Example 1

- **Query**: Does this agreement include any 'Competitive Restriction Exception' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: Yes
- **Metrics**: hit@10=0.0000, mrr@10=0.0000, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Competitive Restriction Exception' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.5000, f1=0.0000, em=0.0000

---

### Revenue/Profit Sharing

- **Samples**: 2
- **Hit@10**: 0.5000
- **MRR@10**: 0.0833
- **F1**: 0.5000
- **Exact Match**: 0.5000

#### Example 1

- **Query**: Does this agreement include any 'Revenue/Profit Sharing' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: No
- **Pred Answer**: No
- **Metrics**: hit@10=0.0000, mrr@10=0.0000, f1=1.0000, em=1.0000

#### Example 2

- **Query**: Does this agreement include any 'Revenue/Profit Sharing' clause? Answer strictly with 'Yes' or 'No'.
- **Gold Answer**: Yes
- **Pred Answer**: No
- **Metrics**: hit@10=1.0000, mrr@10=0.1667, f1=0.0000, em=0.0000

---

### Parties

- **Samples**: 4
- **Hit@10**: 0.5000
- **MRR@10**: 0.0670
- **F1**: 0.0306
- **Exact Match**: 0.0000

#### Example 1

- **Query**: Who are the parties to this agreement? List the entity or individual names separated by semicolons.
- **Gold Answer**: Enterprise Transportation Company ("Carrier"); Enterprise Products  Operating L.P. ("Shipper")
- **Pred Answer**: Not specified
- **Metrics**: hit@10=1.0000, mrr@10=0.1429, f1=0.0000, em=0.0000

#### Example 2

- **Query**: Who are the parties to this agreement? List the entity or individual names separated by semicolons.
- **Gold Answer**: Bravatek Solutions ("Bravatek"); Sibannac Inc. ("COMPANY")
- **Pred Answer**: Not specified
- **Metrics**: hit@10=1.0000, mrr@10=0.1250, f1=0.0000, em=0.0000

#### Example 3

- **Query**: Who are the parties to this agreement? List the entity or individual names separated by semicolons.
- **Gold Answer**: Dazzle Interactive Network Technologies Co., Ltd. ("Party A"); Shenzhen iDreamSky Technology Co., Ltd. ("Party B")
- **Pred Answer**: The parties to this agreement are not specified in the provided contract clauses. Therefore, based on the information given, the entity or individual names cannot be determined. Answer: (no names provided)
- **Metrics**: hit@10=0.0000, mrr@10=0.0000, f1=0.0000, em=0.0000

---
