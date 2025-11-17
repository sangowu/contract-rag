import pandas as pd

df = pd.read_csv("/root/autodl-tmp/results/csv/cuad_v1_e2e_test_20_samples.csv")
# check_columns = ['clause_type', 'query', 'gold_answer_text', 'gold_chunk_ids']
check_columns = ['answer_type', 'gold_answer', 'model_answer', 'hit@k', 'rr@k', 'recall@k', 'f1', 'em', 'acc']
# check_df = df[check_columns]
# print(check_df.head(20))
print(df[check_columns].head(10)) 
# print(df.columns)