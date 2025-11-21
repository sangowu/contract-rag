import pandas as pd

df = pd.read_csv("/root/autodl-tmp/results/csv/cuad_v1_e2e_test_20_samples.csv")
check_columns = ['answer_type', 'gold_answer', 'model_answer', 'hit@k', 'rr@k', 'recall@k', 'f1', 'em', 'acc']

print(df[check_columns].head(10)) 
