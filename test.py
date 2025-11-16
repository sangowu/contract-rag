import pandas as pd

df = pd.read_csv("/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv")
check_columns = ['clause_type', 'query', 'gold_answer_text', 'gold_chunk_ids']

check_df = df[check_columns]
print(check_df.head(20))