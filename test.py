import pandas as pd
df = pd.read_parquet("../local_datasets/combined_artem9k_ai-text-detection_haveMapped.parquet")
has_missing_values = df.isna().any().any()
print(f"是否有空值: {has_missing_values}")




