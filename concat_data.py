import pandas as pd
import os

# 设置文件夹路径
folder_path = 'F:\\old_D\\PycharmProjects\\ai_text_detection\\local_datasets'

# 文件编号列表
file_numbers = ['0000', '0001', '0002', '0003', '0004', '0005', '0006']

# 读取所有 Parquet 文件并合并
df_list = []  # 存储 DataFrame 的列表
for number in file_numbers:
    file_path = os.path.join(folder_path, f'artem9k_ai-text-detection-pile_{number}.parquet')
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        df_list.append(df)
    else:
        print(f"文件不存在: {file_path}")

# 合并所有 DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

print(os.getcwd())
# 打印合并后的 DataFrame 的前几行
combined_df.to_parquet('combined_artem9k_ai-text-detection.parquet')