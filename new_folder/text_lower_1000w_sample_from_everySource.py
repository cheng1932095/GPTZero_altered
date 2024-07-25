import  pandas as pd

dataset = pd.read_parquet("combined_artem9k_ai-text-detection_haveMapped.parquet")
dataset['text'] = dataset['text'].apply(lambda x: x.replace('\n', ''))

def count_words(text):
    # 计算单词数量
    return len(text.split())

def filter_by_word_count(dataset, max_word_count):
    # 计算每个样本的单词数量
    dataset['word_count'] = dataset['text'].apply(count_words)
    # 筛选单词数量小于 max_word_count 的样本
    filtered_dataset = dataset[dataset['word_count'] < max_word_count]
    return filtered_dataset

# 筛选出单词数小于1000的样本
filtered_dataset = filter_by_word_count(dataset, 20)


def sample_by_class(dataset, class_column, num_samples_per_class):
    # 按类别分组
    grouped = dataset.groupby(class_column)
    sampled_dfs = []

    for name, group in grouped:
        # 从每个类别中随机抽取样本
        sampled_group = group.sample(n=num_samples_per_class, random_state=1)
        sampled_dfs.append(sampled_group)

    # 合并所有抽取的样本
    sampled_dataset = pd.concat(sampled_dfs)
    return sampled_dataset


# 提取每个类别各1000个样本
sampled_dataset = sample_by_class(dataset, 'source', 25)

sampled_dataset.to_parquet('F:\\old_D\\PycharmProjects\\ai_text_detection\\local_datasets\\wLower_20_sampled_25_everySource.parquet')
