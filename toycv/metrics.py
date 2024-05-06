import pandas as pd

# 创建一个空的 DataFrame
df = pd.DataFrame(columns=["fold", "stage", "epoch", "acc", "f1score", "precision", "recall",""])

# 添加一行数据
new_row = {"acc": 0.99, "epoch": 18, "fold": 4, "step": "train"}
df.loc[len(df)] = new_row
df.loc[len(df)] = new_row

print(df)
