import pandas as pd
import matplotlib.pyplot as plt

ax = plt.gca()
df = pd.read_csv("news_original.csv")
df["loss"] = df["loss_r"] - df["loss_d"]
df.reset_index().plot(kind='line', x='index', y='loss_r', ax=ax)

df = pd.read_csv("news_sum_0.7.csv")
df["loss"] = df["loss_r"] - df["loss_d"]
df.reset_index().plot(kind='line', x='index', y='loss_r', color='red', ax=ax)

plt.show()
