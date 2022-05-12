import pandas as pd
import matplotlib.pyplot as plt

dataset = "mnist"
loss_type = "3rd"
ratios = [1, 0.8, 0.7, 0.6, 0.5, 0.4]
summary = {"ratio": [], "acc": [], "f1": []}
df = pd.read_csv(f"{dataset}_2nd.csv")
# summary["ratio"].append(1)
# summary["acc"].append(df["acc"].max())
# summary["f1"].append(df["f1"].max())
#
# for ratio in ratios:
#     df = pd.read_csv(f"{dataset}_{loss_type}_{ratio}.csv")
#     summary["ratio"].append(ratio)
#     summary["acc"].append(df["acc"].max())
#     summary["f1"].append(df["f1"].max())
#
# df = pd.DataFrame(summary)
df["ratio"] = ratios
print(df)
ax = plt.gca()
# ax2 = plt.gca
# df1.reset_index().plot(kind='line', x='index', y='acc', ax=ax1)
df.plot(kind='line', x='ratio', y='acc', color='red', ax=ax)
df.plot(kind='line', x='ratio', y='f1', ax=ax, title="MNIST")


# df1.reset_index().plot(kind='line', x='index', y='f1', ax=ax2)
# df2.reset_index().plot(kind='line', x='index', y='f1', color='red', ax=ax2)

plt.ylim([0.9, 1.0])
plt.show()
# plt.savefig(f'{dataset}_{loss_type}.png')
