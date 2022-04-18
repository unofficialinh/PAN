import pandas as pd
import matplotlib.pyplot as plt

dataset = "cifar10"
df = pd.read_csv(f"{dataset}_adam.csv")
max_f1 = df["f1"].max()
id_max_f1 = df["f1"].idxmax()
max_acc = df["acc"].max()
id_max_acc = df["acc"].idxmax()

ax = plt.gca()
df.reset_index().plot(kind='line', x='index', y='f1', ax=ax)
df.reset_index().plot(kind='line', x='index', y='acc', color='red', ax=ax)

plt.plot(id_max_f1, max_f1, "o")
plt.plot(id_max_acc, max_acc, "x")
# plt.show()
plt.savefig(f"{dataset}.png")
