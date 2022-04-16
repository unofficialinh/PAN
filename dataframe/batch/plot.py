import pandas as pd
import matplotlib.pyplot as plt

dataset = "mnist"
# batch_sizes = [300, 500, 1000, 2000, 3000, 4000, 6000]
batch_sizes = [500, 1000, 2000, 3000, 5000, 6000, 10000, 15000, 20000, 30000, 60000]

summary = {"batch_size": [], "acc": [], "f1": []}
# df = pd.read_csv("mnist_original.csv")
# summary["ratio"].append(1)
# summary["acc"].append(df["acc"].max())
# summary["f1"].append(df["f1"].max())

for batch_size in batch_sizes:
    df = pd.read_csv(f"{dataset}_{batch_size}.csv")
    summary["batch_size"].append(batch_size)
    summary["acc"].append(df["acc"].max())
    summary["f1"].append(df["f1"].max())

df = pd.DataFrame(summary)
ax = plt.gca()
ax2 = plt.gca()
# df1.reset_index().plot(kind='line', x='index', y='acc', ax=ax1)
df.plot(kind='line', x='batch_size', y='acc', color='red', ax=ax)
df.plot(kind='line', x='batch_size', y='f1', ax=ax)

# df1 = pd.read_csv(f"mnist_500.csv")
# df2 = pd.read_csv(f"mnist_1000.csv")

# df1.reset_index().plot(kind='line', x='index', y='acc', ylabel="500", ax=ax2)
# df2.reset_index().plot(kind='line', x='index', y='acc', ylabel="1000", color='red', ax=ax2)

# plt.show()
plt.savefig(f'{dataset}_batch.png')
