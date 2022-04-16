import pandas as pd

df = pd.read_csv("mnist_original.csv")
f1 = df["f1"]
acc = df["acc"]
count = 0
max = 0
i = 0

for d in f1.tolist():
    count += 1
    i += 1
    if max < d:
        max = d
        count = 0
    if max - d <= 0.01:
        count -= 1
    if count >= 15:
        print(i)
        print(max)
        print(f1.max())
        print(f1.idxmax())
        break

# import pandas as pd
# import matplotlib.pyplot as plt
#
# dataset = "cifar10"
# df = pd.read_csv(f"full_{dataset}_adam.csv")
# max_f1 = df["f1"].max()
# id_max_f1 = df["f1"].idxmax()
# max_acc = df["acc"].max()
# id_max_acc = df["acc"].idxmax()
#
# ax = plt.gca()
# df.reset_index().plot(kind='line', x='index', y='f1', ax=ax)
# df.reset_index().plot(kind='line', x='index', y='acc', color='red', ax=ax)
# # df.reset_index().plot(kind='line', x='index', y='loss_r', color="green", ax=ax)
# # df.reset_index().plot(kind='line', x='index', y='loss_d', color='orange', ax=ax)
#
# plt.plot(id_max_f1, max_f1, "o")
# plt.plot(id_max_acc, max_acc, "x")
# plt.show()
# plt.savefig(f"{dataset}.png")
