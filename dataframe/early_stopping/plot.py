# import pandas as pd
# import matplotlib.pyplot as plt
#
# dataset = "cifar10"
# # df = pd.read_csv(f"{dataset}_adam.csv")
# df = pd.read_csv('archive/cifar10_original.csv')
# max_f1 = df["f1"].max()
# id_max_f1 = df["f1"].idxmax()
# max_acc = df["acc"].max()
# id_max_acc = df["acc"].idxmax()
#
# ax = plt.gca()
# df.reset_index().plot(kind='line', x='index', y='f1', ax=ax, xlabel='epoch', title='CIFAR10')
# df.reset_index().plot(kind='line', linestyle=":", x='index', y='acc', color='red', ax=ax, xlabel='epoch')
#
# plt.plot(id_max_f1, max_f1, "o")
# plt.plot(id_max_acc, max_acc, "x")
# plt.show()
# # plt.savefig(f"{dataset}.png")

import numpy as np
import matplotlib.pyplot as plt

X = ['MNIST', '20News', 'CIFAR10']
paper_acc = [96.51, 81.06, 87.22]
paper_f1 = [96.42, 81, 89.7]
experiment_acc = [96.74, 82.99, 90.15]
experiment_f1 = [96.67, 85.46, 87.66]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.1, paper_f1, 0.2, label='paper')
plt.bar(X_axis + 0.1, experiment_f1, 0.2, label='experiment')

plt.xticks(X_axis, X)
plt.xlabel("Dataset")
# plt.ylabel("Number of Students")
plt.title("F1")
plt.legend()
plt.show()