import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("mnist_adam.csv")
df2 = pd.read_csv("mnist_nadam.csv")
ax = plt.gca()
df1.reset_index().plot(kind='line', x='index', y='f1', ylabel="Adam", ax=ax)
df1.reset_index().plot(kind='line', x='index', y='f1', ylabel="NAdam", color='red', ax=ax)

# plt.show()
plt.savefig("mnist.png")
