import pickle
import matplotlib.pyplot as plt
import seaborn as sns

with open("embeddings.pickle", "rb") as f:
    data = pickle.load(f)

sns.set_style("darkgrid")

plot = sns.distplot(data["errors"], density=False, bins=30)

plt.axvline(100, 0, data["errors"].mean() + data["errors"].std() * 1.64)

plot.set(xlabel='Reconstruction error', ylabel='Number of images')

plt.show() 
