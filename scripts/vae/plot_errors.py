import pickle
import matplotlib.pyplot as plt
import seaborn as sns

with open("embeddings.pickle", "rb") as f:
    data = pickle.load(f)

sns.set_style("whitegrid")
sns.set_context("paper")


N, bins, patches = plt.hist(data["errors"], bins=30)

colors    = ["#7995c4b0", "#ff0000b0"]
divisions = [range(12), range(12, 30)]
labels    = ["Successful Reconstruction", "Failed Reconstruction"]

for d in divisions:
    patches[list(d)[0]].set_label(labels[divisions.index(d)])
    for i in d:
        patches[i].set_color(colors[divisions.index(d)])

plt.xlabel("Reconstruction error")
plt.ylabel("Number of images")
plt.legend()
plt.savefig("error_fig.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)
plt.show()
