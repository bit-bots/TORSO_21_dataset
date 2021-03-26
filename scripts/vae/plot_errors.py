import pickle
import matplotlib.pyplot as plt

with open("embeddings.pickle", "rb") as f:
    data = pickle.load(f)

plt.hist(data["errors"], density=True, bins=30)

plt.show() 
