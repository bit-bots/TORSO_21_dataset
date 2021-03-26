import pickle
import matplotlib.pyplot as plt

with open("embeddings.pickle", "rb") as f:
    data = pickle.load(f)

plt.hist(data["errors"], density=False, bins=30)
plt.ylabel('Number of images')
plt.xlabel('Reconstruction error')

plt.show() 
