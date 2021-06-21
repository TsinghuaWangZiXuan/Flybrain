import numpy as np
from matplotlib import pyplot as plt
import umap
from matplotlib.colors import rgb2hex
from sklearn.cluster import DBSCAN, KMeans, OPTICS, Birch

# Define a reducer
reducer = umap.UMAP()

# Load data
data = np.load("./data/latent_vector.npy")

# Clustering
y = np.load('y_dna.npy')

colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(int(np.max(y) + 1))])
colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

# Train umap
embedding = reducer.fit_transform(data)
print(embedding.shape)

plt.figure()
for i in range(int(np.max(y) + 1)):
    # Visualize results
    plt.scatter(
        embedding[y == i, 0],
        embedding[y == i, 1]
    )

plt.title('UMAP Projection of the Latent Vector', fontsize=24)
plt.show()

# K-means
kmeans = KMeans(n_clusters=20)
kmeans.fit(embedding.astype('double'))

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest",
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect="auto", origin="lower")

plt.plot(embedding[:, 0], embedding[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
            color="w", zorder=10)

plt.title("K-means clustering on the latent vector (UMAP-reduced data)\n"
          "Centroids are marked with white cross")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
