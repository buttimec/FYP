import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Progress bar
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import threading
import time

#based off:


# @ARTICLE{10296014,
#          author={Tobin, Joshua and Zhang, Mimi},
# journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
# title={A Theoretical Analysis of Density Peaks Clustering and the Component-Wise Peak-Finding Algorithm},
# year={2024},
# volume={46},
# number={2},
# pages={1109-1120},
# doi={10.1109/TPAMI.2023.3327471}}
# https://github.com/tobinjo96/CPFcluster

#mean shift was used as my data was not suited to other methods

#GenAi used to debug errors and add the progress bars

csv_path = "D:/FYP/Data/normalized_ndvi_data_with_sediment.csv"
npy_path = "D:/FYP/Data/ndvi_data.npy"
output_csv_path = "D:/FYP/Data/mean_shift_clusters.csv"

# Function to convert CSV to NPY with progress bar as the runtime was significant
def convert_csv_to_npy(csv_path, npy_path):
    print("Converting CSV to NPY format...")
    df = pd.read_csv(csv_path)

    # NDVI column names
    ndvi_columns = [
        'NDVI_2016H', 'NDVI_2016L', 'NDVI_2017H', 'NDVI_2017L',
        'NDVI_2018H', 'NDVI_2018L', 'NDVI_2019H', 'NDVI_2019L',
        'NDVI_2020H', 'NDVI_2020L', 'NDVI_2021H', 'NDVI_2021L',
        'NDVI_2022H', 'NDVI_2022L', 'NDVI_2023H', 'NDVI_2023L'
    ]

    tqdm.pandas(desc="Cleaning Data")
    df_clean = df.dropna(subset=ndvi_columns).progress_apply(lambda x: x, axis=1)

    # Save as NPY
    np.save(npy_path, df_clean[ndvi_columns].values)
    print(f"NPY file saved at {npy_path}")
    return df_clean

# Convert CSV to NPY if needed
df_clean = convert_csv_to_npy(csv_path, npy_path)

# Load NPY data
ndvi_data = np.load(npy_path)

print("Standardising NDVI values...")
scaler = StandardScaler()
ndvi_scaled = scaler.fit_transform(ndvi_data)

# Estimate bandwidth for mean shift
print("Estimating bandwidth for Mean Shift...")
bandwidth = estimate_bandwidth(ndvi_scaled, quantile=0.2, n_samples=50000)
print(f"Estimated bandwidth: {bandwidth}")

# Apply Mean Shift clustering with a progress bar u

mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = None

def clustering_job():
    global labels
    labels = mean_shift.fit_predict(ndvi_scaled)

# run clustering in a separate thread to help runtime and make the progress bar work properly
clustering_thread = threading.Thread(target=clustering_job)
clustering_thread.start()

# Display a progress bar while clustering is running
with tqdm(desc="Clustering Data", unit="s") as pbar:
    while clustering_thread.is_alive():
        time.sleep(1)
        pbar.update(1)
clustering_thread.join()

print(f"Clustering complete. Found {len(np.unique(labels))} clusters.")

# Save results to CSV
df_clean['Cluster'] = labels
df_clean.to_csv(output_csv_path, index=False)
print(f"Cluster results saved to {output_csv_path}")

# Visualisation Functions suggested by GenAI (UMap, t-sne and pca)
def plot_clusters_umap(X, labels, sample_size=30000, n_neighbors=15, min_dist=0.3, random_state=42):
    # Visualizes clusters using UMAP.
    if X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X, labels = X[indices], labels[indices]

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X_reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10)

    plt.title("Clusters and Outliers (UMAP)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.show()

def plot_clusters_pca(X, labels):
    #Visualises clusters using PCA.
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X_reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10)

    plt.title("Clusters and Outliers (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

def plot_clusters_tsne(X, labels, sample_size=30000, perplexity=30, random_state=42):
    #Visualses clusters using t-SNE
    if X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X, labels = X[indices], labels[indices]

    X_reduced = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(X)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X_reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10)

    plt.title("Clusters and Outliers (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

# Run the plots
plot_clusters_umap(ndvi_scaled, labels)
plot_clusters_pca(ndvi_scaled, labels)
plot_clusters_tsne(ndvi_scaled, labels)

# Play multiple beeps for 10 seconds when the program finishes (was used to alert me when it was done as run time was long)
# (genAI helped)
try:
    import winsound
    start_time = time.time()
    while time.time() - start_time < 10:
        winsound.Beep(1000, 200)  # Beep at 1000 Hz for 200 ms
        time.sleep(0.1)
except ImportError:
    # Fallback: Print bell character repeatedly for 10 seconds
    start_time = time.time()
    while time.time() - start_time < 3:
        print('\a', end='', flush=True)
        time.sleep(0.3)
