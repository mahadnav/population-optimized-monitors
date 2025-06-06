import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def classify_population_density(data):
    data['cluster'] = np.where(data['population'] < 5000, 0, 1)
    density_mapping = {0: 'Low', 1: 'High'}
    data['Density'] = data['cluster'].map(density_mapping)
    return data

def distance(lat1, long1, lat2, long2):
    return haversine((lat1, long1), (lat2, long2), unit=Unit.KILOMETERS)

def weighted_kmeans(data, centers, k, it_max=300):
    points = data.to_dict(orient='records')
    d = 2  # number of dimensions (lat, long)
    
    for center in centers:
        center['n'] = 0
        center['w'] = 0

    for point in points:
        distances = [distance(point['lat'], point['long'], center["coords"][1], center["coords"][0])**2 for center in centers]
        idx = np.argmin(distances)
        point['cluster'] = idx
        centers[idx]["n"] += 1
        centers[idx]["w"] += point['population']

    for center in centers:
        center["coords"] = np.zeros(d)

    for point in points:
        centers[point['cluster']]["coords"] += np.array([point['long'], point['lat']]) * point['population']
    for center in centers:
        center["coords"] /= center['w']

    it_num = 0
    distsq = np.zeros(k)
    while it_num < it_max:
        it_num += 1
        swap = 0
        for point in points:
            ci = point['cluster']
            if centers[ci]['n'] <= 1:
                continue

            for cj, center in enumerate(centers):
                lat1, long1 = point['lat'], point['long']
                lat2, long2 = center["coords"][1], center["coords"][0]
                if ci == cj:
                    distsq[cj] = (distance(lat1, long1, lat2, long2)**2 * center['w']) / (center['w'] - point['population'])
                elif centers[cj]['n'] == 0:
                    centers[cj]["coords"] = np.array([long1, lat1])
                    distsq[cj] = 0
                else:
                    distsq[cj] = (distance(lat1, long1, lat2, long2)**2 * center['w']) / (center['w'] + point['population'])

            nearest_cluster = np.argmin(distsq)
            if nearest_cluster == ci:
                continue

            cj = nearest_cluster
            centers[ci]["coords"] = (centers[ci]['w'] * centers[ci]["coords"] - point['population'] * np.array([point['long'], point['lat']])) / (centers[ci]['w'] - point['population'])
            centers[cj]["coords"] = (centers[cj]['w'] * centers[cj]["coords"] + point['population'] * np.array([point['long'], point['lat']])) / (centers[cj]['w'] + point['population'])
            centers[ci]['n'] -= 1
            centers[cj]['n'] += 1
            centers[ci]['w'] -= point['population']
            centers[cj]['w'] += point['population']

            point['cluster'] = cj
            swap += 1
        
        if swap == 0:
            break
    
    data['cluster'] = [point['cluster'] for point in points]

    # Calculate SSE
    sse = 0
    for point in points:
        ci = point['cluster']
        lat1, long1 = point['lat'], point['long']
        lat2, long2 = centers[ci]["coords"][1], centers[ci]["coords"][0]
        sse += distance(lat1, long1, lat2, long2)**2 * point['population']

    return data, centers, it_num, sse

def randomize_initial_cluster(data, k):
    indices = data.index.tolist()
    random.shuffle(indices)
    centers = []
    for i in indices[:k]:
        centers.append({"coords": np.array([data.at[i, 'long'], data.at[i, 'lat']])})
    return centers


def plot_clusters(data, k):
    vals = data[['population', 'long', 'lat']].copy()
    sampled = vals.sample(int(0.7 * len(data)))
    centers = randomize_initial_cluster(sampled, k)
    points, centers, iters, sse = weighted_kmeans(vals, centers, k)

    # Compute total population per cluster
    cluster_populations = points.groupby('cluster')['population'].sum().to_dict()

    # Extract centroids
    centroids = pd.DataFrame(centers)
    clat = [x[0][1] for _, x in centroids.iterrows()]
    clong = [x[0][0] for _, x in centroids.iterrows()]

    plt.figure(figsize=(10, 10))
    colors = [
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
        '#6a3d9a',
        '#ffff99',
        '#b15928',
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
        '#6a3d9a',
        '#ffff99',
        '#b15928',
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
        '#6a3d9a',
        '#ffff99',
        '#b15928',
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
        '#6a3d9a',
        '#ffff99',
        '#b15928',
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
        '#6a3d9a',
        '#ffff99',
        '#b15928',
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
        '#6a3d9a',
        '#ffff99',
        '#b15928']

    # Plot each cluster as colored scatter
    for cluster in range(k):
        cluster_data = points[points['cluster'] == cluster]
        plt.scatter(cluster_data['long'], cluster_data['lat'], 
                    label=f'Cluster {cluster}', s=40, color=colors[cluster], alpha=0.8)

    # Plot centroids and annotate with population
    for i in range(k):
        plt.text(clong[i], clat[i], f"{int(i):,}", 
                 fontsize=12, ha='center', va='bottom', color='black', fontweight='bold')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Return centroids with annotations
    centroids['clat'] = clat
    centroids['clong'] = clong
    centroids['population'] = centroids.index.map(cluster_populations)
    centroids.drop(columns=['coords', 'n', 'w'], inplace=True, errors='ignore')
    return points, centroids

# def cluster_analysis(data, k_max):

#     print('Average population density:', int(data['population'].sum()/len(data)))

#     vals = data[['population', 'long', 'lat']]

#     # Randomly select n points and spatially distribute them
#     sampled = vals.sample(int(0.7 * len(vals)))

#     sse_list = []
#     k_range = range(1, k_max+1)

#     fig, axes = plt.subplots(nrows=(len(k_range) + 2) // 3, ncols=3, figsize=(25, 5 * ((len(k_range) + 2) // 3)))
#     axes = axes.flatten()

#     for i, k in enumerate(k_range):
#         centers = randomize_initial_cluster(sampled, k)
#         points, centers, iters, sse = weighted_kmeans(vals, centers, k)

#         centers_df = pd.DataFrame(centers)
#         lat = [centers_df['coords'][x][1] for x in range(len(centers_df))]
#         lon = [centers_df['coords'][x][0] for x in range(len(centers_df))]

#         centroids = pd.DataFrame(centers)
#         centroids.drop(columns=['n', 'w'], inplace=True)

#         clat = [x[0][1] for _, x in centroids.iterrows()]
#         clong = [x[0][0] for _, x in centroids.iterrows()]

#         centroids['clat'] = clat
#         centroids['clong'] = clong
#         centroids.drop(columns=['coords'], inplace=True)

#         ax = axes[i]
#         scatter = ax.scatter(data['long'], data['lat'], 
#                             c=data['population'], cmap='RdBu_r', s=20, alpha=1)
#         ax.scatter(clong, clat, s=150, color='white', alpha=1)
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#         ax.set_title(f'k = {k}', fontsize=16)
#         plt.colorbar(scatter, ax=ax)

#         sse_list.append(sse)

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.show()

#     plt.plot(k_range, sse_list, 'ko-')
#     plt.xlabel('K')
#     plt.ylabel('Sum of Squared Errors (SSE)')
#     plt.show()


    def plot_clusters(data, k):
        vals = data[['population', 'long', 'lat']].copy()
        sampled = vals.sample(int(0.7 * len(data)))
        centers = randomize_initial_cluster(sampled, k)
        points, centers, iters, sse = weighted_kmeans(vals, centers, k)

        # Compute total population per cluster
        cluster_populations = points.groupby('cluster')['population'].sum().to_dict()

        # Extract centroids
        centroids = pd.DataFrame(centers)
        clat = [x[0][1] for _, x in centroids.iterrows()]
        clong = [x[0][0] for _, x in centroids.iterrows()]

        plt.figure(figsize=(10, 10))
        colors = [
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928']

        # Plot each cluster as colored scatter
        for cluster in range(k):
            cluster_data = points[points['cluster'] == cluster]
            plt.scatter(cluster_data['long'], cluster_data['lat'], 
                        label=f'Cluster {cluster}', s=40, color=colors[cluster], alpha=0.8)

        # Plot centroids and annotate with population
        for i in range(k):
            plt.text(clong[i], clat[i], f"{int(i):,}", 
                    fontsize=12, ha='center', va='bottom', color='black', fontweight='bold')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Return centroids with annotations
        centroids['clat'] = clat
        centroids['clong'] = clong
        centroids['population'] = centroids.index.map(cluster_populations)
        centroids.drop(columns=['coords', 'n', 'w'], inplace=True, errors='ignore')
        return points, centroids
# def cluster_metrics(data, centers_df):
#     cluster_metrics = []

#     for i, row in centers_df.iterrows():
#         cluster_points = data[data['cluster'] == i]
#         cluster_pop = round(cluster_points['population'].sum())

#         avg_dist = np.average(
#             [distance(r['lat'], r['long'], row['clat'], row['clong']) for _, r in cluster_points.iterrows()],
#             weights=cluster_points['population']
#         ) if cluster_pop > 0 else 0

#         max_dist = max(
#             [distance(r['lat'], r['long'], row['clat'], row['clong']) for _, r in cluster_points.iterrows()],
#             default=0
#         )

#         coverage_5km = cluster_points[
#             cluster_points.apply(lambda r: distance(r['lat'], r['long'], row['clat'], row['clong']) <= 5, axis=1)
#         ]['population'].sum()


#         cluster_metrics.append({
#             "Cluster ID": int(i),
#             "Population": int(cluster_pop),
#             "Average Distance (km)": round(avg_dist, 1),
#             "Max Distance (km)": round(max_dist, 1),
#             "Area (km²)": int(len(cluster_points)),
#             "Coverage within 5km": round(coverage_5km, 1) if cluster_pop else 0,
#         })

#     df_metrics = pd.DataFrame(cluster_metrics)
#     df_metrics.loc['Average'] = df_metrics.drop(columns=['Cluster ID']).mean()
#     df_metrics.loc['Average', 'Cluster ID'] = 'All Clusters'

#     return df_metrics
