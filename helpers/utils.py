import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from geopy.distance import geodesic

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

def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def merge_close_centroids(centroids, threshold=2):
    if centroids.empty:
        return centroids
    merged_centroids = []
    used = set()
    for i, row1 in centroids.iterrows():
        if i in used: continue
        close_centroids = [row1]
        for j, row2 in centroids.iterrows():
            if i != j and j not in used:
                distance = calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon']))
                if distance < threshold:
                    close_centroids.append(row2)
                    used.add(j)
        if len(close_centroids) > 1:
            mean_lat = np.mean([c['lat'] for c in close_centroids])
            mean_long = np.mean([c['lon'] for c in close_centroids])
            merged_centroids.append({'lat': mean_lat, 'lon': mean_long})
        else:
            merged_centroids.append({'lat': row1['lat'], 'lon': row1['lon']})
        used.add(i)
    new_centroids = pd.DataFrame(merged_centroids)
    for i, row1 in new_centroids.iterrows():
        for j, row2 in new_centroids.iterrows():
            if i != j:
                if calculate_distance((row1['lat'], row1['lon']), (row2['lat'], row2['lon'])) < threshold:
                    return merge_close_centroids(new_centroids, threshold)
    return new_centroids