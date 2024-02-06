import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import csv

def load_data(filename):
    dir = open(filename, 'r')
    new_dict = csv.DictReader(dir)

    countries = []

    for data_dict in new_dict:
        countries.append(data_dict)

    return countries


def calc_features(row):
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])

    feature = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)
    
    return feature

def hac(features):

    def distance(cluster_list_one, cluster_list_two):
        distance_ref = -1
        for point1 in cluster_list_one:
            for point2 in cluster_list_two:
                eudist = np.linalg.norm(point1 - point2)
                if eudist > distance_ref:
                    distance_ref = eudist

        return distance_ref

    list_of_clusters = []
    Z = []
    n = len(features)

    for i in range(n):
        group = [i, [features[i]]]
        list_of_clusters.append(group)

    cluster_index = n

    while len(list_of_clusters) >= 2:
        new_size = -1
        closest_distance = np.inf
        cluster_index1 = -1
        cluster_index2 = -1
        
        for i in range(len(list_of_clusters)):
            for j in range(i + 1, len(list_of_clusters)):
                thedist = distance(list_of_clusters[i][1], list_of_clusters[j][1])
                if thedist < closest_distance:
                    cluster_index1 = list_of_clusters[i][0]
                    cluster_index2 = list_of_clusters[j][0]
                    closest_distance = thedist
                elif thedist == closest_distance:
                    if list_of_clusters[i][0] < cluster_index1:
                        cluster_index1 = list_of_clusters[i][0]
                        cluster_index2 = list_of_clusters[j][0]
                        closest_distance = thedist

        one_cluster = None
        two_cluster = None

        for group in list_of_clusters:
            if group[0] == cluster_index1:
                one_cluster = group[1]
            elif group[0] == cluster_index2:
                two_cluster = group[1]

        merged_cluster = one_cluster + two_cluster
        group_new = [cluster_index, merged_cluster]
        list_of_clusters.append(group_new)
        cluster_index += 1

        new_size = len(merged_cluster)

        interpretation = [cluster_index1, cluster_index2, closest_distance, new_size]
        Z.append(interpretation)

        list_of_clusters = [filter_cluster for filter_cluster in list_of_clusters if filter_cluster[0] != cluster_index1 and filter_cluster[0] != cluster_index2]

        Z_array = np.array(Z, dtype=np.float64)

    return Z_array

def fig_hac(Z, names):
    fig = plt.figure()

    dendrogram(Z, labels=names, leaf_rotation=90)

    plt.tight_layout()

    return fig

def normalize_features(features):
    array_features = np.array(features)
    
    features_means = np.mean(array_features, axis=0)
    feature_standard_deviations = np.std(array_features, axis=0)

    normalized_array = (array_features - features_means) / feature_standard_deviations

    normalized_feature_list = [np.array(vector) for vector in normalized_array]

    return normalized_feature_list