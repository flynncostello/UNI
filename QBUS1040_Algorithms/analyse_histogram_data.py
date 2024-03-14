import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

histograms = pd.read_csv('./article_histograms.csv').to_numpy()
dictionary = pd.read_csv('./dictionary.csv').to_numpy()
titles = pd.read_csv('./article_titles.csv').to_numpy()

def create_cluster(histograms, dictionary, titles, k, num_reforms=math.inf):
    np.random.seed(43)
    z = np.random.rand(k, len(dictionary)) * np.mean(histograms) # times by mean of histograms to make centriods in same scale as data

    N = len(histograms)
    c = np.zeros(N)

    h = 0
    while h < num_reforms:
        c_prev = c.copy()
        ### PLACING DATA POINTS IN GROUPS ###
        for i in range(N):
            point = histograms[i]
            distances = np.zeros(k) # Stores distances of a certain point to all k centriods

            for j in range(k):
                centriod = z[j]
                dist_to_centriod = np.linalg.norm(centriod - point)
                distances[j] = dist_to_centriod

            min_dist_index = np.argmin(distances)
            c[i] = min_dist_index # Updating that points centriod group to the centriod index its closes to, for k = 3, we have centriod numbers 0, 1, 2
        
        ### RECALCULATING CENTRIODS BASED ON DATA POINT LOCATIONS IN ITS GROUP ###
        for i in range(k): # Looping through all clusters
            # Looking at centriod group i
            cur_centriod = z[i]
            data_points_in_this_group = histograms[c == i]
    
            if len(data_points_in_this_group) > 0:  # Check if any data points are assigned to this centroid
                z[i] = np.mean(data_points_in_this_group, axis=0) # axis=0 means do this across the columns

        if np.equal(c_prev, c).all():
            break

    ### PRINTING EACH TOPIC AND THE TITLES WITHIN EACH TOPIC ###
    for i in range(k):
        print(f"=== Topic {i} ===")
        centriod_titles = titles[c == i]

        for title in centriod_titles:
            print(title[0])





create_cluster(histograms, dictionary, titles, 3, 100)


