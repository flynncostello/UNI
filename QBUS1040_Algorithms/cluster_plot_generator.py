import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

### FINAL CLUSTERING FUNCTION ALGORITHM ###
def create_cluster(filepath, k, num_reforms=math.inf):
    """
    k = num centriods
    num_reforms = num times centriods and data points will change
    (i.e., become more accurate decreasing the clustering objective)
    """
    data = pd.read_csv(filepath)
    x = np.array(data['x'])
    y = np.array(data['y'])
    data_points = np.array([[x[i], y[i]] for i in range(len(x))])

    np.random.seed(0) # Always get the same sequence of random numbers by setting seed as same start value then python uses algorithm to get next random numbers in same order
    zx = np.random.rand(k)
    zy = np.random.rand(k)
    
    n = len(data_points)
    c = np.zeros(n) # c[2] = 0, means the third data point is in group 0 (first group)

    ### REFORMING CENTRIOD AND DATA POINT LOCATION num_reforms times ###
    h = 0
    while h < num_reforms:
        c_prev = c.copy() # Used to check that c has changed over the course of data point and centriod changes
        # Need to copy as otherwise the reference to c is overwritten and both become same

        ### PLACING DATA POINTS IN GROUPS ###
        # Placing data points in groups (around specific centriod)
        for i in range(len(data_points)):
            cur_point = data_points[i]
            distances = []
            for j in range(k):
                centriod = np.array([zx[j], zy[j]])
                dist_to_centriod = np.linalg.norm(centriod - cur_point)
                distances.append(dist_to_centriod)
            
            min_dist = min(distances)
            min_dist_index = distances.index(min_dist)
            c[i] = min_dist_index

        ### UPDATING CENTRIOD LOCATION ###
        # Updating centriod position in each group, zx and zy for each index needs to be changed
        # to the average x and y coord for all points in that group
        cur_group_num = 0
        while cur_group_num < k:
            # Loop through each group, finds all points in that group, sums them in terms of
            # x and y coords, then divides each total by num points in group giving new zx and zy values for the centriod
            group_points_x_sum = 0
            group_points_y_sum = 0
            num_points_in_group = 0
            for i in range(len(c)):
                if c[i] == cur_group_num:
                    group_points_x_sum += x[i]
                    group_points_y_sum += y[i]
                    num_points_in_group += 1
            
            avg_group_x = group_points_x_sum / num_points_in_group
            avg_group_y = group_points_y_sum / num_points_in_group

            zx[cur_group_num] = avg_group_x
            zy[cur_group_num] = avg_group_y
            
            cur_group_num += 1

        if np.equal(c_prev, c).all():
            break


    ### PLOTTING GRAPH ###
    plt.scatter(x, y, label='Data', cmap='rainbow', c=c)

    plt.scatter(zx, zy, s=100, color='gold', edgecolors='black', label='Centroids')

    plt.grid()
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('K-means clustering')
    plt.legend(loc='upper left')
    plt.savefig('plot')

create_cluster('./data.csv', 3)


