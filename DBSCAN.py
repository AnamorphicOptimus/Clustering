#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dbscan
"""
import math as m
import numpy as np
import matplotlib.pyplot as plt
import queue

NOISE = 0
UNASSIGNED = -1

def dist1(a, b):
    d = np.sqrt(np.sum(np.power(a-b,2),axis=1)).sum()
    return d

def dist(a, b):
    """distance"""
    return m.sqrt(np.power(a-b, 2).sum())

def neighbor_points(data, pointId, radius):
    """
    Get the Id of all sample points in the neighborhood
    :param data: sample point
    :param pointId: core point
    :param radius: radius
    :return: Id of the sample used in the neighborhood
    """
    points = []
    for i in range(len(data)):
#        print(dist(data[i, 0: 2], data[pointId, 0: 2]))
        if dist(data[i, 0: 2], data[pointId, 0: 2]) < radius:
            points.append(i)
    return np.asarray(points)

def to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
    """
    Determine whether a point is a core point, and if so, assign it and the unallocated sample points in its neighborhood to a new class
    If there are other core points in the neighborhood, repeat the previous step, but only process the unallocated points in the neighborhood, and they are still the class of the previous step.
    :param data: sample collection
    :param clusterRes: clustering result
    :param pointId: sample Id
    :param clusterId: Class Id
    :param radius: radius
    :param minPts: minimum local density
    :return: return whether the PointId can be assigned to a class
    """
    points = neighbor_points(data, pointId, radius)
    points = points.tolist()

    q = queue.Queue()

    if len(points) < minPts:
        clusterRes[pointId] = NOISE
        return False
    else:
        clusterRes[pointId] = clusterId
    for point in points:
        if clusterRes[point] == UNASSIGNED:
            q.put(point)
            clusterRes[point] = clusterId

    while not q.empty():
        neighborRes = neighbor_points(data, q.get(), radius)
        if len(neighborRes) >= minPts:                     
            for i in range(len(neighborRes)):
                resultPoint = neighborRes[i]
                if clusterRes[resultPoint] == UNASSIGNED:
                    q.put(resultPoint)
                    clusterRes[resultPoint] = clusterId
                elif clusterRes[clusterId] == NOISE:
                    clusterRes[resultPoint] = clusterId
    return True

def dbscan(data, radius, minPts):
    """
    Scan the entire data set, label each data set with core points, boundary points and noise points at the same time as
    Sample set clustering
    :param data: sample set
    :param radius: radius
    :param minPts: minimum local density
    :return: return clustering result, class id collection
    """
    clusterId = 1
    nPoints = len(data)
    clusterRes = [UNASSIGNED] * nPoints
    for pointId in range(nPoints):
        if clusterRes[pointId] == UNASSIGNED:
            if to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
                clusterId = clusterId + 1
    return np.asarray(clusterRes), clusterId

def cal_SSE(data,group):
    """Use SSE to measure clustering"""
    k = len(set(group))
    SSE = []
    for i in range(k):
        idx = np.where(group==i)
        meanClu = np.mean(data[idx],axis=0)
        clu_dis = dist1(data[idx],meanClu)
        SSE.append(clu_dis)

    return sum(SSE)

def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['red', 'blue', 'green', 'purple', 'black', 'yellow', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')
        
if __name__ == "__main__":
    
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
    # Data processing dataset is a list of 30 samples (density, sugar content)
    a = data.split(',')
    radius=0.09;minPts=3;
    dataset = np.asarray([(float(a[i]), float(a[i+1])) for i in range(1, len(a)-1, 3)])
    clusterRes, clusterNum = dbscan(dataset,radius,minPts)
    plotRes(dataset, clusterRes, clusterNum)
    print("radius:",radius," ","minPts:",minPts)
    print("DBSCAN SSE:",cal_SSE(dataset,clusterRes))
    plt.show()



