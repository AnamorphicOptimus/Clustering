#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmeans
"""
# 初始数据点的选取
# 将数据点归入最近质点
# 更新质点
# 判断迭代是否停止

import random
import math
import numpy as np
import matplotlib.pyplot as plt

def creaCent(data,k):
    """Generate random initial cluster centers"""

    Cen = random.sample(data[:,:].tolist(),k=k)
    return np.asarray(Cen)

def standard(data):
    """Data standardization"""
    mean = np.mean(data,axis=0)
    std = np.std(data, axis=0)
    standard_data = (data - mean)/std
    return standard_data 


def cal_dis(data,Cen,dis="Euclidean"):
    """Calculate distance matrix (mass points and data points)"""
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(len(Cen)):
            dd = math.sqrt(sum(map(lambda x:x**2,data[i]-Cen[j])))
            dis[i].append(dd)
           
    return dis


def grouping(data,dis):
    """Sort data points to the nearest centroid"""
    cluRe = []
    for i in range(len(data)):
        cluRe.append(np.argsort(dis[i])[0])
    
    return np.asarray(cluRe)


def new_Cen_Cal(data,group,k):
    """
    :param group: grouped samples
    :return: take the average and calculate the new centroid
    
    """    
    newclu = []
    for i in range(k):
        idx = np.where(group==i)
        dotMean = data[idx].sum(axis=0)/len(data[idx])
        newclu.append(dotMean)
     
    return np.asarray(newclu)

def classfy(data, clu, k):
    """
    Iterative convergence update centroid
    :param data: sample collection
    :param clu: centroid collection
    :param k: number of categories
    :return: error, new centroid
    """
    clulist = cal_dis(data, clu)
    clusterRes = grouping(data, clulist)
    clunew =  new_Cen_Cal(data, clusterRes, k)
    err = clunew - clu

    return err, clunew, k, clusterRes

def cal_SSE(data,clu,group,k):
    """Use SSE to measure clustering"""
    SSE = []
    for i in range(k):
        idx = np.where(group==i)
        tmpclu = np.asarray([clu[i].tolist()])
        clu_dis = cal_dis(data[idx],tmpclu)
        SSE.append(sum([clu_dis[j][0] for j in range(len(clu_dis))]))

    return sum(SSE)

def Kmeans(data,k,max_iter,creaCenType ="random",dis="Euclidean"):
    """
    :param k: number of clusters
    :param dis: distance calculation method, default Euclidean distance
    :param creaCent: initial point selection method
    :param max_iter: Maximum iteration method
    
    """
    print("K:",k)
    Cen = creaCent(data,k)
    
    # Iterate until the centroid converges
    iter_cnt = 0
    err, clunew,k,clusterRes = classfy(data, Cen, k)
    while iter_cnt<max_iter and np.any(abs(err))>0:
        print("iter_cnt:",iter_cnt)
        err, clunew, k, clusterRes = classfy(data, clunew, k)
        iter_cnt += 1

    disMa = cal_dis(data,clunew)
    clusterResult = grouping(data,disMa)
    
    return clunew, clusterResult

def plotRes(data, clusterRes, clusterNum):
    """
    Visualization of results
    :param data: sample set
    :param clusterRes: clustering result
    :param clusterNum: number of classes
    :return:
    """
    nPoints = len(data)
    scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')

    plt.show()

if __name__ == "__main__":
    
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
    a = data.split(',')
    k=1
    dataset = np.asarray([(float(a[i]), float(a[i+1])) for i in range(1, len(a)-1, 3)])
    clunew, cluRe = Kmeans(dataset,k,max_iter=500)
    print("Kmeans SSE:",cal_SSE(dataset,clunew,cluRe,k))
    plotRes(dataset,cluRe,k)



    


