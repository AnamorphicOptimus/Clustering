#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGNES
"""
import math
import numpy as np
import pylab as pl

def dist(a,b):
    s = math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))
    return s

def dist_avg(Ci, Cj):
    return sum(dist(i, j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

def standard(data):
    """Data standardization"""
    mean = np.mean(data,axis=0)
    std = np.std(data, axis=0)
    standard_data = (data - mean)/std
    return standard_data 

def find_Min(M):
    min = 1000
    x = 0; y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j];x = i; y = j
    return (x, y, min)

def AGNES_SSE(cluRe):
    s=[]
    for clu in cluRe:
        tclu = np.asarray(clu)
        cluMean = np.mean(tclu,axis=0)
        s0=0
        for data in tclu:
            s0+=math.sqrt(sum(map(lambda x:x**2,data-cluMean)))
        s.append(s0)
    return sum(s)

def AGNES(dataset, dist, k):
    C = [];M = []
    for i in dataset:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist(i, j))
        M.append(Mi)
    q = len(dataset)
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)
        q -= 1
    return C

def draw(C):
    colValue = ['r', 'b', 'g', 'y', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []   
        coo_Y = []   
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()
    
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
    k=3
    dataset = [(float(a[i]), float(a[i+1])) for i in range(1, len(a)-1, 3)]

    cluRe = AGNES(dataset,dist=dist_avg,k=k)
    print("K=:",k)
    print("AGNES SSE:",AGNES_SSE(cluRe))
    draw(cluRe)
    
    
    