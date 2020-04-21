#!/usr/bin/python
import time
import numpy as np
import random
import pygame
from pprint import pprint
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


half = int(len(img[0])/2)
left = img[:,:half]
right = img[:,half:]
plt.imshow(left)
plt.show()


def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)


def kmeans (imgarr):

    centroids=[]
    #generates 5 random points
    for i in range(5):
        temp = [random.randint(0,255),random.randint(0,255), random.randint(0,255)]
        centroids.append(temp)

    ctr=0
    avgarr= [0,0,0,0,0]
    while (ctr!=5):
        sumarr=[0,0,0,0,0]
        counter=[0,0,0,0,0]
        ctr=0
        # goes through every pixel
        for item in imgarr:
            shortestDist=3000
            cluster=-1
            for j in range(5):
                # finding the closest centroid
                newDist = euclidDist(item[2], centroids[j])
                if newDist < shortestDist:
                    shortestDist=newDist
                    cluster=j
            #tag the pixel
            #add to the array of sums
            sumarr[cluster] +=shortestDist
            counter[cluster]+= 1
        #finds the average
        for k in range(5):
            newavg= sumarr[k]/counter[k]
            if math.abs(newavg-avgarr[k]) > 1:
                avgarr[k] = newavg
            else:
                ctr += 1
            #find new centroids


    return(cluster)
