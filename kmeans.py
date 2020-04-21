#!/usr/bin/python
import time
import numpy as np
import random
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
@dataclass
class clu():
    r:int
    g:int
    b:int
    cluster:int


img = cv2.imread('kanye.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


half = int(len(img[0])/2)
left = img[:,:half]
right = img[:,half:]
plt.imshow(left)
plt.show()



def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

def cluArr(imgarr):
    cluArray = np.empty((len(imgarr), len(imgarr[0])), dtype=object)
    for i in range (len(imgarr)):
        for j in range (len(imgarr[0])):
            temp= clu(imgarr[i][j][0],imgarr[i][j][1], imgarr[i][j][2], -1)
            cluArray[i][j] = temp
    return (cluArray)

def rgb2hex(item):
    return "#{:02x}{:02x}{:02x}".format(item[0],item[1],item[2])

def kmeans (imgarr):

    centroids=[]
    #generates 5 random points
    for i in range(5):
        temp = [random.randint(0,255),random.randint(0,255), random.randint(0,255)]
        centroids.append(temp)

    ctr=0

    cluArray= cluArr(imgarr)
    while (ctr!=15):
        sumarr=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        counter=[0,0,0,0,0]
        ctr=0

        # goes through every pixel
        for y in range (len(imgarr)):
            for x in range (len(imgarr[0])):
                print("x", x, "y", y)
                shortestDist=3000
                cluster=-1

                for j in range(5):
                    # finding the closest centroid
                    newDist = euclidDist(imgarr[y][x], centroids[j])
                    if newDist < shortestDist:
                        shortestDist=newDist
                        cluster=j

                #tag the pixel
                #print(item[0],item[1])
                #pprint(imgarr)
                #print(len(imgarr),len(imgarr[0]) )
                cluArray[y][x].cluster= cluster
                #add to the array of sums
                sumarr[cluster][0] +=cluArray[y][x].r
                sumarr[cluster][1] +=cluArray[y][x].g
                sumarr[cluster][2] +=cluArray[y][x].b
                counter[cluster]+= 1
        #finds the average
        for k in range(5):
            for l in range (3):
                if(counter[k]!= 0):
                    avg= int(sumarr[k][l]/counter[k])
                else:
                    avg= 0
                if abs(avg-centroids[k][l]) > 1:
                    centroids[k][l] = avg
                else:
                    ctr += 1
            #find new centroids
            print ("centroids      ",centroids)

    for m in range(5):
        k=rgb2hex(centroids[m])
        circle1 = plt.Circle((0, 0), 2.0, color= k)
        #plt.imshow(centroids)
        fig, ax = plt.subplots()
        plt.xlim(-1.25,1.25)
        plt.ylim(-1.25,1.25)
        ax.grid(False)
        ax.add_artist(circle1)
        plt.axis('off')
        plt.show()
    return(centroids)

kmeans(left)
