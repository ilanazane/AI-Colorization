#!/usr/bin/python
import time
import numpy as np
import random
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
from statistics import mode

@dataclass
class clu():
    r:int
    g:int
    b:int
    cluster:int


img = cv2.imread('painting.jpg')
#img = cv2.imread('test2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

half = int(len(img[0])/2)
left = img[:,:half]
greyleft = np.copy(left)
right = img[:,half:]


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
        random_y = random.randint(0, len(imgarr) - 1)
        random_x = random.randint(0, len(imgarr[0]) - 1)
        centroids.append(list(imgarr[random_y][random_x]))

    ctr=0

    cluArray= cluArr(imgarr)
    while (ctr!=15):
        sumarr=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        counter=[0,0,0,0,0]
        ctr=0

        # goes through every pixel
        for y in range (len(imgarr)):
            for x in range (len(imgarr[0])):
                #print("x", x, "y", y)
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
                if abs(avg-centroids[k][l]) > 5:
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

    return centroids, cluArray


def recolorLeft(left, centroids, cluArray):
    for y in range(0, len(left)):
        for x in range(0, len(left[0])):
            left[y][x] = centroids[cluArray[y][x].cluster]

    return left

def toGrey(image):
    #recolor each pixel
    for i in range(0,len(image)):
        for j in range(0, len(image[i])):
            image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]

    return image



centroids, cluArray = kmeans(left)
#FINAL OUTPUT FOR LEFT (representative colors)
final_left = np.copy(recolorLeft(left, centroids, cluArray))










#RECLOR RIGHT
plt.imshow(greyleft)
plt.show()

grayleft = toGrey(greyleft)
grayright= toGrey(right)

plt.imshow(grayleft)
plt.show()

grayleftPatch=[]

#iterate through grayleft
#iterate through rows
for i in range(1,len(grayleft)-1):
    #iterate through columns
    for j in range(1,len(grayleft[0])-1):
        #grayleft[i][j] starts on middle pixel
        #find the rest of the patch (adjacent pixels)
        grayleftPatch.append((grayleft[i-1:i+2,j-1:j+2],(i,j)))


tracker = 0

#iterate through testing
#iterate through rows
for i in range(1,len(grayright)-1):
    #iterate through columns
    for j in range(1,len(grayright[0])-1):
        patch=grayright[i-1:i+2,j-1:j+2]
        min1,min2,min3,min4,min5,min6=1000,1000,1000,1000,1000,1000
        sixPatches=[[],[], [], [], [], []]
        #find six patches

        #take a sample from the total training data to compare with test data
        #the higher the number the better the resulting image quality
        samples = random.sample(list(grayleftPatch), 300)

        for k in samples:
            dist=euclidDist(k[0],grayright[i-1:i+2,j-1:j+2])
            if dist<min1:
                min1=dist
                sixPatches[1]=sixPatches[0]
                sixPatches[0]=k[1]
                continue
            if dist<min2:
                min2=dist
                sixPatches[2]=sixPatches[1]
                sixPatches[1]=k[1]
                continue
            if dist<min3:
                min3=dist
                sixPatches[3]=sixPatches[2]
                sixPatches[2]=k[1]
                continue
            if dist<min4:
                min4=dist
                sixPatches[4]=sixPatches[3]
                sixPatches[3]=k[1]
                continue
            if dist<min5:
                min5=dist
                sixPatches[5]=sixPatches[4]
                sixPatches[4]=k[1]
                continue
            if dist<min6:
                min6=dist
                sixPatches[5]=k[1]
                continue

        #get color of 6 middel pixels
        for l in range(0,len(sixPatches)):
            x=sixPatches[l][1]
            y=sixPatches[l][0]

            #replace the patches/coordinates we got with the colors they represent
            sixPatches[l] = cluArray[y][x].cluster

        try:
            mostFrequent=mode(sixPatches)
            grayright[i][j]=centroids[mostFrequent]
        except:
            x=random.randint(0,len(sixPatches)-1)
            tie=sixPatches[x]
            grayright[i][j]=centroids[tie]

        tracker += 1
    print(tracker/(len(grayright)*len(grayright[0]))*100)



plt.imshow(final_left)
plt.show()

plt.imshow(grayright)
plt.show()

new = []
for i in range(0, len(final_left)):
    new.append(list(final_left[i])+list(grayright[i]))

plt.imshow(new)
plt.show()
