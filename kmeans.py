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
import time

#used to quickly locate the cluster information of a given pixel
@dataclass
class clu():
    r:int
    g:int
    b:int
    cluster:int


#calculate the euclidean distance
def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

#turn the image format to [r,g,b,cluster] for later convenience
def cluArr(imgarr):
    cluArray = np.empty((len(imgarr), len(imgarr[0])), dtype=object)
    for i in range (len(imgarr)):
        for j in range (len(imgarr[0])):
            temp= clu(imgarr[i][j][0],imgarr[i][j][1], imgarr[i][j][2], -1)
            cluArray[i][j] = temp
    return (cluArray)

#the kmeans algorithm to find the centroids of our image data
def kmeans(imgarr):
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


    return centroids, cluArray

#recolor the left image in terms of representative colors
def recolorLeft(left, centroids, cluArray):
    for y in range(0, len(left)):
        for x in range(0, len(left[0])):
            left[y][x] = centroids[cluArray[y][x].cluster]

    return left

#a method to turn an image to greyscale
def toGrey(image):
    #recolor each pixel
    copy = np.copy(image)
    image = image.tolist()
    for i in range(0,len(image)):
        for j in range(0, len(image[i])):
            copy[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
            image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]

    return np.array(image), copy

#get all of the 3x3 patches in an image
def get_patches(img):
    patches=[]
    #iterate through grayleft
    #iterate through rows
    for i in range(1,len(img)-1):
        #iterate through columns
        for j in range(1,len(img[0])-1):
            #grayleft[i][j] starts on middle pixel
            #find the rest of the patch (adjacent pixels)
            patches.append((img[i-1:i+2,j-1:j+2],(i,j)))

    return patches

#recolor the right side by finding 6 most similar in testing data
def recolor_right(right,left):
    centroids, cluArray = kmeans(left)
    #FINAL OUTPUT FOR LEFT (representative colors)
    final_left = np.copy(recolorLeft(left, centroids, cluArray))

    grayleft = toGrey(left)[0]

    grayright, copy= toGrey(right)

    #plt.imshow(grayleft)
    #plt.show()

    grayleftPatch=get_patches(grayleft)

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
            samples = random.sample(list(grayleftPatch), 1000)

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
                copy[i][j]=centroids[mostFrequent]
            except:
                x=random.randint(0,len(sixPatches)-1)
                tie=sixPatches[x]
                copy[i][j]=centroids[tie]

            tracker += 1
        print(tracker/(len(grayright)*len(grayright[0]))*100, "%")

    plt.imshow(final_left)
    plt.show()

    plt.imshow(copy)
    plt.show()
    return final_left, copy

#combine two pictures into one
def combinePic(final_left, copy):
    new = []
    for i in range(0, len(final_left)):
        new.append(list(final_left[i])+list(copy[i]))

    plt.imshow(new)
    plt.show()
    return new
