from kmeans import *
from nn import *
import cv2
import matplotlib.pyplot as plt
import math
from random import randint
import numpy as np
import copy

#import image



print("Please enter one of the options: ")
print("'original[o]', kmeans[k]', 'nn[n]', 'both[b]' or 'quit[q]' ")
option = str(input())

while option != 'q':
    #display original image
    img = cv2.imread('painting.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #divide image into left and right halves
    half = int(len(img[0])/2)
    left = img[:,:half]
    right = img[:,half:]




    if option == 'o':
        plt.imshow(img)
        plt.show()

    #Basic Agent
    elif option == 'k':
        rightcpy_k= copy.deepcopy(right)
        leftcpy_k= copy.deepcopy(left)
        final_leftk, copyk = recolor_right(rightcpy_k,leftcpy_k)
        final_pic_basick = combinePic(final_leftk, copyk)

    #Improved Agent - Neural Network
    elif option == 'n':
        rightcpy_n= copy.deepcopy(right)
        leftcpy_n= copy.deepcopy(left)
        rightgray_n,rightRGB_n=toGrey(rightcpy_n)
        weight1_n,weight2_n=training_data(leftcpy_n)
        rightRGB_n=use_model(weight1_n,weight2_n,rightgray_n,rightRGB_n)
        final_pic_nn = combinePic(left, rightRGB_n)

    #display both basic agent and improved agent
    elif option =='b':
        rightcpyb= copy.deepcopy(right)
        leftcpyb= copy.deepcopy(left)
        final_leftb, copyb = recolor_right(rightcpyb,leftcpyb)
        final_pic_basicb = combinePic(final_leftb, copyb)

        rightgray_b,rightRGB_b=toGrey(rightcpyb)
        weight1_b,weight2_b=training_data(leftcpyb)
        rightRGB_b= use_model(weight1_b,weight2_b,rightgray_b,rightRGB_b)
        final_pic_nnb = combinePic(left, rightRGB_b)

    print("Please enter one of the options: ")
    print("'original[o]', kmeans[k]', 'nn[n]', 'both[b]' or 'quit[q]' ")
    option = str(input())



# #Analysis!!

# #run basic agent
# final_left, final_right = recolor_right(right,left)
# final_pic_basic = combinePic(final_left, final_right)

# #run advanced agent
# rightgray,rightRGB=toGrey(right)
# weight1,weight2=training_data(left)
# rightRGB=use_model(weight1,weight2,rightgray,rightRGB)
# final_pic_nn = combinePic(left, rightRGB)

# #save the right side images
# final_right = cv2.cvtColor(final_right, cv2.COLOR_RGB2BGR)
# cv2.imwrite('basic_right.jpg', final_right)
# rightRGB = cv2.cvtColor(rightRGB, cv2.COLOR_RGB2BGR)
# cv2.imwrite('advanced_right.jpg', rightRGB)

# #load in images
# final_right = cv2.imread('basic_right.jpg')
# # final_right = cv2.cvtColor(final_right, cv2.COLOR_RGB2BGR)
# final_right = cv2.cvtColor(final_right, cv2.COLOR_BGR2RGB)
# rightRGB = cv2.imread('advanced_right.jpg')
# # rightRGB = cv2.cvtColor(rightRGB, cv2.COLOR_RGB2BGR)
# rightRGB = cv2.cvtColor(rightRGB, cv2.COLOR_BGR2RGB)


# #compare 2 images
# def compare_img(img1,img2):
#     differs = []
#     for i in range(50000):
#         dif = 0.0
#         i,j = randint(0,len(right)-1), randint(0,len(right[0])-1)
#         for k in range(3):
#             dif += math.pow(int(img1[i][j][k]) - int(img2[i][j][k]), 2.0)
#         avg = math.sqrt(dif)
#         differs.append(avg)
#     differs = np.array(differs)
#     return differs

# #compare basic to original
# b2o = compare_img(final_right,right)
# b2o_mean = np.mean(b2o)
# #compare advanced to original
# a2o = compare_img(rightRGB,right)
# a2o_mean = np.mean(a2o)
# #compare basic to advanced
# b2a = compare_img(final_right,rightRGB)
# b2a_mean = np.mean(b2a)

# # x = np.linspace(0,len(right)*len(right))
# x = np.linspace(0,50000,50000)
# print("x", len(x))
# #plot data in scatterplot
# plt.scatter(x,b2o,color='magenta', label = 'basic-original, mean=' + str(round(b2o_mean,2)))
# plt.scatter(x,a2o,color='cyan', label = 'advanced-original, mean=' + str(round(a2o_mean,2)))
# # plt.scatter(x,b2a,color='g', label = 'basic-advanced')
# plt.title('Comparison between images')
# plt.xlabel('Pixel')
# plt.ylabel('Distance')
# plt.legend()
# plt.savefig('comparison_all.png')
# plt.show()

# plt.scatter(x,b2o,color='blueviolet', label = 'basic-original, mean=' + str(round(b2o_mean,2)))
# plt.title('Comparison between images')
# plt.xlabel('Pixel')
# plt.ylabel('Distance')
# plt.legend()
# plt.savefig('comparison2.png')
# plt.show()
