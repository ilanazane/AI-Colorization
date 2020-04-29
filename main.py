from kmeans import *
from nn import *
import cv2
import matplotlib.pyplot as plt



#import image
img = cv2.imread('kanye.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#divide image into left and right halves
half = int(len(img[0])/2)
left = img[:,:half]
right = img[:,half:]


#Basic Agent




#Improved Agent - Neural Network
rightgray,rightRGB=toGrey(right)
weight1,weight2=training_data(left)
rightRGB=use_model(weight1,weight2,rightgray,rightRGB)

plt.imshow(rightRGB)
plt.show()

#Analysis??
