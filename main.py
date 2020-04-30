from kmeans import *
from nn import *
import cv2
import matplotlib.pyplot as plt



#import image
img = cv2.imread('painting.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#divide image into left and right halves
half = int(len(img[0])/2)
left = img[:,:half]
right = img[:,half:]

# #display image
# plt.imshow(img)
# plt.show()



# #Basic Agent
# final_left, copy = recolor_right(right,left)
# final_pic_basic = combinePic(final_left, copy)


#Improved Agent - Neural Network
rightgray,rightRGB=toGrey(right)
weight1,weight2=training_data(left)
rightRGB=use_model(weight1,weight2,rightgray,rightRGB)

plt.imshow(rightRGB)
plt.show()


#Analysis??
