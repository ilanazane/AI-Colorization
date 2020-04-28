from kmeans import *
from nn import *



#import image
img = cv2.imread('painting.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#divide image into left and right halves
half = int(len(img[0])/2)
left = img[:,:half]
right = img[:,half:]


#Basic Agent




#Improved Agent - Neural Network




#Analysis??