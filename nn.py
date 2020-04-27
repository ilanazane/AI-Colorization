import numpy as np
from kmeans import *

#initialize 2 weight matrices that are randomly generated
def random_weights():
	weight1 = np.random.rand(5,9)
	weight2 = np.random.rand(3,5)
	return weight1, weight2

def sigmoid(arr):
	result = 1/(1 + np.exp(-arr))
	return result

def sigmoid_prime(arr):
	result = np.exp(-arr)/np.square(1 + np.exp(-arr))
	return result

#input_layer is 1D array
def calc_layers(weight1, weight2, input_layer):
	hidden_layer = sigmoid(np.dot(weight1,input_layer))
	#hidden_layer = sigmoid(hidden_layer)
	output_layer = sigmoid(np.dot(weight2,hidden_layer))
	#output_layer = sigmoid(output_layer)
	return hidden_layer,output_layer



#calculate the cost
def calc_cost(output_layer, y_layer):
	sum = 0
	for a in range(len(output_layer)):
		sum += np.square(output_layer[a] - y_layer[a])
	return sum

#calculate the derivative of cost between output + hidden layers
#should get output of [3x5]
def calc_cost_deriv_1(hidden_layer, output_layer, y_layer, weight2):
	term1 = sigmoid_prime(np.dot(weight2,hidden_layer))
	adj1 = np.dot(term1,np.transpose(hidden_layer))
	for j in range(3):
		scalar = 2*(output_layer[j]-y_layer[j])
		adj1[j] *= scalar
	return adj1



#calculate the derivative of cost between hidden + input layers
#should get output of [5x9]
def calc_cost_deriv_2(input_layer, hidden_layer, y_layer, weight2, weight1):
	term1 = sigmoid_prime(np.dot(weight2,hidden_layer))
	term2 = sigmoid_prime(np.dot(weight1,input_layer))
	adj2 = np.dot(term1,np.transpose(hidden_layer))
	for j in range(3):
		scalar = 2*(output_layer[j]-y_layer[j])
		adj2[j] *= scalar
	return adj2






def training_data():
	img = cv2.imread('painting.jpg')
	#img = cv2.imread('test2.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	half = int(len(img[0])/2)
	left = img[:,:half]

	toGrey(left)
	
	patches=[]
	#iterate through grayleft
	#iterate through rows
	for i in range(1,len(grayleft)-1):
    	#iterate through columns
    	for j in range(1,len(grayleft[0])-1):
        	#grayleft[i][j] starts on middle pixel
        	#find the rest of the patch (adjacent pixels)
        	patches.append((grayleft[i-1:i+2,j-1:j+2],(i,j)))


def train_model(input_data,)






#grayleftPatch = return_grey()
weight1, weight2 = random_weights()
input_data = grayleftPatch[1][1]
y_layer = img[1][1] #feed in image as some sort of input somehow somewhere someday not today
