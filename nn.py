import numpy as np
from kmeans import *

#initialize 2 weight matrices that are randomly generated
def random_weights():
	weight1 = np.random.rand(5,9)
	weight2 = np.random.rand(3,5)
	return weight1, weight2

def sigmoid(arr):
	result = []
	for x in arr:
		result.append(1/(1 + np.exp(-x))) 
	return result

def sigmoid_prime(arr):
	result = []
	for x in arr:
		result.append(np.exp(-x)/np.square(1 + np.exp(-x)))
	return result


def train_model(weight1, weight2, input_layer):
	hidden_layer = np.dot(weight1,input_layer)
	hidden_layer = sigmoid(hidden_layer)
	output_layer = np.dot(weight2,hidden_layer)
	output_layer = sigmoid(output_layer)


#calculate the cost
def calc_cost(output_layer, y_layer):
	sum = 0
	for a in range(length(output_layer)):
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









#grayleftPatch = return_grey()
weight1, weight2 = random_weights()
input_layer = grayleftPatch[1][1]
y_layer = img[1][1] #feed in image as some sort of input somehow somewhere someday not today

