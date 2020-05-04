import numpy as np
from kmeans import *

#initialize 2 weight matrices that are randomly generated
def random_weights():
	weight1 = np.random.rand(5,9)
	weight2 = np.random.rand(3,5)
	return weight1, weight2

#calculate sigmoid function of each value in an array
def sigmoid(arr):
	result = 1/(1 + np.exp(-arr))
	return result

#calculate derivative of sigmoid function of each value in an array
def sigmoid_prime(arr):
	result = np.exp(-arr)/np.square(1 + np.exp(-arr))
	return result

#input_layer is 1D array
def calc_layers(weight1, weight2, input_layer):
	hidden_layer = sigmoid(np.dot(weight1,np.transpose(input_layer)))
	output_layer = sigmoid(np.dot(weight2,hidden_layer))
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
	term1 = sigmoid_prime(np.dot(weight2,hidden_layer)) #calculate da/dw
	term2 = 2*np.diagflat(output_layer - np.transpose(np.array([y_layer])))#diagonalize the dC/da term
	product = np.dot(term2, term1)
	hidden_layer = np.transpose(hidden_layer) #this is a 1x5 [a,b,c,d,e]
	adj1 = np.dot(product,hidden_layer) #this is a 3x5 - adjusts weight2 array

	# print("ADJ1",adj1)
	return adj1



#calculate the derivative of cost between hidden + input layers
#should get output of [5x9]
#dC/dw = dC/da(output) * da(output)/da(hidden) * da(hidden)/dw
def calc_cost_deriv_2(input_layer, hidden_layer, output_layer, y_layer, weight2, weight1):
	term1 = sigmoid_prime(np.dot(weight1,np.transpose(input_layer))) #da/dw term: this makes a 5x1 matrix
	product = np.dot(term1, input_layer) #da/dw w chain rule: this makes a 5x9

	#create a 5x1 matrix: dC/da(output) * da(output)/da(hidden)
	sum_term = np.array([0.0,0.0,0.0,0.0,0.0])
	for j in range(3):
		scalar = 2 * (output_layer[j] - y_layer[j]) * sigmoid_prime(np.dot(weight2[j],hidden_layer))
		sum_term += (scalar * weight2[j])

	sum_term = np.diagflat(sum_term) #make sum a 5x5 diagonal matrix
	adj2 = np.dot(sum_term,product) #this is a 5x9 adjustment array - adjusts weight1

	# print("ADJ2", adj2)
	return adj2



#take in one patch of 3x3 input data - forward and back propagates
def train_model(input_data,y_layer, weight1, weight2):
	#covert to 1x9
	y_layer=y_layer/255
	input_data=np.array([input_data.flatten()])/255
	#forward calculation
	hidden_layer,output_layer=calc_layers(weight1,weight2,input_data)
	#find cost
	cost = calc_cost(output_layer,y_layer)
	#easy weight derivatives(closest to output)
	weight2_derivs = calc_cost_deriv_1(hidden_layer, output_layer, y_layer, weight2)
	#calcuate weight derivatives between input and hidden layers
	weight1_derivs = calc_cost_deriv_2(input_data, hidden_layer, output_layer, y_layer, weight2, weight1)

	return weight2_derivs, weight1_derivs




def training_data(left):
	leftcopy = np.copy(left)
	greyleft = toGrey(left)[0] #turns left image into greyscale
	patches = get_patches(greyleft) #gets all patches from left image
	np.random.shuffle(patches)
	weight1, weight2 = random_weights()

	#patches in format [[a,b,c],[d,e,f],[g,h,i],(i,j)]
	group = []
	numgroups = 200
	for x in range(numgroups):
		group.append(patches[int(x*len(patches)/numgroups):int(len(patches)/numgroups*(1+x)-1)])
	for subgroup in group:
		for j in range(100):
			adj1_total = np.zeros((3,5),dtype = float)
			adj2_total = np.zeros((5,9),dtype = float)
			for patch in subgroup:
				#train model
				a1, a2 = train_model(patch[0],leftcopy[patch[1]], weight1, weight2)
				adj1_total += a1
				adj2_total += a2
			adj1_avg = adj1_total / len(subgroup)
			adj2_avg = adj2_total / len(subgroup)
			weight1 = weight1 - adj2_avg
			weight2 = weight2 - adj1_avg
	return weight1, weight2


def use_model(weight1,weight2,rightgray,rightRGB):
	right_patches=get_patches(rightgray)
	for i in right_patches:
		x=i[0].flatten()/255
		output=calc_layers(weight1,weight2,x)[1]
		rightRGB[i[1]]=output*255
	return rightRGB



