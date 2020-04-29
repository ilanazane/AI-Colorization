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
	hidden_layer = sigmoid(np.dot(weight1,np.transpose(input_layer)))
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
	term1 = sigmoid_prime(np.dot(weight2,hidden_layer)) #calculate da/dw
	term2 = 2*np.diagflat(output_layer - np.transpose(np.array([y_layer])))#diagonalize the dC/da term
	product = np.dot(term2, term1)
	hidden_layer = np.transpose(hidden_layer) #this is a 1x5 [a,b,c,d,e]
	adj1 = np.dot(product,hidden_layer) #this is a 3x5 - adjusts weight2 array

	# print("ADJ1",adj1)
	return adj1

# #TEST CALC COST DERIV
# hidden_layer = np.array([1,2,3,4,5])
# output_layer = np.array([1,3,5])
# y_layer = np.array([102,104,1006])
# weight2 = np.array([[0.5,0.5,0.5,0.5,0.5],[0.3,0.4,0.5,0.6,0.7],[0,0.25,0.5,0.75,1]])
# calc_cost_deriv_1(hidden_layer, output_layer, y_layer, weight2)
# #conclusion: she prob works



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

# #TEST CALC COST DERIV
# input_layer = np.array([1,2,3,4,5,6,7,8,9])
# hidden_layer = np.array([1,2,3,4,5])
# output_layer = np.array([1,3,5])
# y_layer = np.array([102,104,1006])
# weight2 = np.array([[0.5,0.5,0.5,0.5,0.5],[0.3,0.4,0.5,0.6,0.7],[0,0.25,0.5,0.75,1]])
# weight1 = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0,0.25,0.5,0.75,1,0.75,0.5,0.25,0],[1,0.75,0.5,0.25,0,0.25,0.5,0.75,1],[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]])
# calc_cost_deriv_2(input_layer, hidden_layer, output_layer, y_layer, weight2, weight1)
# #conclusion: she gucci




#take in one patch of 3x3 input data - forward and back propagates
def train_model(input_data,y_layer, weight1, weight2):
	#covert to 1x9
	y_layer=y_layer/255
	input_data=np.array([input_data.flatten()])/255
	#initialize random weights
	# weight1,weight2=random_weights()
	#forward calculation
	hidden_layer,output_layer=calc_layers(weight1,weight2,input_data)
	#find cost
	cost = calc_cost(output_layer,y_layer)
	# print("cost",cost)
	#easy weight derivatives(closest to output)
	weight2_derivs = calc_cost_deriv_1(hidden_layer, output_layer, y_layer, weight2)
	#calcuate weight derivatives between input and hidden layers
	weight1_derivs = calc_cost_deriv_2(input_data, hidden_layer, output_layer, y_layer, weight2, weight1)

	return weight2_derivs, weight1_derivs





def training_data(left):
	# img = cv2.imread('kanye.jpg')
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# half = int(len(img[0])/2)
	# left = img[:,:half]
	greyleft = toGrey(left)[0] #turns left image into greyscale
	patches = get_patches(greyleft) #gets all patches from left image
	weight1, weight2 = random_weights()
	# for i in range(500):
	# 	for item in patches:
	# 		adj1_total = np.zeros((3,5),dtype = float)
	# 		adj2_total = np.zeros((5,9),dtype = float)
	# 		#train model
	# 		a1, a2 = train_model(item[0],left[item[1]], weight1, weight2)
	# 		adj1_total += a1
	# 		adj2_total += a2
	# 	adj1_avg = adj1_total / len(patches)
	# 	adj2_avg = adj2_total / len(patches)
	# 	weight1 = weight1 - adj2_avg
	# 	weight2 = weight2 - adj1_avg

	#print("LOOKE HERE", patches[0])
	# #patches[[a,b,c],[d,e,f],[g,h,i],(i,j)]
	group = []
	# numgroups = 50
	numgroups = 50
	for x in range(numgroups):
		group.append(patches[int(x*len(patches)/numgroups):int(len(patches)/numgroups*(1+x)-1)])
	for subgroup in group:
		adj1_total = np.zeros((3,5),dtype = float)
		adj2_total = np.zeros((5,9),dtype = float)
		# for j in range(100):
		for j in range(75):
			for patch in subgroup:
				#train model
				a1, a2 = train_model(patch[0],img[patch[1]], weight1, weight2)
				adj1_total += a1
				adj2_total += a2
			adj1_avg = adj1_total / len(subgroup)
			adj2_avg = adj2_total / len(subgroup)
			weight1 = weight1 - adj2_avg
			weight2 = weight2 - adj1_avg
		print("uno",weight1)
		print("dos",weight2)
	return weight1, weight2


def use_model(weight1,weight2,rightgray,rightRGB):
	right_patches=get_patches(rightgray)
	for i in right_patches:
		x=i[0].flatten()/255
		output=calc_layers(weight1,weight2,x)[1]
		rightRGB[i[1]]=output*255
		# print(output*255)
	return rightRGB


#test_answer2, test_answer3=train_model(test_array,test_y)
# print(test_answer)
# print(test_answer2)
# print(test_answer3)




### NOTE: Do we need to normalize input nodes?? ##
## Another NOTE: We should multiply final r,g,b values by 255 ##
## Another Another NOTE: make sure input layer is 2D ##
##Another Another Another NOTE: shuffle patches data?



# weight1, weight2 = random_weights()
# input_data = grayleftPatch[1][1]
# y_layer = img[1][1] #feed in image as some sort of input somehow somewhere someday not today
