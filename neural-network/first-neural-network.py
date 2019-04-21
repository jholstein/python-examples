##First Neural Network
import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100


# Neuron class, takes number of next layer neurons to generate weights, if no next layer neurons exist, set to output neuron
class Neuron(object):
    def __init__(self,nextLayerSize):
        if nextLayerSize != 0:
            self.Weights = np.random.randn(1,nextLayerSize)
            #print self.Weights
            self.output = False
        else:
            self.output = True
    

class Neural_Layer(object):
    
    def __init__(self,neurons,nextLayerSize):
        self.Neurons = []
        for x in range(neurons):
            self.Neurons.append(Neuron(nextLayerSize))
            #print "adding neuron"
            #print self.Neurons
                
    def Weights(self):
        outputWeights = []
        for n in self.Neurons:
            if outputWeights == []:
                outputWeights = n.Weights
            else:
                outputWeights = np.concatenate((outputWeights,n.Weights), axis=0)
                

        #print "-----"+str(outputWeights)
        return outputWeights
            



class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.hiddenSize = 6
        self.outputSize = 1
        

        self.inputLayer = Neural_Layer(self.inputSize,self.hiddenSize)
        self.hiddenLayer = Neural_Layer(self.hiddenSize,self.outputSize)
        self.outputLayer = Neural_Layer(self.outputSize,0)


    def forward(self, inputData):
        self.firstForward = np.dot(inputData, self.inputLayer.Weights())
        self.secondForward = self.sigmoid(self.firstForward)
        self.thirdForward = np.dot(self.secondForward, self.hiddenLayer.Weights())
        forwardOutput = self.sigmoid(self.thirdForward)
        return forwardOutput

    def backward(self, inputData, realData, predictedData):
        self.outputError = realData-predictedData
        self.outputErrorDelta = self.outputError*self.sigmoidPrime(predictedData)

        self.hiddenError = self.outputErrorDelta.dot(self.hiddenLayer.Weights().T) # z2 error: how much our hidden layer weights contributed to output error
        self.hiddenErrorDelta = self.hiddenError*self.sigmoidPrime(self.secondForward) # applying derivative of sigmoid to z2 error

        self.weightOneAdjust = inputData.T.dot(self.hiddenErrorDelta) # adjusting first set (input --> hidden) weights
        #Loop over the adjust, and apply changes in weight to each neuron
        for n in range(len(self.inputLayer.Neurons)):
            self.inputLayer.Neurons[n].Weights += self.weightOneAdjust[n]
        self.weightTwoAdjust = self.secondForward.T.dot(self.outputErrorDelta) # adjusting second set (hidden --> output) weights
        for m in range(len(self.hiddenLayer.Neurons)):
            self.hiddenLayer.Neurons[m].Weights += self.weightTwoAdjust[m]

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


#NTester = Neural_Network()
#print "Output: "+str(NTester.forward(X))



NN = Neural_Network()
for i in xrange(10000): # trains the NN 1,000 times
    #print "Input: \n" + str(X)
    #print "Actual Output: \n" + str(y)
    #print "Predicted Output: \n" + str(NN.forward(X))
    #print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
    #print "\n"
    NN.train(X, y)

print "Input: \n" + str(X)
print "Actual Output: \n" + str(y)
print "Predicted Output: \n" + str(NN.forward(X))
print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
print "\n"


