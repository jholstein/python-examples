##First Neural Network
import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 5], [1, 5], [3, 5]), dtype=float)
y = np.array(([10], [5], [15]), dtype=float)

predictTest = np.array(([3,5]), dtype=float)
predictTest = predictTest/np.amax(predictTest, axis=0)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/np.amax(y, axis=0) # max test score is 100

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
        self.inputSize = 27 #one neuron per possible state of board represented as [0,0,0]/[1,1,1] = [empty,x,o]
        self.hiddenSize = 9 #one neuron per space on the board
        self.outputSize = 1 #the move to make, final output neuron
        

        self.inputLayer = Neural_Layer(self.inputSize,self.hiddenSize)
        self.hiddenLayer = Neural_Layer(self.hiddenSize,self.outputSize)
        self.outputLayer = Neural_Layer(self.outputSize,0)


    def forward(self, inputData):
        self.firstForward = np.dot(inputData, self.inputLayer.Weights())
        self.secondForward = self.sigmoid(self.firstForward)
        self.thirdForward = np.dot(self.secondForward, self.hiddenLayer.Weights())
        forwardOutput = self.sigmoid(self.thirdForward)
        #print forwardOutput
        return forwardOutput

##    def backward(self, inputData, realData, predictedData):
##        self.outputError = realData-predictedData
##        self.outputErrorDelta = self.outputError*self.sigmoidPrime(predictedData)
##
##        self.hiddenError = self.outputErrorDelta.dot(self.hiddenLayer.Weights().T) # z2 error: how much our hidden layer weights contributed to output error
##        self.hiddenErrorDelta = self.hiddenError*self.sigmoidPrime(self.secondForward) # applying derivative of sigmoid to z2 error
##
##        self.weightOneAdjust = inputData.T.dot(self.hiddenErrorDelta) # adjusting first set (input --> hidden) weights
##        #Loop over the adjust, and apply changes in weight to each neuron
##        for n in range(len(self.inputLayer.Neurons)):
##            self.inputLayer.Neurons[n].Weights += self.weightOneAdjust[n]
##        self.weightTwoAdjust = self.secondForward.T.dot(self.outputErrorDelta) # adjusting second set (hidden --> output) weights
##        for m in range(len(self.hiddenLayer.Neurons)):
##            self.hiddenLayer.Neurons[m].Weights += self.weightTwoAdjust[m]

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def train (self, X, y):
        o = self.forward(X)
        #self.backward(X, y, o)


#NTester = Neural_Network()
#print "Output: "+str(NTester.forward(X))


class TicTacToeGame(object):
    
    def __init__(self):
        #Board State Array, 3x9 elements indicating state of the board with each triplet being empty, X, and O respectively,
        #with 1 being there, and 0 being not there (so the following array is an empty one)
        self.BoardState = np.array(([1,0,0,1,0,0,1,0,0,
                                     1,0,0,1,0,0,1,0,0,
                                     1,0,0,1,0,0,1,0,0]),dtype=float)

    def StartGame(self):
        #print("TEST 1")
        self.playerX = Neural_Network()
        self.playerO = Neural_Network()
        #print(self.checkWinConditions())
        while self.checkWinConditions() == 0:
            #print("Test While")
            if self.checkWinConditions() == 0:
                self.placeMove(self.playerX,1)
            if self.checkWinConditions() == 0:
                self.placeMove(self.playerO,2)
##            elif self.checkWinConditions() != 0:
##                if self.checkWinConditions() == 1:
##                    print "Player One Wins!"
##                elif self.checkWinConditions() == 2:
##                    print "Player Two Wins!"
                
    #Declare Victory if Achieved, Tell network about this
    def checkWinConditions(self):
        #Straight Victories
        for row in self.BoardState.reshape(3,3,3):
            #print self.BoardState.reshape(3,9)
            #print("ROW"+str(row))
            if (row == np.array([[0, 1, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]],dtype=int)).all():
                
                #X Wins
##                print "X Wins!"
##                print("TEST 1")
##                print("--------")
##                print row
##                print("--------")
##                print np.array([0,1,0,0,1,0,0,1,0])
                return 1
            if (row == np.array([[0, 0, 1],
                                 [0, 0, 1],
                                 [0, 0, 1]],dtype=int)).all():
                #O Wins
##                print "O Wins!"
##                print("TEST 2")
                return 2
        for col in self.BoardState.reshape(3,3,3):
##            print("COLUMN"+str(col))
            
            if (row == np.array([[0, 1, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]],dtype=int)).all():
                #X Wins
##                print "X Wins!"
##                print("TEST 3")
                return 1
            if (row == np.array([[0, 0, 1],
                                 [0, 0, 1],
                                 [0, 0, 1]],dtype=int)).all():
                #O Wins
##                print "O Wins!"
##                print("TEST 4")
                return 2
        #Diagonal Victories

            if (self.BoardState.reshape(3,3,3).diagonal() == np.array(
                                [[0, 0, 0],
                                 [1, 1, 1],
                                 [0, 0, 0]],dtype=int)).all() or ((np.fliplr(self.BoardState.reshape(3,3,3))).diagonal() == np.array(
                                [[0, 0, 0],
                                 [1, 1, 1],
                                 [0, 0, 0]],dtype=int)).all():
                #X Wins
##                print "X Wins!"
                return 1
            if (self.BoardState.reshape(3,3,3).diagonal() == np.array(
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [1, 1, 1]],dtype=int)).all() or ((np.fliplr(self.BoardState.reshape(3,3,3))).diagonal() == np.array(
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [1, 1, 1]],dtype=int)).all():
                #O Wins
                print "O Wins!"
                return 2

        return 0
        

    #Put next value into gamefield
    def placeMove(self, player, symbolInt):
        #print("TEST 2")
        move = round(player.forward(self.BoardState) * 8)
        #print(move)
        changeLoc = int(3*move)
        print self.BoardState[changeLoc+3]
        #Disallow the move if already taken
        while self.BoardState[changeLoc+1] == 1 or self.BoardState[changeLoc+2] == 1:
            move = round(player.forward(self.BoardState) * 8)
            changeLoc = int(3*move)
            
            if symbolInt == 1:
                self.BoardState[changeLoc] = 0
                self.BoardState[changeLoc+1] = 1
                self.BoardState[changeLoc+2] = 0

            elif symbolInt == 2:
                self.BoardState[changeLoc] = 0
                self.BoardState[changeLoc+1] = 0
                self.BoardState[changeLoc+2] = 1
        
            print self.BoardState.reshape(3,3,3)


TTT = TicTacToeGame()
TTT.StartGame()

##NN = Neural_Network()
##for i in xrange(10000): # trains the NN 1,000 times
##    print "Input: \n" + str(X)
##    print "Actual Output: \n" + str(y)
##    print "Predicted Output: \n" + str(NN.forward(X))
##    print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
##    print "\n"
##    NN.train(X, y)
##
##print "----"+str(NN.forward(predictTest))
