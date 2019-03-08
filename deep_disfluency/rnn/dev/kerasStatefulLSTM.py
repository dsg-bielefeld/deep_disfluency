

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

'''
This script generates a sequence of numbers, then passes a portion of them 1 at a time to an LSTM 
which is then trained to guess the next number. The LSTM is then tested on its ability to guess the
remaining numbers. A stateful LSTM network is used, so only the most recent time step needs to be 
passed in order for the network to learn. 
'''


data = [.1 , .1 , .4 , .1 , .2 ]
data = data * 300
numOfPrevSteps = 1 # We will only pass in 1 timestep at a time. The network will guess the next step from the previous step and its internal state.
batchSize = 1 # We are only tracking a single set of features through time per epoch.
featurelen = 1 # Only a single feature is being trained on. If our data was guess a list of numbers instead of 1 number each time, this would be set equal to the length of that list.
testingSize = 100 # 100 data points will be used as a test set
totalTimeSteps = len(data) # Each element in the data represents one timestep of our single feature.



print('Formatting Data')
'''
The data must be converted into a list of matrices to be fed to our network.
In this case, one matrix must be generated for item in the batch. Our batchsize
is 1, so there will only be 1 matrix in this list. The matrix consists of a list
of features. Each row has 1 column per feature. There is 1 column in the matrix 
per timestep.

So the final form of the data will be a list containing a single matrix, which has 
1 row per timestep, and only 1 column because we only have 1 feature. 
'''
X = np.zeros([batchSize, totalTimeSteps , featurelen]) 
for r in range(totalTimeSteps):
    X[0][r] = data[r]
print('Formatted Data ',X)


print('Building model...')
'''
This problem is very simple, so only 2 layers with 10 nodes
each are used. For more complicated data, more numerous and 
larger layers will likely be required. This data is very simple and 
could probably be trained off of only 1 layer. Remember to set 
return_sequences=False for the last hidden layer.
'''
model = Sequential()
model.add(LSTM(10 ,return_sequences=True, batch_input_shape=(batchSize, numOfPrevSteps , featurelen), stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(10 , return_sequences=False,stateful=True))
model.add(Dropout(0.2))
model.add(Dense( featurelen ))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.reset_states()

print('starting training')
num_epochs = 100
for e in range(num_epochs):
    print('epoch - ',e+1)
    for i in range(0,totalTimeSteps-testingSize):
        model.train_on_batch(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :], np.reshape(X[:, (i+1)*numOfPrevSteps, :], (batchSize, featurelen)) ) # Train on guessing a single element based on the previous element
    model.reset_states()
print('training complete')


print('warming up on training data') # Predict on all training data in order to warm up for testing data
warmupPredictions = []
for i in range(0,totalTimeSteps-testingSize ):
    pred = model.predict(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :] )
    warmupPredictions.append(pred)


print('testing network')   
predictions = []
testStart = totalTimeSteps-testingSize -1 #We subtract one because we want the last element of the training set to be first element of the testing set
for i in range(testStart,totalTimeSteps-1):
    pred = model.predict(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :] )
    predictions.append(pred)
    
targets = []   
for o in range(len(predictions)):
    target = X[0][o+testStart+1]
    targets.append(target)
    guess = predictions[o]
    inputs = X[0][o + testStart ]
    print('prediction ',guess,'target ',target,'inputs ',inputs)
    
model.reset_states()



#plt.plot(range(len(predictions)), [p[0][0] for p in predictions], 'b')
#plt.plot(range(len(predictions)), [t[0] for t in targets], 'g')
#plt.xlabel('time')
#plt.ylabel('value')
#plt.xlim( -50,len(predictions) + 50)
#plt.ylim( -1,3)
#plt.show()