import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def listSum(list):
    for i in range(len(list)):
       list[len(list)-1-i] = sum(list,len(list)-1-i)
    return list
               
def sum(list, row):
    if row == len(list)-1:
        return list[row]
    else:
        return list[row] + list[row + 1]


Y = [] # output data list

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv('panel_sum_cost.csv')
dataframe = dataframe.fillna(method='ffill') # filling any null or Nan in dataset
dataframe = dataframe[dataframe.Cost != 0] # removing rows that contain 0 cost

#formatting output data
allPatRec = dataframe.values
st_index = 0
end_index = 0
total = 0
calcTrainSize = int(len(allPatRec)*(80/100))
print(len(allPatRec), calcTrainSize)

splitPoint = False
alreadyChecked = False
for i in range(len(allPatRec)-1):
    str1 = allPatRec[i][0].split(" ") #splitting first column string to access claim number only
    str2 = allPatRec[i+1][0].split(" ")
    onePatRec = []

    if str1[1] == str2[1]: 
        end_index = end_index+1
        if end_index == calcTrainSize:
            splitPoint = True
        
    else:
        if splitPoint == True and alreadyChecked == False:
            #if str2[3] = "1":
            calcTrainSize = end_index+1 #first row of next claim (patient) number
            alreadyChecked = True
        for k in range(st_index,end_index+1):
            onePatRec.append(allPatRec[k][20])
        costUntilrecoveryList = listSum(onePatRec)
        Y = Y + costUntilrecoveryList
        end_index = end_index+1 # to move end_index and st_index to the start of next patient
        st_index = end_index

#print("st_index, end_index", st_index , end_index )
# below code is to cater last patiend record since above code does not run for last patiend
# because if statement just updates indices but since there is not next record,
# else will not execute for last patient

lastPatRec = []
for k in range(st_index,end_index+1):
    lastPatRec.append(allPatRec[k][20])
costUntilrecoveryList = listSum(lastPatRec)
Y = Y + costUntilrecoveryList
Y = numpy.array(Y)
Y = Y.reshape(len(Y),1)
Y = Y.astype('float32') #converting those values to float

##print('split index', calcTrainSize)
##print(allPatRec[calcTrainSize])

del dataframe['Patient'] #removing first column
del dataframe['Cost'] # removing target column i.e., Cost
datasetX = dataframe.values # now it has only input features
datasetX = datasetX.astype('float32') #converting those values to float
##print(len(datasetX), datasetX[:1,:]) # printing first row of 2-d array
##X = datasetX[:,0:3] # accessing 2-d array. first 3 elements of each row
##print(X)
## normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) # scalling them between 0-1
datasetX = scaler.fit_transform(datasetX)
##print(datasetX[:1,:]) # printing first row of 2-d array
##print(len(Y), Y)
scalerY = MinMaxScaler(feature_range=(0, 1))
Y = scalerY.fit_transform(Y)
##print(Y[0]) # printing first row of 2-d array

## split into train and test sets
##train_size = int(len(datasetX) * 0.80) # splitting dataset list of lists into two lists of lists
##test_size = len(datasetX) - train_size
trainX, testX = datasetX[0:calcTrainSize,:], datasetX[calcTrainSize:len(datasetX),:]
trainY, testY = Y[0:calcTrainSize,:], Y[calcTrainSize:len(datasetX),:]

trainX = numpy.array(trainX)
trainY = numpy.array(trainY)
testX = numpy.array(testX)
testY = numpy.array(testY)
##print(trainY[len(trainY)-1], testY[0])

## reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) #now its 3 dimetional list instead of 2 dimentional but still having one value in each list of list of list
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1])) #gives a new shape to an array without changing its data. shape returns the dimensions of array. if array has n rows and m columns, then array.shape is (n,m) and shape[0] returns n and shape[1] returns m


## create and fit the LSTM network
model = Sequential() # network has a visible layer with 1 input
model.add(LSTM(19, activation='tanh', input_shape=(1, 19), use_bias=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

## make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
##print(len(trainPredict), trainPredict)
##print(len(trainY), trainY)

## invert predictions
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform(trainY)
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform(testY)
##print(trainY[0])
##print(len(trainPredict), trainPredict[:,0])
##print(len(trainY), trainY[0])

## calculate root mean squared error # make sure dimensions of parameter arrays are same
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
