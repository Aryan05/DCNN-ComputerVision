#For running the model:
import lasagne
import theano
from theano import tensor as T
import numpy as np

#For loading the dataset:
import os
import gzip
import cPickle
import urllib

from data import load_cifar10


data_size=(None,3,32,32) # Batch size x Img Channels x Height x Width
output_size=10 # We will run the example in mnist - 10 digits

input_var = T.tensor4('input')
target_var = T.ivector('targets')

epochs = 3  #You can reduce the number of epochs to run it  faster (or run it for longer for a better model)
batch_size = 200   
lr = 1
weight_decay = 0.5
kernel_pool_size = 2

#file_result = open('result_cnn_1.txt','a')
with open("result_cnn_1.txt", "a") as file_result:
    file_result.write('****************\n')
    file_result.write('CNN_cifar\n')
    file_result.write('Parameters\n')
    file_result.write('num_epochs: %d\n' % epochs)
    file_result.write('batch_size: %d\n' % batch_size)
    file_result.write('learning rate: %f\n' % lr)
    file_result.write('weight decay: %f\n' % weight_decay)
    file_result.write('Kernel Pool Size: %d\n' % kernel_pool_size)
    file_result.write('\n\n')


net = {}

#Input layer:
net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

#Convolution + Pooling
net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv1'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['pool1'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)

net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['conv4'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv3'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['pool2'] = lasagne.layers.Pool2DLayer(net['conv4'], pool_size=2)

net['conv5'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['conv6'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv5'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['pool3'] = lasagne.layers.Pool2DLayer(net['conv6'], pool_size=2)

net['conv7'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool3'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['conv8'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv7'], num_filters=128, filter_size=3,pad='same', nonlinearity=lasagne.nonlinearities.rectify))
net['pool4'] = lasagne.layers.Pool2DLayer(net['conv8'], pool_size=2)


#Fully-connected + dropout
#net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=100)
#net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)

#Output layer:
net['out'] = lasagne.layers.DenseLayer(net['pool4'], num_units=output_size, 
                                       nonlinearity=lasagne.nonlinearities.softmax)

#Note: Lasagne does not implement a class for "Model". Usually, you only need the output layer to:
# 1) Obtain the result of the layer (predictions)
# 2) Obtain a list of all parameters from the model (e.g. for weight decay)

#Define hyperparameters. These could also be symbolic variables 

#Loss function: mean cross-entropy
prediction = lasagne.layers.get_output(net['out'])
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

#Also add weight decay to the cost function
weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
loss += weight_decay * weightsl2

#Get the update rule. Here we will use a more advanced optimization algorithm: ADAM [1]
params = lasagne.layers.get_all_params(net['out'], trainable=True)
updates = lasagne.updates.adam(loss, params)

test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

#Note that train_fn has a "updates" rule. Whenever we call this function, it updates the parameters of the model.
train_fn = theano.function([input_var, target_var], loss, updates=updates, name='train')
val_fn = theano.function([input_var, target_var], [test_loss, test_acc], name='validation')
get_preds = theano.function([input_var], test_prediction, name='get_preds')

#Load the dataset
load_data = load_cifar10.load_data
datasets = load_data()
train_x, valid_x, test_x = [data[0] for data in datasets]
train_y, valid_y, test_y = [data[1] for data in datasets]

train_x = np.concatenate([train_x, valid_x], axis=0)
train_y = np.concatenate([train_y, valid_y], axis=0)
valid_x = test_x
valid_y = test_y

#Convert the dataset to the shape we want
train_x = train_x.reshape(-1, 3, 32, 32)
test_x = test_x.reshape(-1, 3, 32, 32)
train_y = train_y.astype(np.int32)
test_y = test_y.astype(np.int32)

import time

#Run the training function per mini-batches
n_examples = train_x.shape[0]
n_batches = n_examples / batch_size

start_time = time.time()

cost_history = []
for epoch in xrange(epochs):
    st = time.time()
    batch_cost_history = []
    for batch in xrange(n_batches):
        print (batch, n_batches)
        x_batch = train_x[batch*batch_size: (batch+1) * batch_size]
        y_batch = train_y[batch*batch_size: (batch+1) * batch_size]
        
        this_cost = train_fn(x_batch, y_batch) # This is where the model gets updated
        
        batch_cost_history.append(this_cost)
    epoch_cost = np.mean(batch_cost_history)
    cost_history.append(epoch_cost)
    en = time.time()
    print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
    loss, acc = val_fn(test_x, test_y)
    test_error = 1 - acc
    print('Test error: %f' % test_error)
    with open("result_cnn_1.txt", "a") as file_result:
        file_result.write('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds\n' % (epoch+1, num_epochs, epoch_cost, en-st))
        file_result.write('Test error: %f\n' % test_error)
end_time = time.time()
print('Training completed in %.2f seconds.' % (end_time - start_time))

with open("result_cnn_1.txt", "a") as file_result:
    file_result.write('\n\nEnd of run\n\n\n')
file_result.close()


start_time = time.time()

loss, acc = val_fn(test_x, test_y)
test_error = 1 - acc
print('Test error: %f' % test_error)

end_time = time.time()
print('Classifying %d images completed in %.2f seconds.' % (test_x.shape[0], end_time - start_time))