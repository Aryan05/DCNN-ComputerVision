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

data_size=(None,1,28,28) # Batch size x Img Channels x Height x Width
output_size=10 # We will run the example in mnist - 10 digits

input_var = T.tensor4('input')
target_var = T.ivector('targets')

lr = 1e-2
weight_decay = 1e-5
epochs = 100  #You can reduce the number of epochs to run it  faster (or run it for longer for a better model)
batch_size = 200
kernel_pool_size = 1

with open("result_cnn_1.txt", "a") as file_result:
    file_result.write('****************\n')
    file_result.write('CNN_with_dropout\n')
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
net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=6, filter_size=5)
net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=kernel_pool_size)

net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5)
net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=kernel_pool_size)


#Fully-connected + dropout
net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=100)
net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)

#Output layer:
net['out'] = lasagne.layers.DenseLayer(net['drop1'], num_units=output_size, 
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



dataset_file = 'mnist.pkl.gz'

#Download dataset if not yet done:
if not os.path.isfile(dataset_file):
    urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset_file)

#Load the dataset
f = gzip.open(dataset_file, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Convert the dataset to the shape we want
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

import time

#Run the training function per mini-batches
n_examples = x_train.shape[0]
n_batches = n_examples / batch_size

start_time = time.time()

cost_history = []
for epoch in xrange(epochs):
    st = time.time()
    batch_cost_history = []
    for batch in xrange(n_batches):
        print (batch, n_batches)
        x_batch = x_train[batch*batch_size: (batch+1) * batch_size]
        y_batch = y_train[batch*batch_size: (batch+1) * batch_size]
        
        this_cost = train_fn(x_batch, y_batch) # This is where the model gets updated
        
        batch_cost_history.append(this_cost)
    epoch_cost = np.mean(batch_cost_history)
    cost_history.append(epoch_cost)


    with open("train_errors_cnn.txt", "a") as file_result:
        file_result.write('%f\n' % epoch_cost)
    file_result.close()

    en = time.time()
    print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
    loss, acc = val_fn(x_test, y_test)
    test_error = 1 - acc
    print('Test error: %f' % test_error)
    with open("test_errors_cnn.txt", "a") as file_result:
        file_result.write('%f\n' % test_error)
    file_result.close()

    with open("result_cnn_1.txt", "a") as file_result:
        file_result.write('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds\n' % (epoch+1, epochs, epoch_cost, en-st))
        file_result.write('Test error: %f\n' % test_error)    
end_time = time.time()
print('Training completed in %.2f seconds.' % (end_time - start_time))

start_time = time.time()

loss, acc = val_fn(x_test, y_test)
test_error = 1 - acc
print('Test error: %f' % test_error)

end_time = time.time()
print('Classifying %d images completed in %.2f seconds.' % (x_test.shape[0], end_time - start_time))


with open("result_cnn_1.txt", "a") as file_result:
    file_result.write('Classifying %d images completed in %.2f seconds.' % (x_test.shape[0], end_time - start_time))
    file_result.write('\n\nEnd of run\n\n\n')
file_result.close()
