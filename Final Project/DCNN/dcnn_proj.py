import lasagne
import theano
from theano import tensor as T
import numpy as np
import os
import gzip
import cPickle
import urllib
import time
from data import load_cifar10

class DCNN(lasagne.layers.conv.BaseConvLayer):
    def __init__(self, incoming, cl_1, cl, z_large, z, pool_size, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
convolution=theano.tensor.nnet.conv2d, **kwargs):
        super(DCNN, self).__init__(incoming, cl_1, (z_large, z_large), (1,1), 0, untie_biases, W, b, nonlinearity, flip_filters, n=None, **kwargs)
        self.cl_1 = cl_1
        self.cl = cl
        self.z_large = z_large
        self.z = z
        self.pool_size = pool_size

    def get_output_for(self, input, **kwargs):
        # creating the identity matrix
        identity = theano.tensor.eye(self.cl*self.z*self.z, self.cl*self.z*self.z)
        # reshaping the identity matrix
        reshaped_identity = identity.reshape((self.cl*self.z*self.z, self.cl, self.z, self.z))
        # forming smaller filters from meta filter
        W_tilda = theano.tensor.nnet.conv2d(self.W, reshaped_identity, border_mode='valid', filter_flip=False)
        # reshape smaller filters
        W_tilda = theano.tensor.reshape(W_tilda.dimshuffle(0, 2, 3, 1), (self.cl_1*(self.z_large-self.z+1)*(self.z_large-self.z+1), self.cl, self.z, self.z))
        # output 
        output = theano.tensor.nnet.conv2d(input, W_tilda, self.input_shape, (self.cl_1*(self.z_large-self.z+1)*(self.z_large-self.z+1), self.cl, self.z, self.z), subsample=(1, 1), border_mode='half', filter_flip=True)
        # pooling
        if self.pool_size > 1:
            output = theano.tensor.reshape(output, (input.shape[0]*self.cl_1, (self.z_large-self.z+1), (self.z_large-self.z+1), np.prod(self.input_shape[2:]))).dimshuffle(0, 3, 1, 2)
            output = theano.tensor.signal.pool.pool_2d(output, (self.pool_size,)*2, ignore_border=True)
            output = theano.tensor.reshape(output.dimshuffle(0, 2, 3, 1), (input.shape[0], -1) + self.input_shape[2:])
        return output        

    def get_output_shape_for(self, input_shape):
        n = (self.z_large-self.z+1)*(self.z_large-self.z+1) / (self.pool_size**2)
        return (input_shape[0], self.cl_1*n) + input_shape[2:]       

if __name__ == '__main__':
    data_size = (None, 3, 32, 32) # Batch size x Img Channels x Height x Width
    output_size = 10 # We will run the example in mnist - 10 digits

    input_var = T.tensor4('input')
    target_var = T.ivector('targets')

    # Variables
    num_epochs = 10
    learning_rate = 1
    patience = 10
    metafilter_shape = [(128, 3, 4, 4),(128, 128, 4, 4),(128, 128, 4, 4),(128, 128, 4, 4),(128, 128, 4, 4),(128, 128, 4, 4),(128, 128, 4, 4),(128, 128, 4, 4)]
    kernel_size = 3
    kernel_pool_size = 2
    learning_decay = 0.5
    dropout_rate = 0.5
    batch_size = 200

    #results
    with open("result_dcnn_1.txt", "a") as file_result:
        file_result.write('****************\n')
        file_result.write('DCNN Cifar 10 with dropout and patience\n')
        file_result.write('Parameters\n')
        file_result.write('num_epochs: %d\n' % num_epochs)
        file_result.write('patience: %d\n' % patience)
        file_result.write('batch_size: %d\n' % batch_size)
        file_result.write('learning decay: %f\n' % learning_decay)
        file_result.write('dropout rate: %f\n' % dropout_rate)
        file_result.write('Kernel Pool Size: %d\n' % kernel_pool_size)
        file_result.write('metafilter number: %d %d\n' % (metafilter_shape[0][0], metafilter_shape[0][1]))
        file_result.write('metafilter number: %d %d\n' % (metafilter_shape[1][0], metafilter_shape[1][1]))
        file_result.write('\n\n')   

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
    valid_x = valid_x.reshape(-1, 3, 32, 32)
    test_x = test_x.reshape(-1, 3, 32, 32)
    train_y = train_y.astype(np.int32)
    test_y = test_y.astype(np.int32)

    xmean = train_x.mean(axis=0)
    train_x -= xmean
    valid_x -= xmean
    test_x -= xmean

    # Convolution model
    net = {}

    # Input layer
    net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)
    # First hidden layer
    net['conv1'] = lasagne.layers.batch_norm(DCNN(net['data'], metafilter_shape[0][0], metafilter_shape[0][1], metafilter_shape[0][2], kernel_size, kernel_pool_size))
    net['conv2'] = lasagne.layers.batch_norm(DCNN(net['conv1'], metafilter_shape[1][0], metafilter_shape[1][1], metafilter_shape[1][2], kernel_size, kernel_pool_size))
    net['pool1'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)

    net['conv3'] = lasagne.layers.batch_norm(DCNN(net['pool1'], metafilter_shape[2][0], metafilter_shape[2][1], metafilter_shape[2][2], kernel_size, kernel_pool_size))
    net['conv4'] = lasagne.layers.batch_norm(DCNN(net['conv3'], metafilter_shape[3][0], metafilter_shape[3][1], metafilter_shape[3][2], kernel_size, kernel_pool_size))
    net['pool2'] = lasagne.layers.Pool2DLayer(net['conv4'], pool_size=2)

    net['conv5'] = lasagne.layers.batch_norm(DCNN(net['pool2'], metafilter_shape[4][0], metafilter_shape[4][1], metafilter_shape[4][2], kernel_size, kernel_pool_size))
    net['conv6'] = lasagne.layers.batch_norm(DCNN(net['conv5'], metafilter_shape[5][0], metafilter_shape[5][1], metafilter_shape[5][2], kernel_size, kernel_pool_size))
    net['pool3'] = lasagne.layers.Pool2DLayer(net['conv6'], pool_size=2)

    net['conv7'] = lasagne.layers.batch_norm(DCNN(net['pool3'], metafilter_shape[6][0], metafilter_shape[6][1], metafilter_shape[6][2], kernel_size, kernel_pool_size))
    net['conv8'] = lasagne.layers.batch_norm(DCNN(net['conv7'], metafilter_shape[7][0], metafilter_shape[7][1], metafilter_shape[7][2], kernel_size, kernel_pool_size))
    net['pool4'] = lasagne.layers.Pool2DLayer(net['conv8'], pool_size=2)

    
    # Fully connected and dropout layer
    
    #net['fc1'] = lasagne.layers.DenseLayer(net['conv2'], num_units=100)
    #net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
    
    # Output layer
    net['out'] = lasagne.layers.DenseLayer(net['pool4'], num_units=output_size, nonlinearity=lasagne.nonlinearities.softmax)

    #Note: Lasagne does not implement a class for "Model". Usually, you only need the output layer to:
    # 1) Obtain the result of the layer (predictions)
    # 2) Obtain a list of all parameters from the model (e.g. for weight decay)
    
    #Define hyperparameters. These could also be symbolic variables 
    lr = learning_rate
    weight_decay = learning_decay

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
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    #Note that train_fn has a "updates" rule. Whenever we call this function, it updates the parameters of the model.
    train_fn = theano.function([input_var, target_var], loss, updates=updates, name='train')
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], name='validation')
    get_preds = theano.function([input_var], test_prediction, name='get_preds')

    #Run the training function per mini-batches
    num_examples = train_x.shape[0]
    num_valid = valid_x.shape[0]
    num_batches = num_examples / batch_size

    start_time = time.time()

    cost_history = []
    valid_history = []
    best_valid = 1.
    bad_count = 0
    for epoch in xrange(num_epochs):
        st = time.time()
        batch_cost_history = []
        perm = np.random.permutation(num_examples)
        for batch in xrange(num_batches):
            print (batch, num_batches)
            index = perm[batch*batch_size: (batch+1) * batch_size]
            x_batch = train_x[index]
            y_batch = train_y[index]    
            this_cost = train_fn(x_batch, y_batch) # This is where the model gets updated
            batch_cost_history.append(this_cost)
        
        epoch_cost = np.mean(batch_cost_history)
        cost_history.append(epoch_cost)
        en = time.time()
        print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, num_epochs, epoch_cost, en-st))
        
        c = []
        for batch in xrange(num_valid/batch_size):
            print (batch, num_valid/batch_size)
            x_batch = valid_x[batch*batch_size: (batch+1) * batch_size]
            y_batch = valid_y[batch*batch_size: (batch+1) * batch_size]
            loss, acc = val_fn(x_batch, y_batch)
            c = np.append(c, (1 - acc))
        valid_history.append(np.mean(c))

        if valid_history[-1] <= best_valid:
            best_valid = valid_history[-1]
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= patience:
                lr = lr*float(weight_decay)
                bad_count=0
                
        loss, acc = val_fn(test_x, test_y)
        test_error = 1 - acc
        print('Test error: %f' % test_error)
        
        with open("result_dcnn_1.txt", "a") as file_result:
            file_result.write('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds\n' % (epoch+1, num_epochs, epoch_cost, en-st))
            file_result.write('Test error: %f\n' % test_error)
    
    end_time = time.time()
    print('Training completed in %.2f seconds.' % (end_time - start_time))

    with open("result_dcnn_1.txt", "a") as file_result:
        file_result.write('\n\nEnd of run\n\n\n')
    file_result.close()

    start_time = time.time()

    loss, acc = val_fn(test_x, test_y)
    test_error = 1 - acc
    print('Test error: %f' % test_error)

    end_time = time.time()
    print('Classifying %d images completed in %.2f seconds.' % (test_x.shape[0], end_time - start_time))