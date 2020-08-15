import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_mode_cpu();

net = caffe.Net('/home/parv/Dropbox/SMAI_Project/Alex/deploy.prototxt','/home/parv/Downloads/bvlc_alexnet.caffemodel',caffe.TEST);

W1=net.params['conv1'][0].data;
W1.tofile('conv1W.txt',sep=" ",format="%s");

W1=net.params['conv2'][0].data;
W1.tofile('conv2W.txt',sep=" ",format="%s");

W1=net.params['conv3'][0].data;
W1.tofile('conv3W.txt',sep=" ",format="%s");

W1=net.params['conv4'][0].data;
W1.tofile('conv4W.txt',sep=" ",format="%s");

W1=net.params['conv5'][0].data;
W1.tofile('conv5W.txt',sep=" ",format="%s");


net = caffe.Net('/home/parv/Dropbox/SMAI_Project/VGG/deploy.prototxt','/home/parv/Downloads/VGG_ILSVRC_19_layers.caffemodel',caffe.TEST);

W1=net.params['conv1_1'][0].data;
W1.tofile('VGG1_1.txt',sep=" ",format="%s");


W1=net.params['conv1_2'][0].data;
W1.tofile('VGG1_2.txt',sep=" ",format="%s");


W1=net.params['conv2_1'][0].data;
W1.tofile('VGG2_1.txt',sep=" ",format="%s");


W1=net.params['conv2_2'][0].data;
W1.tofile('VGG2_2.txt',sep=" ",format="%s");


W1=net.params['conv3_1'][0].data;
W1.tofile('VGG3_1.txt',sep=" ",format="%s");


W1=net.params['conv3_2'][0].data;
W1.tofile('VGG3_2.txt',sep=" ",format="%s");


W1=net.params['conv3_3'][0].data;
W1.tofile('VGG3_3.txt',sep=" ",format="%s");


W1=net.params['conv3_4'][0].data;
W1.tofile('VGG3_4.txt',sep=" ",format="%s");


W1=net.params['conv4_1'][0].data;
W1.tofile('VGG4_1.txt',sep=" ",format="%s");


W1=net.params['conv4_2'][0].data;
W1.tofile('VGG4_2.txt',sep=" ",format="%s");


W1=net.params['conv4_3'][0].data;
W1.tofile('VGG4_3.txt',sep=" ",format="%s");


W1=net.params['conv4_4'][0].data;
W1.tofile('VGG4_4.txt',sep=" ",format="%s");


W1=net.params['conv5_1'][0].data;
W1.tofile('VGG5_1.txt',sep=" ",format="%s");


W1=net.params['conv5_2'][0].data;
W1.tofile('VGG5_2.txt',sep=" ",format="%s");


W1=net.params['conv5_3'][0].data;
W1.tofile('VGG5_3.txt',sep=" ",format="%s");


W1=net.params['conv5_4'][0].data;
W1.tofile('VGG5_4.txt',sep=" ",format="%s");

