% Read the image (preparinkldg the data)
%I=imread('/home/sonal/Caffe_Folder/caffe/examples/images/data2/mnist_0_1.jpg');
% convert it to double
%I=double(I);
formatSpec='%f';

% conv1
fileID = fopen('./Alex/conv1W.txt','r');
W1=fscanf(fileID,formatSpec);
Wconv1=reshape(W1,[11,11,3,96]);

% conv2
fileID = fopen('./Alex/conv2W.txt','r');
W2=fscanf(fileID,formatSpec);
Wconv2=reshape(W2,[5,5,96,128]);

%conv3
fileID = fopen('./Alex/conv3W.txt','r');
W3=fscanf(fileID,formatSpec);
Wconv3=reshape(W3,[3,3,256,384]);

% conv4
fileID = fopen('./Alex/conv4W.txt','r');
W4=fscanf(fileID,formatSpec);
Wconv4=reshape(W4,[3,3,384,192]);

% conv5
fileID = fopen('./Alex/conv5W.txt','r');
W5=fscanf(fileID,formatSpec);
Wconv5=reshape(W5,[3,3,384,128]);

%b=zeros(size(Wconv1,4),1);
%pad=0;
%stride=1;
%conv1O=convlayer(I,stride,pad,b,Wconv1);

% pool1
%pool1O=pooling(conv1O,2,2);
% 
% % conv2
% fileID = fopen('/home/sonal/Caffe_Folder/caffe/conv2W.txt','r');
% W=fscanf(fileID,formatSpec);
% Wconv2=reshape(W,[5,5,20,50]);
% b=zeros(size(Wconv2,4),1);
% pad=0;
% stride=1;
% conv2O=convlayer(pool1O,stride,pad,b,Wconv2);
% 
% % pool2
% pool2O=pooling(conv2O,2,2);
% 
% % ip1
% fileID = fopen('/home/sonal/Caffe_Folder/caffe/IP1W.txt','r');
% W=fscanf(fileID,formatSpec);
% Wip1=reshape(W,[4*4*50,500]);
% ip1O=fc(pool2O,Wip1);
% 
% % relu 
% relu1=relu(ip1O);
% 
% % ip2
% fileID = fopen('/home/sonal/Caffe_Folder/caffe/IP2W.txt','r');
% W=fscanf(fileID,formatSpec);
% Wip2=reshape(W,[500*1*1,10]);
% ip2O=fc(relu1,Wip2);
% 
% %softmax
% finaloutput=softmax(ip2O);

save('./variablesalexnet.mat');