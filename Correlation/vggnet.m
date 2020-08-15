% Read the image (preparinkldg the data)
%I=imread('/home/sonal/Caffe_Folder/caffe/examples/images/data2/mnist_0_1.jpg');
% convert it to double
%I=double(I);
formatSpec='%f';
X={};

% conv1
fileID = fopen('./VGG/VGG1_1.txt','r');
W1=fscanf(fileID,formatSpec);
X{1}=reshape(W1,[3,3,3,64]);
%X{1}=Wconv1;

% conv2
fileID = fopen('./VGG/VGG1_2.txt','r');
W1=fscanf(fileID,formatSpec);
X{2}=reshape(W1,[3,3,64,64]);

% conv3
fileID = fopen('./VGG/VGG2_1.txt','r');
W1=fscanf(fileID,formatSpec);
X{3}=reshape(W1,[3,3,64,128]);

% conv4
fileID = fopen('./VGG/VGG2_2.txt','r');
W1=fscanf(fileID,formatSpec);
X{4}=reshape(W1,[3,3,128,128]);

% conv5
fileID = fopen('./VGG/VGG3_1.txt','r');
W1=fscanf(fileID,formatSpec);
X{5}=reshape(W1,[3,3,128,256]);

% conv6
fileID = fopen('./VGG/VGG3_2.txt','r');
W1=fscanf(fileID,formatSpec);
X{6}=reshape(W1,[3,3,256,256]);

% conv7
fileID = fopen('./VGG/VGG3_3.txt','r');
W1=fscanf(fileID,formatSpec);
X{7}=reshape(W1,[3,3,256,256]);

% conv8
fileID = fopen('./VGG/VGG3_4.txt','r');
W1=fscanf(fileID,formatSpec);
X{8}=reshape(W1,[3,3,256,256]);

% conv9
fileID = fopen('./VGG/VGG4_1.txt','r');
W1=fscanf(fileID,formatSpec);
X{9}=reshape(W1,[3,3,256,512]);

% conv10
fileID = fopen('./VGG/VGG4_2.txt','r');
W1=fscanf(fileID,formatSpec);
X{10}=reshape(W1,[3,3,512,512]);

% conv11
fileID = fopen('./VGG/VGG4_3.txt','r');
W1=fscanf(fileID,formatSpec);
X{11}=reshape(W1,[3,3,512,512]);

% conv12
fileID = fopen('./VGG/VGG4_4.txt','r');
W1=fscanf(fileID,formatSpec);
X{12}=reshape(W1,[3,3,512,512]);

% conv13
fileID = fopen('./VGG/VGG5_1.txt','r');
W1=fscanf(fileID,formatSpec);
X{13}=reshape(W1,[3,3,512,512]);

% conv14
fileID = fopen('./VGG/VGG5_2.txt','r');
W1=fscanf(fileID,formatSpec);
X{14}=reshape(W1,[3,3,512,512]);

% conv15
fileID = fopen('./VGG/VGG5_3.txt','r');
W1=fscanf(fileID,formatSpec);
X{15}=reshape(W1,[3,3,512,512]);

% conv16
fileID = fopen('./VGG/VGG5_4.txt','r');
W1=fscanf(fileID,formatSpec);
X{16}=reshape(W1,[3,3,512,512]);

save('Xvgg.mat','X');