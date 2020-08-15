clear;
clc;
close all;

%NOTE: 
%For all 3D matrics, parameters follow [channels, width, height] format.

%% Input portion:
% change this when actually want to work with this

cl=3;
wl=150;
hl=150;
%imageL=zeros(cl,wl,hl);%this is Il
imageL=(255).*rand([cl,wl,hl]);

cl2=12;%this is cl+1
z1=12;%this is z'-size of meta filter
%Wl=zeros(cl2,cl,z1,z1);
Wl=(0.4).*rand([cl2,cl,z1,z1])-0.2;

z=4;%effective filter size
s=3;%pooling size;

%% Output portion:
n=((z1-z+1)^2)/s^2;
wl2=wl+z-1;%this is wl+1
hl2=hl+z-1;%this is hl+1
%imageL2=zeros(n*cl2,wl2, hl2); this is not needed

%% Algorithm starts here:
fprintf('Step 1: Creating identity matrix.\n');
identityL=eye(cl*z*z);

fprintf('Step 2: Reshaping identity matrix.\n');
identityL=reshape(identityL,[cl*z*z,cl,z,z]);

%reshape(identityL(1,:,:,:),[size(identityL,2),size(identityL,3),size(identityL,4)]);

fprintf('Step 3: Convolving meta filter with identity matrices to generate smaller filters.\n');
wTildaL=convolution2(Wl,identityL);

fprintf('Step 4: Reshaping to smaller filters.\n');
wTildaL=reshape(wTildaL,[cl2*(z1-z+1)*(z1-z+1),cl,z,z]);

fprintf('Step 5: Convolving filters with input image.\n');
oL2=convolution(wTildaL, imageL);

fprintf('Step 6: Reshaping output image.\n');
oL2=reshape(oL2,[cl2*wl2*hl2,(z1-z+1),(z1-z+1)]);

fprintf('Step 7: Pooling.\n');
imageL2=pool(oL2,s);

fprintf('Step 8: Reshaping.\n');
imageL2=reshape(imageL2,[cl2*n,wl2,hl2]);

fprintf('Completed.\n');
