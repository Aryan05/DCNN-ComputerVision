vgg=importdata('final_vgg.mat');
vgg_random=importdata('final_vgg_random.mat');
alex=importdata('final_alex.mat');
alex_random=importdata('final_alex_random.mat');

subplot(1,3,1);
plot([1:5],alex','*','Color','red'), hold on
plot([1:5],alex_random','*','Color','blue');
xlabel('Convolution Layers');
ylabel('Average Maximum Correlation');
title('Alex');

subplot(1,3,[2,3]);
plot([1:9],vgg','*','Color','red'), hold on
plot([1:9],vgg_random','*','Color','blue');
xlabel('Convolution Layers');
ylabel('Average Maximum Correlation');
title('VGG-19 first 9 layers');
