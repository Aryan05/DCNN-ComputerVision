%clc; clear; close all;
X=importdata('./Xvgg.mat');
k=1;
ro=zeros(size(X,2),1);

for lselect=1:9
     'current layer'
     lselect
   
    convlayer=X{lselect};
    p_avg=0;
    W={};
    
    N=size(convlayer,4);
    

    for i=1:N
        %rng(i);
       % W{i}=convlayer(:,:,:,i);
        W{i}=randn(size(convlayer(:,:,:,i)));
    end
    %N number of filters;
    %W filter cell;
    
    for i=1:N
        parray=zeros(N-1,1);
        Count=1;
        for j=1:N
            if(i~=j)
                parray(Count)=computeCorrelation(W{i},W{j},k);
                Count=Count+1;
                %    if(p>p_max)
                %       p_max=p;
                %  end
            end
        end
        p_avg=p_avg+max(parray);
    end
    ro(lselect)=p_avg/N;
    save('final_vgg_random.mat','ro');

end
