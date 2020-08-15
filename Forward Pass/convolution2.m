function O = convolution2( W, I )
    % both W and I should be 4-D
    % W is the filter and I is the image
    [cl2, cl, z2, z2] = size(W);
    [clzz, cl, z, z] = size(I);%clzz is cl*z*z
    
    O = zeros(cl2, clzz, z2-z+1, z2-z+1);
    img = zeros(cl, z, z);
    filters = zeros(cl, z2, z2);
    
    for iter=1:clzz
        img(:,:,:)=I(iter,:,:,:);
        for i=1:cl2
            filters(:,:,:)=W(i,:,:,:);
            for j=1:cl
                subImage=img(j,:,:);
                subImage=reshape(subImage,[z,z]);
                it1=1;
                for k=1:z:z2
                    if(k+z-1>z2)
                        break;
                    end
                    it2=1;
                    for l=1:z:z2
                        if(l+z-1>z2)
                            break;
                        end
                        subFilter=filters(j,k:k+z-1,l:l+z-1);
                        subFilter=reshape(subFilter,[z,z]);
                        temp=conv2(subImage, subFilter, 'valid');
                        O(i,iter,it1,it2)=temp;
                        it2=it2+1;
                    end
                    it1=it1+1;
                end
            end
        end
    end
end

