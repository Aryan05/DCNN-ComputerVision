function O = convolution(W, I)
    % W should be 4-D, I should be 3-D
    % W is the filter and I is the image
    [cl2, cl, z, z] = size(W);
    [cl, wl, hl] = size(I);
    O = zeros(cl2, wl+z-1, hl+z-1);
    for k=1:cl2
        for i=1:wl+z-1
            for j=1:hl+z-1
                val=0;
                    for c=1:cl
                        for i2=1:z
                            if(i+i2-1>wl)
                                break;
                            end
                            for j2=1:z
                                if(j2+j-1>hl)
                                    break;
                                end
                                val=val+W(k,c,i2,j2)*I(c,i+i2-1,j+j2-1);                      
                            end
                        end
                    end
                O(k,i,j)=val;
            end
        end
    end
end
