function max_temp = computeCorrelation( wi, wj, k )

    t1=wi.^2;
    t1=sqrt((sum(t1(:))));
    t2=wj.^2;
    t2=sqrt((sum(t2(:))));

    storetemp=zeros((2*k+1)^2-1,1);
    Count=1;         
    for x=-k:k
        for y=-k:k
        
            if(x~=0 || y~=0)
                wjt=imtranslate(wj,[x, y],'FillValues',0);
                %t2=wjt.^2;
                %t2=sqrt((sum(t2(:))));
                temp=(wi.*wjt);
                temp=sum(temp(:));
                storetemp(Count)=temp/(t1*t2);
                Count=Count+1;
              
%                 if(x==-k && y==-k)
%                     max_temp=temp;
%                 elseif (temp>max_temp)
%                     max_temp=temp;
%                 end
            end
        end 
    end                   
max_temp=max(storetemp);
end

