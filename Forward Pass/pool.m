function I = pool(W, s)
    [c,w,h]=size(W);
    I=zeros(c,w/s,h/s);
    for i=1:c
        it1=1;
        for j=1:s:w
           leftx=j;
           rightx=min(j+s-1,w);
           it2=1;
           for k=1:s:h
               lefty=k;
               righty=min(k+s-1,h);
               temp=W(i,leftx:rightx,lefty:righty);
               temp=reshape(temp,[s,s]);
               mx=max(max(temp));
               I(i,it1,it2)=mx;
               it2=it2+1;
           end
           it1=it1+1;
        end
    end
end

