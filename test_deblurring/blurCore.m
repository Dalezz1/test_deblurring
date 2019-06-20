function PSF=blurCore(R)
for i=1:2*R+1
    for j=1:2*R+1
        if (i-1-R)*(i-1-R)+(j-1-R)*(j-1-R)<=R*R
            PSF(i,j)=1/(pi*R*R);
        end
    end
end
end