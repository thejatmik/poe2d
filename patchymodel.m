clear 
clc

lingx=5000;
lingy=240;
modx=lingx;
mody=2800;
r=8;
dia=2*r;
xx=1:lingx;
yy=1:lingy;

[rr,cc]=meshgrid(xx,yy);

%--------------------------------------------------------------------------------------
le=zeros(lingy,lingx);
for i=r+1:dia:lingx
    for j=r+1:dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le=le+C;
    end
end
for i=1:lingx
    for j=1:lingy
        if (le(j,i)>1) 
            le(j,i)=1;
        end
    end
end
subplot(6,1,1)
xlim([0 lingy])

for i=1:lingx
    for j=1:lingy
        if (le(j,i)==1) 
            le(j,i)=2;
            %le(j,i)=4;
        elseif (le(j,i)==0)
            le(j,i)=3;
        end
    end
end
model=zeros(mody,modx)+1;
for j=2001:2001+lingy-1
    model(j,:)=le(j-2000,:);
end
imagesc(model)
axis equal
title('tipe 1')
model=reshape(model,[mody*modx,1]);
arr2bin('patchy1water.bin',model);

%-------------------------------------------------------------------------------------
le2=zeros(lingy,lingx);
for i=r+1:2*dia:lingx
    for j=r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le2=le2+C;
    end
end
for i=1:lingx
    for j=1:lingy
        if (le2(j,i)>1) 
            le2(j,i)=1;
        end
    end
end
subplot(6,1,2)
xlim([0 lingy])

for i=1:lingx
    for j=1:lingy
        if (le2(j,i)==1) 
            le2(j,i)=2;
            %le2(j,i)=4;
        elseif (le2(j,i)==0)
            le2(j,i)=3;
        end
    end
end
model=zeros(mody,modx)+1;
for j=2001:2001+lingy-1
    model(j,:)=le2(j-2000,:);
end
imagesc(model)
axis equal
title('tipe 2')
model=reshape(model,[mody*modx,1]);
arr2bin('patchy2water.bin',model);

%-------------------------------------------------------------------------------------
le3=zeros(lingy,lingx);
for i=r+1:4*dia:lingx
    for j=r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le3=le3+C;
    end
end
for i=1:lingx
    for j=1:lingy
        if (le3(j,i)>1) 
            le3(j,i)=1;
        end
    end
end
subplot(6,1,3)
xlim([0 lingy])

for i=1:lingx
    for j=1:lingy
        if (le3(j,i)==1) 
            le3(j,i)=2;
            %le3(j,i)=4;
        elseif (le3(j,i)==0)
            le3(j,i)=3;
        end
    end
end
model=zeros(mody,modx)+1;
for j=2001:2001+lingy-1
    model(j,:)=le3(j-2000,:);
end
imagesc(model)
axis equal
title('tipe 3')
model=reshape(model,[mody*modx,1]);
arr2bin('patchy3water.bin',model);

%-------------------------------------------------------------------------------------
le4=zeros(lingy,lingx);
off=0;
for i=r+1:2*dia:lingx
    for j=r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le4=le4+C;
    end
end
for i=3*r+1:2*dia:lingx
    for j=3*r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le4=le4+C;
    end
end
for i=1:lingx
    for j=1:lingy
        if (le4(j,i)>1) 
            le4(j,i)=1;
        end
    end
end
subplot(6,1,4)
xlim([0 lingy])

for i=1:lingx
    for j=1:lingy
        if (le4(j,i)==1) 
            le4(j,i)=2;
            %le4(j,i)=4;
        elseif (le4(j,i)==0)
            le4(j,i)=3;
        end
    end
end
model=zeros(mody,modx)+1;
for j=2001:2001+lingy-1
    model(j,:)=le4(j-2000,:);
end
imagesc(model)
axis equal
title('tipe 4')
model=reshape(model,[mody*modx,1]);
arr2bin('patchy4water.bin',model);

%-------------------------------------------------------------------------------------
le5=zeros(lingy,lingx);
for i=r+1:2*dia:lingx
    for j=3*r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le5=le5+C;
    end
end
for i=r+1:dia:lingx
    for j=r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le5=le5+C;
    end
end
for i=1:lingx
    for j=1:lingy
        if (le5(j,i)>1) 
            le5(j,i)=1;
        end
    end
end
subplot(6,1,5)
xlim([0 lingy])

for i=1:lingx
    for j=1:lingy
        if (le5(j,i)==1) 
            le5(j,i)=2;
            %le5(j,i)=4;
        elseif (le5(j,i)==0)
            le5(j,i)=3;
        end
    end
end
model=zeros(mody,modx)+1;
for j=2001:2001+lingy-1
    model(j,:)=le5(j-2000,:);
end
imagesc(model)
axis equal
title('tipe 5')
model=reshape(model,[mody*modx,1]);
arr2bin('patchy5water.bin',model);

%-------------------------------------------------------------------------------------
le6=zeros(lingy,lingx);
for i=r+1:4*dia:lingx
    for j=r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le6=le6+C;
    end
end
for i=5*r+1:4*dia:lingx
    for j=3*r+1:2*dia:lingy
        C = sqrt((rr-i).^2+(cc-j).^2)<=r;
        le6=le6+C;
    end
end
for i=1:lingx
    for j=1:lingy
        if (le6(j,i)>1) 
            le6(j,i)=1;
        end
    end
end
subplot(6,1,6)
xlim([0 lingy])

for i=1:lingx
    for j=1:lingy
        if (le6(j,i)==1) 
            le6(j,i)=2;
            %le6(j,i)=4;
        elseif (le6(j,i)==0)
            le6(j,i)=3;
        end
    end
end
model=zeros(mody,modx)+1;
for j=2001:2001+lingy-1
    model(j,:)=le6(j-2000,:);
end
imagesc(model)
axis equal
title('tipe 6')
model=reshape(model,[mody*modx,1]);
arr2bin('patchy6water.bin',model);
