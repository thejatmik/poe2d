function [ ] = arr2bin( filename,data);
siz1=size(data);

for j=1:siz1(1);
    for i=1:siz1(2);
        ij=i+(j-1)*siz1(2);
        datar(ij)=data(j,i);
    end
end


fid = fopen(filename,'wb');
fwrite(fid,datar,'float32','ieee-le');
fclose(fid);
end

