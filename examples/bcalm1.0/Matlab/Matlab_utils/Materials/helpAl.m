hbar=1.05457148*1e-34 
c=3e8;
e=1.6e-19;
close all
fid=fopen([optpath 'Al'],'w');
eV=(0.6:0.01:2);

optpath=['/home/pwahl/Documents/Phd/Metal_Optical_Constants/']
data=load([optpath 'Altemp']);
eV=data(:,1);
espr=-data(:,2);
epsi=data(:,3);
n=data(:,4);
k=data(:,5);
nki=find(eV>0.651,1,'first')
n(nki:end)=real(sqrt(espr(nki:end)-1i*epsi((nki:end))))
k(nki:end)=-imag(sqrt(espr(nki:end)-1i*epsi((nki:end)))) 
lambda=2*pi*c*hbar./(e*eV)*1e6;

datan(:,1)=eV;
datan(:,2)=zeros(1,length(eV));
datan(:,3)=lambda;
datan(:,4)=n;
datan(:,5)=k;
datan=double(datan);
datan=flipud(datan);
for cnt=1:length(datan(:,1))
    for cnt2=1:length(datan(1,:))
fprintf(fid,[num2str(datan(cnt,cnt2)) '\t']);
    end
fprintf(fid,'\n');
end

