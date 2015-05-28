function TestFFTPoint(fftfile,temporalfile,x,y,z)


  dataF=my_extract2([fftfile],'/Ez_abs',4);
  dataT=my_extract2([temporalfile],'/Ez',4);
  deltaF=hdf5read([fftfile],'deltaF');
  fstart=hdf5read([fftfile],'fstart');
  fstop=hdf5read([fftfile],'fstop');
  freq=(fstart:deltaF:fstop);
  traceF=reshape(dataF(x,y,z,:),1,length(freq));
  dim=size(dataT);
  traceT=reshape(dataT(x,y,z,:),1,dim(4));
  traceT=[traceT zeros(1,128-dim(4))];
  ffttest=fft(traceT);
  ffttest=ffttest(1:length(traceF))
    figure
  hold on
  plot(freq,abs(traceF),'*');
figure
plot(traceT)

  
  
