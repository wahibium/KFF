function Out=PlotSource(t,p)
    t=t-1; % Alling with C code
    Out=sin(p(2)*t).*exp(-((t-p(3))/p(4)).^2);