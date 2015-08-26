function g=AddOutputZones(g,output)
 
 output=SetPosition(g,output);
 if (isfield(output,'calibrate')&& (output.calibrate==1))
    if (isfield(g,'calibrate')&& (g.calibrate==1))
        
        output.name=[output.name 'Cal'];
        
    end
 end
 if ~isfield(g,'outputzones')
     g.outputzones=[];
 end
 
 if ~isfield(output,'deltaT')
     B=output.foutstop*1.1;
     dt=g.info.dt;
     output.deltaT=floor(1/(2*B*dt))
 end
 
 if ~isfield(output,'average')
     output.average=0;
 end
 
 g.outputzones=[g.outputzones;output];
     
     
     
 
 
end

 