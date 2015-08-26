function [data,mplot,POSITION]= GetModE(mplot,LinearGrid,hdf5_file)

fieldEr=[{'Ex_real'},{'Ey_real'},{'Ez_real'}];
fieldEi=[{'Ex_imag'},{'Ey_imag'},{'Ez_imag'}];

fieldHr=[{'Hx_real'},{'Hy_real'},{'Hz_real'}];
fieldHi=[{'Hx_imag'},{'Hy_imag'},{'Hz_imag'}];

mod=mplot;

switch mplot.field
    
    case 'ME2'
        
        for cnt=1:length(fieldEr)
            
            try
                mod.field=fieldEr{cnt};
                [Er,mplot,POSITION]=JustGetData(mod,hdf5_file);
                mod.field=fieldEi{cnt};
                [Ec,mplot,POSITION]=JustGetData(mod,hdf5_file);
                
            catch
                Ec=0;
                Er=0;
                warning(['Field ' fieldEr{cnt} ' is not in the output and assumed zero'])
            end
            
            if cnt==1
                data=zeros(size(Ec));
            end
            
            data=data+abs((Er+1i*Ec)).^2;
            
            
            
            
        end
        
        
    case 'MH2'
        
        for cnt=1:length(fieldHr)
            try
                mod.field=fieldHr{cnt};
                [Hr,mplot,POSITION]=JustGetData(mod,hdf5_file);
                mod.field=fieldHi{cnt};
                [Hc,mplot,POSITION]=JustGetData(mod,hdf5_file);
                
            catch
                Hc=0;
                Hr=0;
                warning(['Field ' fieldHr{cnt} ' is not in the output and assumed zero'])
            end
            if cnt==1
                data=zeros(size(Hc));
            end
            
            data=data+abs((Hr+1i*Hc)).^2;
            
            
            
            
        end
        
    otherwise
        error(sprintf('Wrong field name %s',mplot.field));
        
end


end