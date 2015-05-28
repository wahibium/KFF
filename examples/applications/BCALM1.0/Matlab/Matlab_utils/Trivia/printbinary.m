function res=printbinary(input)
res=[];
input=uint64(input);
for(i=(1:64))
     
    bit=bitget(input,i);
    if (mod(i-1,8)==0)
        res=[' ' res];
    end
    if (bit ==1)
        res=['1' res ];
    else
         res=['0' res];
    end
    
    
end
