%% Creates some plots of the returened field for Test purpose only.

function MyLittlePlotOut(data)



   
  
 dimT=size(data)
      p1=figure
      gca
            p2=figure
      gca
 for t=(1:dimT(4))
 
    PlotSlice(p1,data(:,:,1,t),dimT(1),dimT(2))
title(['t= ' num2str(t)]);
    PlotSlice(p2,data(:,:,1,t),dimT(1),dimT(2))

    title(['t= ' num2str(t)]);
    
    pause()
     
 end



end
