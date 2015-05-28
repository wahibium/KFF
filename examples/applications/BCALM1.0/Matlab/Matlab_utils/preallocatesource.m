         

function g=preallocatesource(g,start,stop)

          maxnsources=prod(stop-start+1);
          if isfield(g,'source')
          
         g.source=[g.source, zeros(18,maxnsources)];
          else
              g.source= zeros(18,maxnsources);
          end
          
          
end