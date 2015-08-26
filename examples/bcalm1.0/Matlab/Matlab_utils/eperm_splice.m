function [dddx] = eperm_encap(dddx, POS, NUMLAYERS)
% function  [dx] = eperm_encap(s, cutinx , xdim, dddx, POS, NUMLAYERS)


for itr = 1:NUMLAYERS
    dddx = [dddx(1:POS-1) dddx(POS)/2 dddx(POS)/2 dddx(POS+1)/2 dddx(POS+1)/2 dddx(POS+2:end)];
    POS = POS+1;
end
end