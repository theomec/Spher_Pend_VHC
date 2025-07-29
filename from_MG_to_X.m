function [ X_total, dX_total ] = from_MG_to_X(t, s, ds, dds, params)
    
    X = zeros(2, length(t));
    dX = zeros(2, length(t));
    ddX = zeros(2, length(t));
    
    for i = 1 : length(t) 
        [ Q, dQ, ddQ ] = servo(ppval(s,t(i)), params);
        X(:,i)   = Q;
        dX(:,i)  = dQ * ppval(ds,t(i));
        ddX(:,i) = ( ddQ .* ppval(ds,t(i)).^2 + dQ .* ppval(dds,t(i)) );
    end

    X_total = [ X; dX ];
    dX_total = [ dX; ddX ];

end