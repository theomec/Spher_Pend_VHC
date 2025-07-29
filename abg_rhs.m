function dds = abg_rhs(t,y,params)

    dds = zeros(2,1);
    
    s = y(1);
    ds = y(2);
    
    [ Q, dQ, ddQ ] = servo(s, params);
    
    [ M, C, G, B, B_perp ] = dynamics(Q,dQ,params);
    
    alpha =  B_perp * M * dQ;
    beta = B_perp * ( M * ddQ + C * dQ);
    gamma = B_perp * G;
    
    dds(1) = ds;
    dds(2) = (-1 ./ alpha) .* (beta .* ds.^2 + gamma);

end