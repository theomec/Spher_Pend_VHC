function U = find_u_eq(s,ds,dds,params)

    [ Q, dQ, ddQ ] = servo(s, params);
    
    [ M, C, G, B, B_perp ] = dynamics(Q, dQ, params);

    B_plus = [ -params.R/2 1/2 ];

    if (abs(B_plus * B - 1) > 1e-6)
        disp('incorrect B_plus')
    end
    
    U = B_plus * (M * ( ddQ .* ds.^2 + dQ .* dds ) + C * dQ .* ds.^2 + G);

end
