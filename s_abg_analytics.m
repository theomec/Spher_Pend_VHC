function [ alpha_fcn, beta_fcn, gamma_fcn ] = s_abg_analytics(params)

    syms s real    
    [ Q, dQ, ddQ ] = servo(s, params);
    
    [ M, C, G, B, B_perp ] = dynamics(Q, dQ, params);

    alpha = simplify(B_perp * M * dQ);
    beta  = simplify(B_perp * ( M * ddQ + C * dQ));
    gamma = simplify(B_perp * G);

    alpha_fcn = matlabFunction(alpha);
    beta_fcn  = matlabFunction(beta);
    gamma_fcn = matlabFunction(gamma);
    
end