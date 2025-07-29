function [ alpha, beta, gamma ] = plot_abg(s, params)

    [ Q, dQ, ddQ ] = servo(s, params);
    [ M, C, G, B, B_perp ] = dynamics(Q, dQ, params);
    
    alpha =  B_perp * M * dQ;
    beta = B_perp * ( M * ddQ + C * dQ);
    gamma = B_perp * G;

end
