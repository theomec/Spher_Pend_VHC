function [ alpha, beta, gamma ] = s_plot_abg(s, params)

    [ Q, dQ, ddQ ] = s_servo(s, params);
    [ M, C, G, B, B_perp ] = s_dynamics(Q, dQ, params);
    
    alpha =  B_perp * M * dQ;
    beta = B_perp * ( M * ddQ + C * dQ);
    gamma = B_perp * G;

end
