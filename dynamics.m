function [ M, C, G, B, B_perp ] = dynamics(q, dq, params)

    [ a, b_0, b_1, c, d, g, R ] = params2p(params);

    M = [ a,                       b_0 + b_1 * cos(q(2));
          b_0 + b_1 * cos(q(2)),   c      ];
    
    C = [ 0,  -d * sin(q(2)) * dq(2);
          0,   0];
    
    G = [ 0; d * g * sin(q(2)) ];

    B = [ -1/R; 1 ];

    B_perp = [ 1, 1/R ];

end