function ddq = g_rhs(y,params) 

    ddq = zeros(4,1);

    q  = [ y(1); y(2) ];
    dq = [ y(3); y(4) ];

    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);

    ddq(1:2) = zeros(2,1);
    ddq(3:4) = M \ B;

end