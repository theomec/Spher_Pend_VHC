function ddq = f_rhs(y,params)

    ddq = zeros(4,1);

    q  = [ y(1); y(2) ];
    dq = [ y(3); y(4) ];

    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);

    ddq(1:2) = dq;
    ddq(3:4) = M \ (- C * dq - G);
end