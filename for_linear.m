function ddq = for_linear(y,params)

    q  = [ y(1); y(2) ];
    dq = [ y(3); y(4) ];
    
    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);

    ddq = [ y(3); y(4) ];
    ddq = [ ddq; M \ (- C * dq - G) ];

end