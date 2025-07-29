function U_fcn = u2v_transformation(params)
    
    syms u v z real
    syms x theta theta_dot x_dot real

    q  = [ x; theta ];
    dq = [ x_dot; theta_dot ];

    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);

    ddX_ddTheta = simplify(M \ ( - C * dq - G + B * u));
    
    g = simplify([ diff(ddX_ddTheta(1), u); diff(ddX_ddTheta(2), u) ]);
    f = simplify(ddX_ddTheta - g * u);

    [ Q, dQ, ddQ ] = servo(x,params);

    V = simplify(f(2) + g(2) * u - dQ(2) * (f(1) + g(1) * u) - ddQ(2) * x_dot^2);
    v_2 = simplify(diff(V, u));
    v_1 = simplify(V - v_2 * u);

    U = simplify(v / v_2 - z * v_1 / v_2);
    U_fcn = matlabFunction(U);
    
end