function [ g_y_fcn, g_dy_fcn, g_v_fcn ] = g_fcn(params)
    
    syms s ds dds y dy ddy real
    
    [ Q, dQ, ddQ ] = servo(s, params);

    q   = [   s;   y + Q(2) ];
    dq  = [  ds;  dy + dQ(2) * ds ];
    ddq = [ dds; ddy + dQ(2) * dds + ddQ(2) * ds^2];
    
    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);
    
    N = simplify(- B_perp * ( M * ddq + C * dq + G));
    
    
    g_y  = simplify(subs(diff(N, y),  [ y dy ddy ], [ 0 0 0 ]));
    
    g_dy = simplify(subs(diff(N, dy), [ y dy ddy ], [ 0 0 0 ]));
    
    g_v  = simplify(subs(diff(N, ddy),[ y dy ddy ], [ 0 0 0 ]));

    g_y_fcn  = matlabFunction(g_y);
    g_dy_fcn = matlabFunction(g_dy);
    g_v_fcn  = matlabFunction(g_v);

end