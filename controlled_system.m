function z = controlled_system(y, t, t0, X, X_ref, pu, KKK, T_per, dt, X_per, U_fcn, I_main, params)

    ddq = zeros(11, 1);

    q = reshape(y(1:2), 2, 1);
    dq = reshape(y(3:4), 2, 1);

    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);

    X_0 = [ q; dq ];

    [ tau, X_star, min_dist ] = proj_fcn_cylinder(t, t0, [ X_0(1); X_0(3) ], X(1:2:end,:), X_ref, dt, T_per, X_per);

%     xi = calc_tran_coord_s(X_0, X_star_0, X_star, I_main, params);
    xi = calc_tran_coord_s(X_0, X_star, I_main, params, X_per);

    u_transv = calc_k(tau, KKK) * xi;
    
%     u_fb = U_fcn(q(2), dq(2), u_transv, q(1), dq(1), 0);
    u_fb = U_fcn(X_star(2), X_star(4), u_transv, X_star(1), X_star(3), 1);

    u_nom = ppval(pu, tau);
    u = 0 + u_fb;

    ddq(1:2) = dq;
    ddq(3:4) = M \ (- C * dq - G + B * u);

    ddq(5) = tau;
    ddq(6) = u_fb;
    ddq(7) = u_nom;
    ddq(8) = min_dist;
    ddq(9:11) = xi;
    
    z = ddq;
end