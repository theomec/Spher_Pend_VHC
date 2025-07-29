function ddq = true_system(t, y, pu, T, params)

    ddq = zeros(4,1);
    tt = mod(t,T);
    u = ppval(pu,tt);

    q = reshape(y(1:2),2,1);
    dq = reshape(y(3:4),2,1);

    [ M, C, G, B, B_perp ] = dynamics(q, dq, params);

    ddq(1:2) = dq;
    ddq(3:4) = M \ (- C * dq - G + B * u);

end