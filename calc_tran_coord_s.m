function xi = calc_tran_coord_s(X, X_star, I_main, params, X_per)
  
    format long

    s = mod(X(1),X_per);
%     s = X(1);
    theta = X(2);
    ds = X(3);
    dtheta = X(4);

%     if (length(X_star_0) > 2)
%         disp("Wrong arguments")
%     end
    
    I = I_main(s,ds);


    [ Q, dQ, ddQ ] = servo(s, params);
%     [ Q, dQ, ddQ ] = servo(X_star(1), params);

    y  =  theta -  Q(2);
    dy = dtheta - dQ(2) * ds;

%     y  =  theta -  Q(2);
%     dy = dtheta - dQ(2) * X_star(3);

%     y  = X_star(2) -  Q(2);
%     dy = X_star(4) - dQ(2) * ds;

    xi = [ I; y; dy ];

end