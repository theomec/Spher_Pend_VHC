function xi = calc_tran_coord_tr(X, X_star_0, X_star_t, I_1_fcn, I_2_fcn, params)
  
    format long

    s = X(1);
    theta = X(2);
    ds = X(3);
    dtheta = X(4);

    if (length(X_star_0) > 2)
        disp("Wrong arguments")
    end

    s_0 = X_star_0(1);
    ds_0 = X_star_0(2);
    
    % Define the outer integral function
    [ alpha_fcn, beta_fcn, gamma_fcn ] = s_abg_analytics(params);
    
    
    final_int = @(q) exp(-2 .* ppval(I_2_fcn, q)) .* 2 .* gamma_fcn(q) ./ alpha_fcn(q);

    % Perform numerical integration
    I_1 = integral(final_int, s_0, s, 'RelTol', 1e-8, 'AbsTol', 1e-12);


    I = ds^2 - exp(-2 * ppval(I_1_fcn, s))  * (ds_0^2 - I_1);


    [ Q, dQ, ddQ ] = servo(s, params);

    y  =  theta -  Q(2);
    dy = dtheta - dQ(2) * ds;


    xi = [ I; y; dy ];

end