function I_main = calc_I_fcn_tr(X_starr_0, params)

    phi = -0.1 : 0.001 : 2*pi + 0.1;
    
    I_1 = zeros(1,length(phi));
    I_2 = zeros(1,length(phi));
    I_3 = zeros(1,length(phi));

    [ alpha_fcn, beta_fcn, gamma_fcn ] = s_abg_analytics(params);

    int_psi = @(z) beta_fcn(z) ./ alpha_fcn(z);
     
    for i = 1 : length(phi)
        
        I_1(i) = integral(int_psi, X_starr_0(1), phi(i), 'RelTol', 1e-8, 'AbsTol', 1e-12);
        
        I_2(i) = integral(int_psi, phi(i), X_starr_0(1), 'RelTol', 1e-8, 'AbsTol', 1e-12);
        
    end
    
    I_1_fcn = spline(phi, I_1);
    I_2_fcn = spline(phi, I_2);

    int_main = @(z) exp(-2 .* ppval(I_2_fcn, z)) .* 2 .* gamma_fcn(z) ./ alpha_fcn(z);


    for i = 1 : length(phi)

        I_3(i) = integral(int_main, X_starr_0(1), phi(i), 'RelTol', 1e-8, 'AbsTol', 1e-12);

    end

    I_3_fcn = spline(phi, I_3);

    I_main = @(q, dq) dq^2 - exp(-2 .* ppval(I_1_fcn, q)) * ( X_starr_0(2)^2 - ppval(I_3_fcn,q) );

    
    figure()
    plot(phi, I_1, '-r',LineWidth=1.5)
    hold on
    grid on
    plot(phi,ppval(I_1_fcn,phi),'--k',LineWidth=2.0)
    title("Integral 1")
    axis([-0.1 2*pi+0.1 -2e-3 2e-3])

    figure()
    plot(phi, I_2, '-r',LineWidth=1.5)
    hold on
    grid on
    plot(phi,ppval(I_2_fcn,phi),'--k',LineWidth=2.0)
    axis([-0.1 2*pi+0.1 -2e-3 2e-3])
    title("Integral 2")

    figure()
    plot(phi, I_3, '-r',LineWidth=1.5)
    hold on
    grid on
    plot(phi,ppval(I_3_fcn,phi),'--k',LineWidth=2.0)
    axis([-0.1 2*pi+0.1 -0.4 0.4])
    title("Integral 3")

end