function [ I_1_fcn, I_2_fcn, I_fcn ] = calc_I_fcn_4(X_starr_0,params)

    phi = -0.6 : 0.0001 : 0.6;
    
    I_1 = zeros(1,length(phi));
    I_2 = zeros(1,length(phi));
    I   = zeros(1,length(phi));

    [p_1, p_2, p_3, p_4] = params2p(params);
    L = params.servo_L;


    if (L == 0.0)
        disp('Error in servo!!!')
    end

    alpha = @(s) p_3 - L * p_2 .* cos(s) .* cos(s);  
    beta = @(s) p_2 * L .* sin(s) .* cos(s);
    gamma = @(s) - p_4 .* sin(s);

    int_psi = @(z) beta(z) ./ alpha(z);
%     
    for i = 1 : length(phi)
        
        I_1(i) = integral(int_psi, X_starr_0(1), phi(i), 'RelTol', 1e-8, 'AbsTol', 1e-12);
        
        I_2(i) = integral(int_psi, phi(i), X_starr_0(1), 'RelTol', 1e-8, 'AbsTol', 1e-12);
        
    end
    
    I_1_fcn = spline(phi, I_1);
    I_2_fcn = spline(phi, I_2);

    final_int = @(q) exp(-2 .* ppval(I_2_fcn, q)) .* 2 .* gamma(q) ./ alpha(q);

    syms dtheta real;

    for i = 1 : length(phi)

        II_1 = integral(final_int, X_starr_0(1), phi(i), 'RelTol', 1e-8, 'AbsTol', 1e-12);
        I(i) = exp(-2 * ppval(I_1_fcn, phi(i)))  * (X_starr_0(2)^2 - II_1);

    end

    I_fcn = spline(phi, I);
    
    figure()
    plot(phi, I_1, '-r',LineWidth=1.5)
    hold on
    plot(phi,ppval(I_1_fcn,phi),'--k',LineWidth=2.0)
    title("Integral 1")

    figure()
    plot(phi, I_2, '-r',LineWidth=1.5)
    hold on
    plot(phi,ppval(I_2_fcn,phi),'--k',LineWidth=2.0)
    title("Integral 2")

    figure()
    plot(phi, I, '-r',LineWidth=1.5)
    hold on
    plot(phi,ppval(I_fcn,phi),'--k',LineWidth=2.0)
    title("Integral")

end