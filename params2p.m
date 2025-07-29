function [ a, b_0, b_1, c, d, g, R ] = params2p(params)

    a = params.M_m + 1/(params.R)^2 * (params.I + params.i);
    b_0 = - params.i / params.R;
    b_1 = params.m * params.rho;
    d = params.m * params.rho;
    c = params.m * params.rho^2 + params.i;
    g = params.g;
    R = params.R;

end