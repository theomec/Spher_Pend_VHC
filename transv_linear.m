function [ A, B ] = transv_linear(t,y,E1,E2,E3,dE1,dE2,dE3,J_fcn,P,params)

    f = f_rhs(y, params);
    g = g_rhs(y, params);
    J = J_fcn(y(2),y(4));

    E  = [ from_t_to_vector4(t,E1), from_t_to_vector4(t,E2), from_t_to_vector4(t,E3)];
    dE = [ from_t_to_vector4(t,dE1), from_t_to_vector4(t,dE2), from_t_to_vector4(t,dE3)];


%     A = E' * ( f' * P * f * J - f * f' * P * J - f * f' * J' * P ) * E ./ ( f' * P * f ) + dE' * E;
%  
%     B = E' * (eye(4) - (f * f' * P) / ( f' * P * f )) * g;

%     A = E' * ( f' * f * J - f * f' * J - f * f' * J' ) * E ./ ( f' * f ) + dE' * E;
%  
%     B = E' * (eye(4) - (f * f') / ( f' * f )) * g;

    A = dE' * E + E' * J * E;
    B = E' * g;

end