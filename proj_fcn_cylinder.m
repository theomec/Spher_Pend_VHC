function [ tau, X_star, min_dist_fin ] = proj_fcn_cylinder(t, t0, X_0, X, X_ref, dt, T_per, X_per)

% description:
% X_0 is current point
% X is trajectory array

% choose the interval [t1 t2]
lb = t0;
rb = t0 + dt;

if (rb > t(end))
    t_int = ((t >= t0) & ((t <= t(end)))) | ((t >= t(1)) & ((t <= mod(rb,T_per))));
elseif (lb < 0)
    t_int = ((t >= t0) & ((t <= rb))) | ((t >= mod(lb,T_per)) & ((t <= t(end))));
else
    t_int = ((t >= t0) & ((t <= rb)));
end

tt   =  t(t_int);
XX   =  X(:, t_int);

N = length(tt);
dist = zeros(1,length(tt));

for i = 1 : N
    dist(i) = abs(XX(1,i) - mod(X_0(1), X_per));
%     dist(i) = abs(XX(2,i) - X_0(2));
    
end

% find the exact value of \tau with optimization
[ min_dist, min_dist_idx ] = min(dist);

tau = tt(min_dist_idx);
X_star = from_t_to_vector4(tau, X_ref);
min_dist_fin = abs(X_0(2) - X_star(3));
% min_dist_fin = norm(from_t_to_vector2(tau, X_ref) - X_0);

% j_min_idx = find(t == tt(min_dist_idx));

% sv_fcn = @(ttt) norm( from_t_to_vector2(ttt, X_ref) - L(v' * (from_t_to_vector2(ttt, X_ref) - X_c) / (v' * v)));
% 
% N_max = length(t);
% 
% if ((j_min_idx > 2) && (j_min_idx < N_max-1)) 
%     
%     [ tau, froot ] = brent(sv_fcn, t(j_min_idx - 1), t(j_min_idx + 1), 1e-5);
% 
% else
% 
%     [ tau1, froot1 ] = brent(sv_fcn, t(N_max - 2), t(N_max), 1e-5);
%     [ tau2, froot2 ] = brent(sv_fcn, t(1), t(3), 1e-5);
%     
%     if(abs(froot1) <= abs(froot2))
%         tau = tau1;
%     else
%         tau = tau2;
%     end
% 
% end

% calculate the desired X^*(\tau) and distance d = dist(X_0,X^*(\tau))
% tau = tt(min_dist_idx);

% X_star = from_t_to_vector4(tau, X_ref);
% min_dist_fin = norm(from_t_to_vector2(tau, X_ref) - X_0);
% tau = mod(tau, T_per);


end