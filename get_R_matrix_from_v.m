function R = get_R_matrix_from_v(x,y)

    u = x / norm(x);
    
    v = y - u' * y * u;
    
    v=v/norm(v);
    
    cost=x'*y/norm(x)/norm(y);
    
    sint=sqrt(1-cost^2);
    
    R = eye(length(x)) - u * u' - v * v' + [u v] * [cost -sint; sint cost] *[u v]';
    
    if ( (norm(R * R' - eye(length(x)), 2)) > 1e-12 )
        disp('error 1 in R')
    end

    if ( det(R)-1 > 1e-12 )
        disp('error 2 in R')
    end

end