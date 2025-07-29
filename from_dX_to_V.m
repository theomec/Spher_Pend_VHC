function V = from_dX_to_V(t, dX, P)

    V = zeros(4, length(t));
    
    for i = 1 : length(t) 
        V(:,i) = (P * dX(:,i)) / norm(P * dX(:,i));
    end
end