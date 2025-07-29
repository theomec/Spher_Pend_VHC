function dRv = rhs(t,y,v,dv)
    
    y = reshape(y,4,4);
    
    v_calc  = from_t_to_vector4(t,v);
    dv_calc = from_t_to_vector4(t,dv);


    S = dv_calc * v_calc' - v_calc * dv_calc';
    dR = S * y;
    dRv = reshape(dR, 16, 1);
end