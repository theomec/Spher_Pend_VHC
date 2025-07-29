function V_calc = from_t_to_vector4(t,V)
    
    V_calc = [ ppval(V(1),t); ppval(V(2),t); ppval(V(3),t); ppval(V(4),t) ];

end