function K_vect = calc_k(t,K)
   
    K_vect = [ ppval(K(1),t), ppval(K(2),t), ppval(K(3),t) ];

end