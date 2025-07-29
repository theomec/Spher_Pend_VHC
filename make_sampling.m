function t = make_sampling(t_gen,t_spec,T_per)

    t = 0 : t_gen : T_per;
    
%     t1 = 0.15:t_spec:0.26;
%     t2 = 0.58:t_spec:0.67;
    t1 = [];
    t2 = [];
%     
    t = [t,t1,t2];

    bs = sort(t);
    t = bs([true, diff(bs) ~= 0]);
end