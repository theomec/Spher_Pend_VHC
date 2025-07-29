function [ Q, dQ, ddQ ] = s_servo(s, params)

    Q  = [ s; params.p ];
    dQ = [ 1; params.dp ];
    ddQ =[ 0; params.ddp ];

end