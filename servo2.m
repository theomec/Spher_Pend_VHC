function [ Q, dQ, ddQ ] = servo2(s, params)

    Q  = [ s;   s + params.servo_0 ];
    dQ = [ 1;   1 ];
    ddQ =[ 0;   0 ];

end