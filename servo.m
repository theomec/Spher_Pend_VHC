function [ Q, dQ, ddQ ] = servo(s, params)

    Q  = [ s;   params.servo_0 + params.servo_1 * s + params.servo_2 / 2 * s^2 + params.servo_3 * cos(s) + params.servo_4 * sin(s) ];
    dQ = [ 1;   params.servo_1 + params.servo_2 * s - params.servo_3 * sin(s) + params.servo_4 * cos(s) ];
    ddQ =[ 0;   params.servo_2 - params.servo_3 * cos(s) - params.servo_4 * sin(s) ];

%     Q  = [ params.servo_0 + params.servo_1 * s + params.servo_2 / 2 * s^2 + params.servo_3 * cos(s) + params.servo_4 * sin(s); s + params.servo_6 ];
%     dQ = [ params.servo_1 + params.servo_2 * s - params.servo_3 * sin(s) + params.servo_4 * cos(s); 1 ];
%     ddQ =[ params.servo_2 - params.servo_3 * cos(s) - params.servo_4 * sin(s); 0 ];

end