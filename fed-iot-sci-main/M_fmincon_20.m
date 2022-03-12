function [x] = fmin(x0,A,B,C,lb,ub,R,Budget,alpha)
MyValue=10e6;
options = optimoptions('fmincon','Algorithm','sqp', 'MaxFunctionEvaluations',MyValue);
function [f] = fun(x, A, B,R,alpha)
f = alpha/R*(1-x(1))*A(1)/x(1) + alpha/R*(1-x(2))*A(2)/x(2) + alpha/R*(1-x(3))*A(3)/x(3) + alpha/R*(1-x(4))*A(4)/x(4) + alpha/R*(1-x(5))*A(5)/x(5) + alpha/R*(1-x(6))*A(6)/x(6) + alpha/R*(1-x(7))*A(7)/x(7) + alpha/R*(1-x(8))*A(8)/x(8) + alpha/R*(1-x(9))*A(9)/x(9) + alpha/R*(1-x(10))*A(10)/x(10) + alpha/R*(1-x(11))*A(11)/x(11) + alpha/R*(1-x(12))*A(12)/x(12) + alpha/R*(1-x(13))*A(13)/x(13) + alpha/R*(1-x(14))*A(14)/x(14) + alpha/R*(1-x(15))*A(15)/x(15) + alpha/R*(1-x(16))*A(16)/x(16) + alpha/R*(1-x(17))*A(17)/x(17) + alpha/R*(1-x(18))*A(18)/x(18) + alpha/R*(1-x(19))*A(19)/x(19) + alpha/R*(1-x(20))*A(20)/x(20);
end
function [c, ceq] = nonlcon(x, B, C, R, Budget,alpha)
c = 2*C(1)*x(1)*x(1)-alpha/R*B(1)/x(1) + 2*C(2)*x(2)*x(2)-alpha/R*B(2)/x(2) + 2*C(3)*x(3)*x(3)-alpha/R*B(3)/x(3) + 2*C(4)*x(4)*x(4)-alpha/R*B(4)/x(4) + 2*C(5)*x(5)*x(5)-alpha/R*B(5)/x(5) + 2*C(6)*x(6)*x(6)-alpha/R*B(6)/x(6) + 2*C(7)*x(7)*x(7)-alpha/R*B(7)/x(7) + 2*C(8)*x(8)*x(8)-alpha/R*B(8)/x(8) + 2*C(9)*x(9)*x(9)-alpha/R*B(9)/x(9) + 2*C(10)*x(10)*x(10)-alpha/R*B(10)/x(10) + 2*C(11)*x(11)*x(11)-alpha/R*B(11)/x(11) + 2*C(12)*x(12)*x(12)-alpha/R*B(12)/x(12) + 2*C(13)*x(13)*x(13)-alpha/R*B(13)/x(13) + 2*C(14)*x(14)*x(14)-alpha/R*B(14)/x(14) + 2*C(15)*x(15)*x(15)-alpha/R*B(15)/x(15) + 2*C(16)*x(16)*x(16)-alpha/R*B(16)/x(16) + 2*C(17)*x(17)*x(17)-alpha/R*B(17)/x(17) + 2*C(18)*x(18)*x(18)-alpha/R*B(18)/x(18) + 2*C(19)*x(19)*x(19)-alpha/R*B(19)/x(19) + 2*C(20)*x(20)*x(20)-alpha/R*B(20)/x(20) - Budget;
ceq=[];
end
[x,fmin,exitflag,output] = fmincon(@(x)fun(x,A,B,R,alpha), x0, [], [], [], [], lb, ub, @(x)nonlcon(x, B, C, R, Budget,alpha), options)
end
