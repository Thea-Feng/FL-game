
N = input("Please enter N")
Name = "M_fmincon_" + N +".m"
N = eval(N)
print(Name)
f = open(Name,'w')
f.writelines("function [x] = fmin(x0,A,B,C,lb,ub,R,Budget,alpha)\n")
f.writelines("MyValue=10e6;\n")
f.writelines("options = optimoptions('fmincon','Algorithm','sqp', 'MaxFunctionEvaluations',MyValue);\n")
f.writelines("function [f] = fun(x, A, B,R,alpha)\n")
f.write("f = ")
for i in range(1, N):
    f.write("alpha/R*(1-x({}))*A({})/x({}) + ".format(i,i,i))
f.write("alpha/R*(1-x({}))*A({})/x({});\n".format(N,N,N))
f.writelines("end\n")

f.writelines("function [c, ceq] = nonlcon(x, B, C, R, Budget,alpha)\n")
f.write("c = ")
for i in range(1, N):
    f.write("2*C({})*x({})*x({})-alpha/R*B({})/x({}) + ".format(i,i,i,i,i))
f.write("2*C({})*x({})*x({})-alpha/R*B({})/x({}) - Budget;\n".format(N,N,N,N,N))
f.writelines("ceq=[];\n")
f.writelines("end\n")

f.writelines("[x,fmin,exitflag,output] = fmincon(@(x)fun(x,A,B,R,alpha), x0, [], [], [], [], lb, ub, @(x)nonlcon(x, B, C, R, Budget,alpha), options)\n")
f.writelines("end\n")


f.close()
print("done")

