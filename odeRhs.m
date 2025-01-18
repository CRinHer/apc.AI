function dxdt = odeRhs(t,x,model,u)

    A = model.A;
    B = model.B;

    dxdt = A*x + B*u;


end