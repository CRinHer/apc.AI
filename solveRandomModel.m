function y_track = solveRandomModel

%
nx = 4;
nu = 4;
ny = 4;

%Generate a random linear model
model = rss(nx,ny,nu);
tau = eigs(model.A);
tau = max(abs(tau));
tf  = 10*tau;

y_track = struct();
%Choose a nominal steady-state input u_ss = zeros
%Steady-state solution is always x_ss = zeros
%Set the initial condition to x_ss
%Choose one specific input i , and step it from u_ss(i) to a new value
%Keep all other u_ss(j) the same.
%Solve the ODEs with this u vector constant
x0 = zeros(nx,1);
u = zeros(nu,1);
ssInOutMatrix = zeros(ny,nu);
for i = 1:nu
    u(i) = 1;
    fun = @(t,x) odeRhs(t,x,model,u);
    [t,x] = ode45(fun,[0 tf],x0);
    u(i) = 0;
    
    C = model.C;
    D = model.D;

    y = (C*x' + D*u)';

    ssInOutMatrix(:,i) = y(end,:)';
    fieldName = sprintf('Field_%d', i);
    y_track.(fieldName) = y;

    %figure;
    %hold on;
    %plot(t,y(:,1),t,y(:,2),t,y(:,3),t,y(:,4))
end
