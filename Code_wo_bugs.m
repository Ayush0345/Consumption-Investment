%%  Initial Variable and Parameter Setup and Preallocation

%%% Model Parameters
beta=0.96;
gamma=1.3;
r=0.04;

Y_n=5;                  %   Number of income gridpoints
sd=Y_n/2-0.5;           %   Max number of standard deviations away from steady state for the income process.
mu = 0;                 %   Mean of the Income process
sigma = 0.04;           %   Scalar Parameter
rho   = 0.9;            %   Persistence Parameter (1)
s = sd/(Y_n-1);         %   Step size for the income grid

%%% Income Grid Setup

% Discretize AR(1) by Rouwenhorst Method
[P, Y] = rouwen(rho, mu, sigma, Y_n); %(2)


%%% Asset Grid Setup
a_n=1000;               %   Number of asset gridpoints
a_max=4*exp(Y(Y_n));    %   Arbitrarily high number for the max of assets

% The minimum is set to be the negative of the present value of a lifetime 
% income stream, received from the next period and on, that is always 
% realized as the lowest possible income draw. Such a present value of the 
% stream would be,

%  exp(y1)/(1+r) + exp(y1)/(1+r)^2 + exp(y1)/(1+r)^3 + ...

%   = exp(y1)/(1+r)[1 + 1/(1+r) + 1/(1+r)^2 + ...] 

%   = exp(y1)/(1+r)[1/(1-(1/(1+r)))]

%   = exp(y1)/r

A=linspace(-exp(Y(1))/r,a_max,a_n)';    %   Asset Grid Discretization

%% Calculations that do not need to be repeated

%%%%%   We will calculate the period utility here for all possible choices
%%%%%   of assets next period at each gripoint.

c=zeros(a_n,Y_n,a_n);        %   Note that this is a three dimensional matrix, which is possible in Matlab. (3) setting c instead of c_choice because we dont use c_choice anywhere for recovery of consumption
utility=c;                   %   Preallocation

for ap=1:a_n
     c(:,:,ap)=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-repmat(A(ap),a_n,Y_n);
end

c(c<0)=0;  

if gamma==1
    utility=log(c);
else
    utility(c==0)=-inf;
    utility(c>0)=c(c>0).^(1-gamma)/(1-gamma);
end

tol=10^(-9);            %   Maximum error tolerance       
maxits=10^4;            %   Maximum number of iterations


V0 = repmat(utility(a_n/2,:,1), a_n, 1);     %   Initial Guess of the Value Function (bug)
V1=V0;                                       %   Preallocation of the updated value function
c=V1;                                        %   Preallocation of the consumption policy function
a_prime = (1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-c;                                 %  (4)  Preallocation of the asset choice policy function based on the budget constraint

count=0;
dif=0;

tic
V_candidate = ones(1,a_n);       % (5) Preallocating converged value function
while dif>tol && count<maxits
    for y=1:Y_n
        for ap=1:a_n
            if utility(:,y,ap) <=0  % (6) Adding an if statement to let the value function converge to the correct guess
                  V_candidate(:,ap) = -inf;
            else
            V_candidate(:,ap)=utility(:,y,ap)+beta*repmat(V0(ap,:),a_n,1)*P(y,:)';
           end
        [V1(:,y),a_prime(:,y)]=max(V_candidate);  % (7) Removing transpose from max(V_candidate')
       end
    dif=max(abs(V0-V1));
    count=count+1;
    V0=V1;  % (8) Replace V0=V_candidate to V0=V1
   end
end
toc

%%  Recovery of Consumption Policy Function


% (9) Create a loop to recover preallocated consumption policy function

for ap=1:a_n
    for y=1:Y_n
     c(:,:,ap)=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-a_prime(ap,y);
    end
end


figure(1)
subplot(3,1,1)
plot(A,V1(:,1),A,V1(:,Y_n/2+0.5),A,V1(:,Y_n))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Value')
title('Value Function')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

subplot(3,1,2)
plot(A,c(:,1),A,c(:,Y_n/2+0.5),A,c(:,Y_n))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Consumption')
title('Consumption')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

subplot(3,1,3)
plot(A,a_prime(:,1),A,a_prime(:,Y_n/2+0.5),A,a_prime(:,Y_n))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Assets')
title('Optimal Savings')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')


%%  Simulations

sims=1000;
y_sim=simulate(dtmc(P),sims); 
a_index=1;

for t=1:sims
    c_sim(t)=(1+r)*A(t)+exp(Y(y_sim(t)))-a_prime(1,y_sim(t));  % (10) Index exceeds the number of array elements. Index must not exceed 1. Fix this #########
    a_index(t+1)=a_prime(t,y_sim(t));
    a_sim(t+1)=a_prime(t,y_sim(t));
end

figure(2)
subplot(3,1,1)
plot(sims/2+1:sims,exp(Y(y_sim(sims/2+1:sims))),sims/2+1:sims,ones(sims/2,1))
xlabel('Time')
ylabel('Income')

subplot(3,1,2)
plot(sims/2+1:sims,c_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Consumption')

subplot(3,1,3)
plot(sims/2+1:sims,a_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Assets')

%% (d) Correlogram between Simulated income and consumption series upto 4 lags

[clgm,lags] = xcorr(y_sim,c_sim,4);

%% (e) Plotting the Correlogram
figure(3)
plot(lags,clgm);
xlabel("lags")
ylabel("Correlation Vector")
title("Correlogram between Simulated Income and Consumption Series")
legend("Correlation at 4 lags")

%% No Borrowing

%%  Initial Variable and Parameter Setup and Preallocation

%%% Model Parameters
beta=0.96;
gamma=1.3;
r=0.04;

Y_n=5;                  %   Number of income gridpoints
sd=Y_n/2-0.5;           %   Max number of standard deviations away from steady state for the income process.
mu = 0;                 %   Mean of the Income process
sigma = 0.04;           %   Scalar Parameter
rho   = 0.9;            %   Persistence Parameter (1)
s = sd/(Y_n-1);         %   Step size for the income grid

%%% Income Grid Setup

% Discretize AR(1) by Rouwenhorst Method
[P, Y] = rouwen(rho, mu, sigma, Y_n); %(2)


%%% Asset Grid Setup
a_n=1000;               %   Number of asset gridpoints
a_max=4*exp(Y(Y_n));    %   Arbitrarily high number for the max of assets

% The minimum is set to be the negative of the present value of a lifetime 
% income stream, received from the next period and on, that is always 
% realized as the lowest possible income draw. Such a present value of the 
% stream would be,

%  exp(y1)/(1+r) + exp(y1)/(1+r)^2 + exp(y1)/(1+r)^3 + ...

%   = exp(y1)/(1+r)[1 + 1/(1+r) + 1/(1+r)^2 + ...] 

%   = exp(y1)/(1+r)[1/(1-(1/(1+r)))]

%   = exp(y1)/r

A=linspace(0,a_max,a_n)';    %   Asset Grid Discretization

%% Calculations that do not need to be repeated

%%%%%   We will calculate the period utility here for all possible choices
%%%%%   of assets next period at each gripoint.

c=zeros(a_n,Y_n,a_n);        %   Note that this is a three dimensional matrix, which is possible in Matlab. (3) setting c instead of c_choice because we dont use c_choice anywhere for recovery of consumption
utility=c;                   %   Preallocation

for ap=1:a_n
     c(:,:,ap)=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-repmat(A(ap),a_n,Y_n);
end

c(c<0)=0;  

if gamma==1
    utility=log(c);
else
    utility(c==0)=-inf;
    utility(c>0)=c(c>0).^(1-gamma)/(1-gamma);
end

tol=10^(-9);            %   Maximum error tolerance       
maxits=10^4;            %   Maximum number of iterations


V0 = repmat(utility(a_n/2,:,1), a_n, 1);     %   Initial Guess of the Value Function (bug)
V1=V0;                                       %   Preallocation of the updated value function
c=V1;                                        %   Preallocation of the consumption policy function
a_prime = (1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-c;                                 %  (4)  Preallocation of the asset choice policy function based on the budget constraint

count=0;
dif=0;

tic
V_candidate = ones(1,a_n);       % (5) Preallocating converged value function
while dif>tol && count<maxits
    for y=1:Y_n
        for ap=1:a_n
            if utility(:,y,ap) <=0  % (6) Adding an if statement to let the value function converge to the correct guess
                  V_candidate(:,ap) = -inf;
            else
            V_candidate(:,ap)=utility(:,y,ap)+beta*repmat(V0(ap,:),a_n,1)*P(y,:)';
           end
        [V1(:,y),a_prime(:,y)]=max(V_candidate);  % (7) Removing transpose from max(V_candidate')
       end
    dif=max(abs(V0-V1));
    count=count+1;
    V0=V1;  % (8) Replace V0=V_candidate to V0=V1
   end
end
toc

%%  Recovery of Consumption Policy Function


% (9) Create a loop to recover preallocated consumption policy function

for ap=1:a_n
    for y=1:Y_n
     c(:,:,ap)=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-a_prime(ap,y);
    end
end


figure(4)
subplot(3,1,1)
plot(A,V1(:,1),A,V1(:,3),A,V1(:,5))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Value')
title('Value Function')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

subplot(3,1,2)
plot(A,c(:,1),A,c(:,3),A,c(:,5))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Consumption')
title('Consumption')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

subplot(3,1,3)
plot(A,a_prime(:,1),A,a_prime(:,3),A,a_prime(:,5))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Assets')
title('Optimal Savings')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')


%%  Simulations

sims=1000;
y_sim=simulate(dtmc(P),sims-1); 
a_index=1;

for t=1:sims
    c_sim(t)=(1+r)*A(t)+exp(Y(y_sim(t)))-a_prime(1,y_sim(t));  % (10) Index exceeds the number of array elements. Index must not exceed 1. Fix this #########
    a_index(t+1)=a_prime(t,y_sim(t));
    a_sim(t+1)=a_prime(t,y_sim(t));
end

figure(5)
subplot(3,1,1)
plot(sims/2+1:sims,exp(Y(y_sim(sims/2+1:sims))),sims/2+1:sims,ones(sims/2,1))
xlabel('Time')
ylabel('Income')

subplot(3,1,2)
plot(sims/2+1:sims,c_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Consumption')

subplot(3,1,3)
plot(sims/2+1:sims,a_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Assets')

%% (d) Correlogram between Simulated income and consumption series upto 4 lags

[clgm,lags] = xcorr(y_sim,c_sim,4);

%% (e) Plotting the Correlogram
figure(6)
plot(lags,clgm);
xlabel("lags")
ylabel("Correlation Vector")
title("Correlogram between Simulated Income and Consumption Series")
legend("Correlation at 4 lags")


%% High RRA (Sigma=0.08)

%%  Initial Variable and Parameter Setup and Preallocation

%%% Model Parameters
beta=0.96;
gamma=1.3;
r=0.04;

Y_n=5;                  %   Number of income gridpoints
sd=Y_n/2-0.5;           %   Max number of standard deviations away from steady state for the income process.
mu = 0;                 %   Mean of the Income process
sigma = 0.08;           %   Scalar Parameter
rho   = 0.9;            %   Persistence Parameter (1)
s = sd/(Y_n-1);         %   Step size for the income grid

%%% Income Grid Setup

% Discretize AR(1) by Rouwenhorst Method
[P, Y] = rouwen(rho, mu, sigma, Y_n); %(2)


%%% Asset Grid Setup
a_n=1000;               %   Number of asset gridpoints
a_max=4*exp(Y(Y_n));    %   Arbitrarily high number for the max of assets

% The minimum is set to be the negative of the present value of a lifetime 
% income stream, received from the next period and on, that is always 
% realized as the lowest possible income draw. Such a present value of the 
% stream would be,

%  exp(y1)/(1+r) + exp(y1)/(1+r)^2 + exp(y1)/(1+r)^3 + ...

%   = exp(y1)/(1+r)[1 + 1/(1+r) + 1/(1+r)^2 + ...] 

%   = exp(y1)/(1+r)[1/(1-(1/(1+r)))]

%   = exp(y1)/r

A=linspace(-exp(Y(1))/r,a_max,a_n)';    %   Asset Grid Discretization

%% Calculations that do not need to be repeated

%%%%%   We will calculate the period utility here for all possible choices
%%%%%   of assets next period at each gripoint.

c=zeros(a_n,Y_n,a_n);        %   Note that this is a three dimensional matrix, which is possible in Matlab. (3) setting c instead of c_choice because we dont use c_choice anywhere for recovery of consumption
utility=c;                   %   Preallocation

for ap=1:a_n
     c(:,:,ap)=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-repmat(A(ap),a_n,Y_n);
end

c(c<0)=0;  

if gamma==1
    utility=log(c);
else
    utility(c==0)=-inf;
    utility(c>0)=c(c>0).^(1-gamma)/(1-gamma);
end

tol=10^(-9);            %   Maximum error tolerance       
maxits=10^4;            %   Maximum number of iterations


V0 = repmat(utility(a_n/2,:,1), a_n, 1);     %   Initial Guess of the Value Function (bug)
V1=V0;                                       %   Preallocation of the updated value function
c=V1;                                        %   Preallocation of the consumption policy function
a_prime = (1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-c;                                 %  (4)  Preallocation of the asset choice policy function based on the budget constraint

count=0;
dif=0;

tic
V_candidate = ones(1,a_n);       % (5) Preallocating converged value function
while dif>tol && count<maxits
    for y=1:Y_n
        for ap=1:a_n
            if utility(:,y,ap) <=0  % (6) Adding an if statement to let the value function converge to the correct guess
                  V_candidate(:,ap) = -inf;
            else
            V_candidate(:,ap)=utility(:,y,ap)+beta*repmat(V0(ap,:),a_n,1)*P(y,:)';
           end
        [V1(:,y),a_prime(:,y)]=max(V_candidate);  % (7) Removing transpose from max(V_candidate')
       end
    dif=max(abs(V0-V1));
    count=count+1;
    V0=V1;  % (8) Replace V0=V_candidate to V0=V1
   end
end
toc

%%  Recovery of Consumption Policy Function


% (9) Create a loop to recover preallocated consumption policy function

for ap=1:a_n
    for y=1:Y_n
     c(:,:,ap)=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-a_prime(ap,y);
    end
end


figure(7)
subplot(3,1,1)
plot(A,V1(:,1),A,V1(:,3),A,V1(:,5))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Value')
title('Value Function')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

subplot(3,1,2)
plot(A,c(:,1),A,c(:,3),A,c(:,5))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Consumption')
title('Consumption')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

subplot(3,1,3)
plot(A,a_prime(:,1),A,a_prime(:,3),A,a_prime(:,5))
xlim([A(1) A(a_n)])
xlabel('Assets')
ylabel('Assets')
title('Optimal Savings')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')


%%  Simulations

sims=1000;
y_sim=simulate(dtmc(P),sims-1); 
a_index=1;

for t=1:sims
    c_sim(t)=(1+r)*A(t)+exp(Y(y_sim(t)))-a_prime(1,y_sim(t));  % (10) Index exceeds the number of array elements. Index must not exceed 1. Fix this #########
    a_index(t+1)=a_prime(t,y_sim(t));
    a_sim(t+1)=a_prime(t,y_sim(t));
end

figure(8)
subplot(3,1,1)
plot(sims/2+1:sims,exp(Y(y_sim(sims/2+1:sims))),sims/2+1:sims,ones(sims/2,1))
xlabel('Time')
ylabel('Income')

subplot(3,1,2)
plot(sims/2+1:sims,c_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Consumption')

subplot(3,1,3)
plot(sims/2+1:sims,a_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Assets')

%% (d) Correlogram between Simulated income and consumption series upto 4 lags

[clgm,lags] = xcorr(y_sim,c_sim,4);

%% (e) Plotting the Correlogram
figure(9)
plot(lags,clgm);
xlabel("lags")
ylabel("Correlation Vector")
title("Correlogram between Simulated Income and Consumption Series")
legend("Correlation at 4 lags")
