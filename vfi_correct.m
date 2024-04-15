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

c_choice=zeros(a_n,Y_n,a_n);        %   Note that this is a three dimensional matrix, which is possible in Matlab.
utility=c_choice;                   %   Preallocation

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

%%% VFI Preallocations and Tolerances
tol=10^(-9);            %   Maximum error tolerance       
maxits=10^4;  

V0=repmat(utility(a_n/2,:,1),a_n,1);        %   Initial Guess of the Value Function (3) Remove NaN
%V0=repmat(rnd(),a_n,y_n);
V1=V0;                                      %   Preallocation of the updated value function
c=V1;                                       %   Preallocation of the consumption policy function
a_prime=c;                                  %   Preallocation of the asset choice policy function

%%  Main VFI Loop

count=0;
dif=0.00002;                                % (4) Increase dif by a small amount to allow for difference b/w consecutive iterations of V0
tic
while dif>tol && count<maxits
    dif;
    V_candidate=NaN*ones(a_n,a_n);
    for y=1:Y_n
        for ap=1:a_n
            V_candidate(:,ap)=utility(:,y,ap)+beta*repmat(V0(ap,:),a_n,1)*P(y,:)';
        end
        [V1(:,y),a_prime(:,y)]=max(V_candidate');
    end
    dif=max(max(abs(V0-V1)));             % (5) Add another max
    count=count+1;
    V0=V1;                                % (6) V0 should map to V1 instead of V_candidate
    toc
end

%%  Recovery of Consumption Policy Function

c=(1+r)*repmat(A,1,Y_n)+exp(repmat(Y',a_n,1))-A(a_prime);   % (7) Replace Y with Y_n to specify size of matrix by a constant instead of vector

%%  Plots

figure(1)
plot(A,V1(:,1),A,V1(:,Y_n/2+0.5),A,V1(:,Y_n))
xlabel('Assets')
ylabel('Value')
title('Value Function')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

figure(2)
plot(A,c(:,1),A,c(:,Y_n/2+0.5),A,c(:,Y_n))
xlabel('Assets')
ylabel('Consumption')
title('Consumption')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

figure(3)
plot(A,a_prime(:,1),A,a_prime(:,Y_n/2+0.5),A,a_prime(:,Y_n))
xlabel('Assets')
ylabel('Assets')
title('Asset Choice')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

%%  Simulations

sims=1000;
y_sim=simulate(dtmc(P),sims);
a_index=1;

for t=1:sims
    c_sim(t)=(1+r)*A(a_index(t))+exp(Y(y_sim(t)))-A(a_prime(a_index(t),y_sim(t)));
    a_index(t+1)=a_prime(a_index(t),y_sim(t));
    a_sim(t+1)=A(a_prime(a_index(t),y_sim(t)));
end

figure(4)
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
