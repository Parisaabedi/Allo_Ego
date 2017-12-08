%% Allo_Ego_Dom
% This function will calculate the position of a missing object by
% combining Allo vs Ego centric information. The method was proposed by
% Dominik Endres
% Written by PA (April 2017)

%% Initialization
sigv = 10;
sigpBC = 2;
spBC = ones(2) * sigpBC;
mpBC = [0 , 0];
sigpR = 1;
spR = ones(2) * sigpR;
mpR = zeros(6,2);
% O = randi([-200 200],5,2); % 5 objects (x,y) Postions
load('test1.mat')
OT = O(1:6,:);
%% Encoding phase
% Prior distribution for BC
pBC = mvnrnd(mpBC,spBC,1000);
% Prior distribution for R
pR = zeros(6,2,1000);
for i = 1 : length(mpR)
    pR(i,:,:) = mvnrnd(mpR(i,:),spR,1000)';
end

% Prior beliefs of observer
sigpR = ones(length(mpR),1) * sigpR;
[muRC, sigRC] = JPrior(mpBC,sigpBC,mpR,sigpR);

% observe and compute posterior
[muRC_O,sigRC_O] =PDist(sigv,muRC(:,1),sigRC,O , 0);

%% Decoding phase
load('expcond.mat')
results = zeros(5*4,6,2);
% new data observation
% NewO = [OD(1:5,:,:) OD(7:end,:,:)];
for i1 = 1:4
    for j1 = 1:6*5
        % Remove one object and shift the rest
        Odec = OT(1:5,1);
        Odec = Odec(:,1) + shifts((i1-1)*5+1:i1*5,j1);
        mind = 5;
        % Observe and compute Posterior
        [muRC_OO,sigRC_OO] =PDist(sigv,muRC_O(:,1),sigRC_O,Odec,mind);

        % Calculate the missing object position and variance
        muMO = muRC_OO(end,end) + muRC_OO(end-1);
        sigMO = sigRC_OO(end,end) + sigRC_OO(end-1,end-1) + 0 * 2 * sigRC_OO(end-1,end);
        results((i1-1)*5+j1-(ceil(j1/5)-1)*5,ceil(j1/5),:) = [muMO sigMO];
    end
end
% [muOO , sigOO] = JDist(sigv, muRC_OO,sigRC_OO);
% muMO = muOO(MOind,:);
% sigMO = sigRC_OO(MOind , MOind);
%% Plot the Results!
% save('Dprop.mat','results')

load('Dprop.mat')
load('expcond.mat')
load('test1.mat')
temp = zeros(4*5,6);
% temp2 = zeros(4*5,6);
s = [5 -5 10 -10 15 -15];
for i = 1:6
   temp(1:15,i) =  s(i);
end
t = mean(shifts(16:end,:));
for i = 1:6
    temp(16:end,i) = t((i-1)*5+1:i*5);
end
tempy = results(:,:,1); tempx = temp;
x = zeros(30,4);y = zeros(30,4);
for i = 1 : 4
    t = tempx((i-1)*5+1:i*5,:);
    x(:,i) = t(:);
    t = tempy((i-1)*5+1:i*5,:);
    y(:,i) = t(:);
end
Aw = zeros(4,1);
varN = zeros(4,6);
for i = 1: 4
    yr = y(:,i)-O(6,1);
    xr = [ones(30,1) x(:,i)];
    b = regress(yr,xr);
    Aw(i) = b(2);
    varN(i,:) = mean(mean(results((i-1)*5+1:i*5,:,2)));
end

% save('Dprop.mat','results','Aw','varN')

%% Plot results
vm = mean(varN,2);
subplot(1,2,2); bar(1:4, vm); title('Variabiliy')
subplot(1,2,1); bar(1:4, abs(Aw)); title('Allocentric weight')



