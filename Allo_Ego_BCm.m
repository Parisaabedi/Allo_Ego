% Allo-Ego (Method1- using barycenter)

% This function will calculate the mean and varaiance associated with the
% position of missing target in breakfast table project

% the algorithm has two phase of Encoding and Decoding

% in the Encoding the BC = barycenter and R = distance to BC for each
% object is calculated

% in the decoding phase the missing object positon is calculated by
% estimating the BCprime and adding the rn to this value

% written by PA(March 2017)
% updated by PA (Oct 2017)

%% Initialization
clear all
clc
runm = 1;
if runm == 1
    load test1.mat % contains some randomely generated data!
    varv = 6; % visual variability
    varego = .8; % Ego-centric variability
    % Find object positions
    % O = randi([-200 200],12,2); % For the example code!
    Ro = 6; % reletive Objects
    Roi = 6;
    ot = O(1:Ro,:);
    % Generate the distribution
%     OD = zeros(12,2,1000);
%     vm = varv*eye(2);
%     for i = 1:12
%         OD(i,:,:) = mvnrnd(O(i,:),vm,1000)';
%     end
% 
%     save('test1.mat','OD','O','Ro')

    OR = OD(1:Ro,:,:);
    OIR = OD(Ro+1:end,:,:);
    R = 1; % relativity index

    %% Encoding

    % calculate barycenter (center of mass)
    NR = Ro;NIR = Roi;
    mr = R/NR; mir = (1-R)/NIR;
    if mir == 0
        mir = 1e-50;
    end
    % [A, bc(1,1),bc(1,2)] = polycenter(Poly(1:end-1,1),Poly(1:end-1,2));
    cm = mr * sum(OR(:,:,:)) + mir * sum(OIR(:,:,:));
    cmA = mr * sum(O(1:6,:)) + mir * sum(O(6:end,:)); % mean analytical value

    % create a polygon (convex or concave)
    v = boundary(ot(:,1),ot(:,2));
%     plot(ot(v,1),ot(v,2)) % Plot the polygon
    % hold on; scatter(cmcheck(1,1),cmcheck(1,2),'r*')
    lambda1 = [ mr*ones(NR,1) ; mir*ones(NIR,1)];
        % calculate the variance for BC 
    varX = eye(NR + NIR);
    varV = ones(NR + NIR,1);
    for i = 1 : NR + NIR
        if i < NR || i == NR
            varX(i,i) = (varv+1/mr);
            varV(i) = (varv+1/mr);
        else
            varX(i,i) = (varv+1/mir);
            varV(i) = (varv+1/mir);
        end   
    end
    %     varVBC = varX * eye(length(O));
    % varBC = lambda' * varX * lambda;
    varBC = (6*(mr^2)) * varv(1);
    varbc = var(cm(1,2,:));
    % Calculate R vectors (The distance of each object from the centroid)
    temp = [OR ; OIR];
    R = [OR ; OIR] - repmat(cm, [12,1,1]);
    RA = O - repmat(cmA,[12,1]);
       
    % calculate the variance for R
    % analytical Calculation
    varRA = zeros(2,2,12);
    % Z = X - Y => Var(Z) = Var(X) + Var(Y) - 2 cov(X,Y)
    for i = 1 : 12
        if i < (Ro + 1)
            varRA(1,1,i) = varv + varBC - 2 * mr * (varv); % cov(Xi,BC) = mi * var(Xi)
            varRA(2,2,i) = varv + varBC - 2 * mr * (varv);

        else
            varRA(1,1,i) = varv + varBC - 2 * mir * (varv);
            varRA(2,2,i) = varv + varBC - 2 * mir * (varv);
        end

    end
    covr1r2 = varBC - mr * varv - mr * varv ; % cov(r1,r2) = varBc - m1*varx1 - m2 * varx2;
    % Numerical Calculation
    varR = zeros(2,2,12);
    for i = 1 : 12
        t = R(i,:,:);
        t = reshape (t, [2,1000]);
        varR(:,:,i) = cov(t');
    end

    %% Decoding 
    % goal is to find p(Xn|X1 ... Xn-1)
    load('expcond.mat') % different shifts and different simulation conditions!
    
    results = zeros(5*4,6,2);
    % new data observation
    % NewO = [OD(1:5,:,:) OD(7:end,:,:)];
    for i = 1:4
        for j1 = 1:6*5
            NewO = OD(1:5,:,:); % for now didn't consider the irrelevant at all!\
            % NewO(1,1,:) = NewO(1,1,:)  + 50 * ones(1,1,1000);
            NewO(:,1,:) = NewO(:,1,:)  + repmat(1*shifts((i-1)*5+1:i*5,j1),[1,1,1000]);
            Rn = R(1:5,:,:);
            NO = O(1:5,:);
            NO(:,1) = NO(:,1) + 1*shifts((i-1)*5+1:i*5,j1);
            RnA = RA(1:5,:);
            %----------------
            % calculate new center of mass
            bcs = NewO-Rn;
            mubcs = mean(bcs , 3);
            ccm12 = zeros(5,5);
            for i1 = 1:5
                for j = 1:5
                    temp1 = reshape(bcs(i1,1,:),[1000,1]);
                    temp2 = reshape(bcs(j,1,:),[1000,1]);
                    t = cov(temp1,temp2);
                    ccm12(i1,j) = t(1,2) ;                    
                end
            end
            lambda0 = [ 0.2 0.2 0.2 0.2 0.2];
            % set the boundaries
            lb = zeros(5,1);
%             for i1 = 1:5
%                 c = mubcs(:,1) == mubcs(i1,1);
%                 lb(i1) = 1/sum(c); 
%             end
            ub = ones(5,1);
            a = ones(5,1); b = 1;
            options = optimset('Display','iter','TolFun',1e-8,'MaxFunEvals',1e10);
            [lambda,sigma] = fmincon(@(lambda)bayesagg(lambda,ccm12,mubcs(:,1)),...
                                       lambda0,[],[],a',b,lb,ub,[],options);
            muBC = lambda * mubcs(:,1);
            varBC = sigma;
            d = mubcs(:,1) - muBC(:,1);
            ccm12u = ccm12 + eye(5) .* d;
            check = (lambda * inv(ccm12u) * lambda')^-1;
            % Update the posterior = likelihood * prior(=pBC in egocentric driven during encoding)
            VBCp = (varBC^-1 + (varBC+varego)^-1)^-1;
            uBCp = VBCp * (muBC./varBC + cmA./(varBC+varego));

            % calculate the new position (Xn = BCprim + Rn)
            % calculate the 
%             load('BCdist.mat')
            Vxn = zeros(2,1);
            for j = 1 : 2
                Rn = reshape(R(6,j,:),[1000,1]);
                temp = reshape(varR(1,2,1:5),[5,1]);
                Vxn(j) = VBCp + varRA(1,1,1) -2 * lambda*temp;
            end
            Uxn = uBCp + mean(R(6,:,:),3);
            results((i-1)*5+j1-(ceil(j1/5)-1)*5,ceil(j1/5),:) = [Uxn(1) Vxn(1)];
        end
    end

end


%% save the Results
% save('Decoding3.mat','Uxn','Vxn','VBCp','uBCp','x')
% save('Myprop.mat','results')

%% Plot the Results

% figure;hold on; scatter(O(1:6,1),O(1:6,2),140,'b'); scatter(cmA(1),cmA(2),'+k') % Encoding
% scatter(NO(1:5,1), NO(1:5,2),'*r');scatter(uBCp(1),uBCp(2),'xr') % Decoding
% scatter(Uxn(1),Uxn(2),'+g') % new Object position 
% load('Myprop.mat')
% load('expcond.mat')
% load('test1.mat')
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

save('myprop.mat','results','Aw','varN')

%% Plot results
figure
vm = mean(varN,2);
subplot(1,2,2); bar(1:4, vm); title('Variabiliy')
subplot(1,2,1); bar(1:4, Aw); title('Allocentric weight')


