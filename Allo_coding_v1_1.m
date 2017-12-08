%% Allocentric (PA: March 2017)
% This function performs the required operations for allocentric
% information of our targeted object

% updated by PA: October 2017
% The value for covariances and variances is updated based on the correct
% assumptions!

%% Initialization
% Find object positions
clear all
sigmav = 5;
varego = 2;
no = 6;
load('test1.mat')
        
% O = randi(10,6,2); % For the example code!
% Tid = randi(6,1); % Find the target
Tid = 6;
ot = O(1:Tid-1,:); % Use the other points to creat Triangles (Polygons)
muego = O(Tid);

%% Traditional Barrycentric coordinate Transformation (based on Triangle)
dt = delaunayTriangulation(ot(:,1),ot(:,2)); % Create the Delaunay Triangulation
nt = length(dt.ConnectivityList); % number of Triangles
B = cartesianToBarycentric(dt,(1:nt)',repmat(O(Tid,:),nt,1)); % Transform the target coding from Cartesian to Barycentric

%% Generalized Barrycnetric coordian                    te Transformation (any simplex-polygon here)

% Create a convex polygon, the problem of having convex polygon is that
% some of the points have to be dropped, which is not good for us!
% k = convhull(ot(:,1),ot(:,2));
% plot(ot(k,1),ot(k,2),'r-',ot(:,1),ot(:,2),'b*') % Plot the polygon

% create a polygon (convex or concave)
% V = boundary(ot(:,1),ot(:,2));
% plot(ot(V,1),ot(V,2)) % Plot the polygon
% % Calculate the vertices 
% E = zeros(5,2);
% for i = 1 : 5
%     E(i,:) = ot(V(i+1),:)-ot(V(i),:);
% end



%% Encoding

% Build p(lambda|x,y)
F = sigmav * eye(5);
a = zeros(nt,1);
SA = zeros(nt,1);
C = zeros(no-1,1);
for n = 1:nt
    a(n,1) = dt.Points(dt.ConnectivityList(n,:),1)' * B(n,:)';
    SA(n,1) = B(n,:) * F(1:3,1:3) * B(n,:)';
end
for n = 1:no-1
    q = dt.ConnectivityList == n;
    lambda = sum(q.*B);
    C(n,1) = sum(lambda)/3;
end


%% Decoding

load('expcond.mat')
results = zeros(5*4,6,2);
% new data observation
% NewO = [OD(1:5,:,:) OD(7:end,:,:)];
for i1 = 1:4
    for j1 = 1:6*5
        % Present the new configuration
        ot(:,1) = ot(:,1) + shifts((i1-1)*5+1:i1*5,j1);

        % calculate p(Xp|lambda,ynew)
        xn = zeros(nt,1);
        Cn = zeros(nt,nt);
        for n = 1: nt
            xn(n,1) = ot(dt.ConnectivityList(n,:),1)' * B(n,:)';
            if n ~= nt
                for n1 = n+1 : nt
                    for i = 1: 3
                        q = dt.ConnectivityList(n1,:) == dt.ConnectivityList(n,i);
                        k = sum(q)*B(n,i) + sum(q.*B(n1,:));
                        Cn(n,n1) = Cn(n,n1) + k;
                    end
                end
            end
        end

        % calculate p(Xp|Ynew) = p(Xp|lambda,ynew) * p(lambda|x,y)
        % since lambda is not a random variable then p(Xp|Ynew) =
        % p(Xp|lambda,ynew).

        muxn = mean(xn);
        d = 0;
        for jj = 1 : length(xn)
            for j2 = jj : length(xn)
                k = k + 1;
                d = d + abs(xn(jj) - xn(j2));
            end
        end
        varxn = (sum(SA)+d+2*sum(sum(Cn)))/(no-1)^2;


        %% combine Allo-centric and Ego-centric informaiton
        

        varae = ((1/varxn)+(1/varego))^-1;
        muae = varae*((muego/varego)+(muxn/varxn));
        results((i1-1)*5+j1-(ceil(j1/5)-1)*5,ceil(j1/5),:) = [muae varae];
    end
end
%% save results
save('Gprop.mat','results')

% load('Gprop.mat')
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

save('Gprop.mat','results','Aw','varN')

%% Plot results
vm = mean(varN,2);
subplot(1,2,2); bar(1:4, vm); title('Variabiliy')
subplot(1,2,1); bar(1:4, abs(Aw)); title('Allocentric weight')

