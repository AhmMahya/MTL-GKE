function [l, u] = ComputeDistanceExtremes(X, a, b)
% function [l, u] = ComputeDistanceExtremes(X, a, b, M)
%
% Computes sample histogram of the distances between rows of X and returns
% the value of these distances at the a^th and b^th percentils.  This
% method is used to determine the upper and lower bounds for
% similarity / dissimilarity constraints.  
%
% X: (n x m) data matrix 
% a: lower bound percentile between 1 and 100
% b: upper bound percentile between 1 and 100
% M: Mahalanobis matrix to compute distances 
%
% Returns l: distance corresponding to a^th percentile
% u: distance corresponding the b^th percentile

if (a < 1 || a > 100),
    error('a must be between 1 and 100')
end
if (b < 1 || b > 100),
    error('b must be between 1 and 100')
end

n = size(X, 1); % tedade kole dade ha baraye train --> 600

num_trials = min(100, n*(n-1)/2); % tedade tekrar ha --> 100

% we will sample with replacement
dists = zeros(num_trials, 1); % 100*1
for i=1:num_trials
    j1 = ceil(rand(1)*n); % yek adad random beine 1 ta 600 be dast miavarad
    j2 = ceil(rand(1)*n);    
    dists(i) = (X(j1,:) - X(j2,:))*(X(j1,:) - X(j2,:))'; % faseleye eucleadian 2 sample k be surate random entekhab shodand be dast miad
end


[f, c] = hist(dists, 100); % f mehvar amudi histogram va c mehvar ofoqi histogram. farvani dar  meqdar c(1,1) barabar ast ba f(1,1) va hamintor ta akhar
l = c(floor(a)); % l=50.2684
u = c(floor(b)); % u=448.5122