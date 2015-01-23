function G = multrnd_unnormalized(pmf,dex,option)
%Mingyuan Zhou, mz31@duke.edu
if nargin<3
    option = 'Gibbs';
end
%pmf is a matrix with each column corresponding to a unnormalized pmf
if strcmp(option, 'Gibbs')
    cdf =cumsum(pmf,1);
   
    
%     for i=1:N
%         G(i) = find(rnd(i)<=cdf(i,:),1,'first');
%     end    
    if nargin<2
        N = size(pmf,2);
        rnd = rand(1,N).*cdf(end,:);
        G = sum(bsxfun(@gt,rnd, cdf),1)+1;
    else
        N = length(dex);
        rnd = rand(1,N).*cdf(end,dex);
        G = sum(bsxfun(@gt,rnd, cdf(:,dex)),1)+1;

        %G = find( bsxfun(@le,rnd, cdf(:,dex)),1,'first');
    end   
    
else
    [tmp,G] = max(pmf,[],2);
end
end
