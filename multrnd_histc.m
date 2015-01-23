function r = multrnd_histc(n,p)
p=p(:);
%p=p/sum(p);
edges = [0; cumsum(p,1)];
%edges(:,end) = 1; % guard histc against accumulated round-off, but after above check
r = histc(rand(n,1)*edges(end),edges);
r(end)=[];
r=r(:);
    