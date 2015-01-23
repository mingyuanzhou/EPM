function AUC = aucROC(rate,links)
[~,dex]=sort(rate);
links_ranked = links(dex);
num_nonlinks = nnz(~links);
num_links = nnz(links);
AUC = (sum(find(links_ranked>0))  - (num_links*(num_links+1))/2)/(num_nonlinks *num_links);
