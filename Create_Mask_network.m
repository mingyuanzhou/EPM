function [idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio)
%B is an upper triangular matrix
%Randomly hold out (1-TrainRatio)*100% of pairs of nodes
%If removing an edge disconnects a node from all the other nodes, then this 
%edge would be kepted in the training set.

B = triu(B,1);
N=size(B,2);
X = triu(true(N),1);
idx = find(X);
idx = idx(randperm(length(idx)));
count = 0;
IsTest = false(length(idx),1);
BB= B+B';
for i=1:length(idx)
    if count>= floor((1-TrainRatio)*length(idx));
        break;
    end
    if B(idx(i))==0
        count = count+1;
        IsTest(i)=true;
    else
        %B(idx(i))=0;
        [ii,jj]=ind2sub([N,N],idx(i));
        BB(ii,jj)=0;
        BB(jj,ii)=0;
        %if (sum(B(ii,:))==0 && sum(B(:,jj))==0)
        %if (sum(B(ii,:))+sum(B(:,ii)))==0 || (sum(B(jj,:))+sum(B(:,jj)))==0
        %if sum(B(ii,:))==0 || (sum(B(jj,:)))==0
        if sum(BB(ii,:))==0 || sum(BB(jj,:))==0
            %B(idx(i))=1;
            BB(ii,jj)=1;
            BB(jj,ii)=1;
        else
            IsTest(i)=true;
            count = count+1;
        end
    end
    
end
idx_train = idx(~IsTest);
idx_test = idx(IsTest);
    
BTrain_Mask = zeros(size(B));
BTrain_Mask(idx_train) = 1;
BTrain_Mask=BTrain_Mask+BTrain_Mask';