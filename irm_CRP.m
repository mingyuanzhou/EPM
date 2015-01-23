function [AUCroc,AUCpr,F1,ProbAve,z]=irm_CRP(B,K, idx_train,idx_test,Burnin, Collections, IsDisplay)
%Code for Infinite Relational Model
%Mingyuan Zhou, Oct, 2014
%
%Input:
%B is an upper triagular matrix, the diagonal terms are equal to zero
%idx_train: training indices
%idx_test: test indices
%K: truncated number of atoms
%
%Output:
%Phi: each row is a node's feature vector
%r: each elment indicates a community's popularity
%AUCroc: area under the ROC curve
%AUCpr: area under the precition-recall curve
%mi_dot_k: mi_dot_k(k,i) is the count that node $i$ sends to community $k$
%ProbeAve: ProbeAve(i,j) is the estimated probability for nodes $i$ and $j$ to be linked.

if ~exist('idx_train','var')
    idx_train = find(triu(true(size(B)),1));
end
if ~exist('idx_test','var')
    idx_test = [];
end
if ~exist('K','var')
    K = 100;
end
if ~exist('Burnin','var')
    Burnin = 2000;
end
if ~exist('Collections','var')
    Collections = 500;
end
if ~exist('IsDisplay','var')
    IsDisplay = true;
end


%B is an upper triangular matrix
%BTrain constains all the links used to train the model
%BTrain_Mask indicates which (i,j) pairs are used for training
BTrain = B;
BTrain(idx_test)= 0;
BTrain_Mask = zeros(size(B));
BTrain_Mask(idx_train) = 1;

%A, ATrain and ATrain_Mask are all full matrices, their diagonal terms
%should be zero as self links are not allowed
A = B+B';
ATrain = BTrain+BTrain';
ATrain_Mask=BTrain_Mask+BTrain_Mask';



iterMax = Burnin+Collections;

N=size(B,1);

ProbSamples = zeros(N,N);

KK=K;

z = randi(KK,N,1);

alpha=0.1;

Pi = ones(KK,1);
Phi_KK = 0.1*ones(KK,KK);

option = 'block_Gibbs'; %Use this option for blocked Gibbs sampling (with truncation)
option = 'collapsed_CRP'; %Use this option for collapsed Gibbs sampling


a0=1e-1;
b0=1e-0;

output.K_positive = zeros(1,iterMax);
output.K_hardassignment= zeros(1,iterMax);
output.Loglike_Train = zeros(1,iterMax);
output.Loglike_Test = zeros(1,iterMax);
output.AUC = zeros(1,iterMax);

if IsDisplay
    figure
end

for iter=1:iterMax
    
    if strcmp(option, 'block_Gibbs')
        Phi_k_zj = Phi_KK(:,z);
        for i=1:N
            %j_i = [1:i-1,i+1:N]';
            j_i = ATrain_Mask(i,:)==1;
            logprob = log(Pi) + log(max(1-Phi_k_zj(:,j_i),realmin))*sparse(1-A(j_i,i)) + log(max(Phi_k_zj(:,j_i),realmin))*sparse(A(j_i,i));
            logprob = logprob-max(logprob);
            z(i) = multrnd_unnormalized(exp(logprob));
            Phi_k_zj(:,i)=Phi_KK(:,z(i));
        end
        
        zKN=sparse(z,1:N,1,KK,N);
        
        Count_link = zKN*ATrain*zKN';
        Count_link = Count_link - diag(diag(Count_link)/2);
        Count_all = zKN*ATrain_Mask*zKN';
        Count_all = Count_all - diag(diag(Count_all)/2);
        
        
        Phi_KK=betarnd(a0+triu(Count_link),b0+triu(Count_all)-triu(Count_link));
        Phi_KK = triu(Phi_KK)+triu(Phi_KK,1)';
        
        p = betarnd(1e-2+N,1e-2+alpha);
        Pi=gamrnd(sparse(z,1,1,KK,1)+alpha/KK,p);
        %ell = CRT_sum_mex(full(sparse(z,1,1,KK,1)),alpha/KK);
        ell = CRT_sum(full(sparse(z,1,1,KK,1)),alpha/KK);
        alpha = gamrnd(1e-2+ell,1/(1e-2-log(max(1-p,realmin))));
        
        Prob = Phi_KK(z,z);
        
    elseif strcmp(option, 'collapsed_CRP')
        
        if iter==1
            [Uz,~,z] = unique(z);
            KK = length(Uz);
            n_k=sparse(z,1,1,KK,1);
            zKN=sparse(z,1:N,1,KK,N);
            Count_link = zKN*ATrain*zKN';
            Count_link = Count_link - diag(diag(Count_link)/2);
            Count_all = zKN*ATrain_Mask*zKN';
            Count_all = Count_all - diag(diag(Count_all)/2);
        end
        for i=1:N
            j_i = ATrain_Mask(i,:)==1;
            Count_link(z(i),:) = Count_link(z(i),:) - sparse(1,z(j_i),ATrain(i,j_i),1,KK);
            Count_link(:,z(i))=Count_link(z(i),:)';
            Count_all(z(i),:) = Count_all(z(i),:) - sparse(1,z(j_i),ATrain_Mask(i,j_i),1,KK);
            Count_all(:,z(i))=Count_all(z(i),:)';
            
            n_k(z(i))=n_k(z(i))-1;
            
            dex=find(n_k==0);
            if ~isempty(dex)
                
                z(z>dex)=z(z>dex)-1;
                n_k(dex)=[];
                KK= KK-1;
                Count_link(dex,:)=[];
                Count_link(:,dex)=[];
                Count_all(dex,:)=[];
                Count_all(:,dex)=[];
                
            end
            
            %           Count_link_i = zeros(KK,KK);
            %           Count_all_i= zeros(KK,KK);
            %             for k=1:KK
            %                 z(i)=k;
            %                 Count_link_i(k,:)= sparse(1,z(j_i),ATrain(i,j_i),1,KK);
            %                 Count_all_i(k,:)= sparse(1,z(j_i),ATrain_Mask(i,j_i),1,KK);
            %             end
            
            Count_link_i = repmat(sparse(1,z(j_i),ATrain(i,j_i),1,KK),KK,1);
            Count_all_i = repmat(sparse(1,z(j_i),ATrain_Mask(i,j_i),1,KK),KK,1);
            
            logprob=[log(n_k);log(alpha)];
            
            
            
            logprob(1:KK) =  logprob(1:KK) ...
                +sum(betaln(a0+Count_link+Count_link_i, b0+ Count_all-Count_link+Count_all_i-Count_link_i)...
                -betaln(a0+Count_link, b0+ Count_all-Count_link),2);
            
            
            z(i)=KK+1;
            Count_link_i0= sparse(1,z(j_i),ATrain(i,j_i),1,KK+1);
            Count_all_i0= sparse(1,z(j_i),ATrain_Mask(i,j_i),1,KK+1);
            
            logprob(KK+1) = logprob(KK+1) + sum(betaln(a0+Count_link_i0, b0+Count_all_i0-Count_link_i0)...
                -betaln(a0, b0));
            logprob = logprob-max(logprob);
            z(i) = multrnd_unnormalized(exp(logprob));
            if z(i)>KK
                KK = KK+1;
                Count_link(end+1,end+1)=0;
                Count_all(end+1,end+1)=0;
                n_k(end+1,:)=0;
                
                %eta(end+1,:)=gamrnd(0.05,1);
            end
            
            n_k(z(i))=n_k(z(i))+1;
            
            Count_link(z(i),:) = Count_link(z(i),:) + sparse(1,z(j_i),ATrain(i,j_i),1,KK);
            % Count_link(:,z(i)) = Count_link(:,z(i)) + sparse(1,z(j_i),ATrain(j_i,i),1,KK)';
            %\Count_link(z(i),z(i)) = Count_link(z(i),z(i))+1;
            Count_link(:,z(i))=Count_link(z(i),:)';
            Count_all(z(i),:) = Count_all(z(i),:) + sparse(1,z(j_i),ATrain_Mask(i,j_i),1,KK);
            %Count_all(z(i),z(i)) = Count_all(z(i),z(i))+1;
            Count_all(:,z(i))=Count_all(z(i),:)';
            %Count_all(:,z(i)) = Count_all(:,z(i)) + sparse(1,z(j_i),BTrain_Mask(j_i,i),1,KK)';
            
            
        end
        
        
        Phi_KK=betarnd(a0+triu(Count_link),b0+triu(Count_all)-triu(Count_link));
        Phi_KK = triu(Phi_KK)+triu(Phi_KK,1)';
        Prob = Phi_KK(z,z);
               
        p = betarnd(1e-2+N,1e-2+alpha);
        alpha = gamrnd(1e-2+length(unique(z)),1/(1e-2-log(max(1-p,realmin))));  
        %The inference of alpha and p is based on the negative binomial
        %process discussed in M. Zhou and L. Carin, "Negative binomial 
        %process count and mixture modeling," IEEE TPAMI, Feb. 2015.
    end
    
    if iter>Burnin
        %ProbSamples(:,:,iter-Burnin) = Prob;
        ProbSamples = ProbSamples +  Prob;
        ProbAve = ProbSamples/(iter-Burnin);
    else
        ProbAve = Prob;
    end
        
    rate= Prob(idx_test);
    links = double(B(idx_test)>0);
    output.AUC(iter) = aucROC(rate,links);
    output.K_positive(iter)  = nnz(sparse(z,1,1,KK,1));
    
    if mod(iter,100)==0 && IsDisplay
        %  rate = 1-exp(-ProbAve(idx_test));
        
        [~,Rankdex]=sort(z,'descend');
        subplot(3,3,1);imagesc(A(Rankdex,Rankdex));
        subplot(3,3,2);imagesc(ProbAve(Rankdex,Rankdex));
        subplot(3,3,3);plot(z,'*');
        subplot(3,3,4);plot(output.AUC(1:iter));
        subplot(3,3,5);plot(output.K_positive(1:iter));title(num2str(alpha))
        subplot(3,3,6);plot(Pi)
        subplot(3,3,7);imagesc(Phi_KK)
        %subplot(3,3,7);imagesc(Count_link);
        % subplot(3,3,8);imagesc(Count_link);
        %    subplot(3,3,9);imagesc(abs(Count_link-Count_link0));
        drawnow
    end
    if mod(iter,1000)==0
        fprintf('Iter= %d, Number of Communities = %d \n',iter, output.K_positive(iter));
    end
end

figure;
subplot(1,2,1)
links = double(B(idx_test)>0);
rate= ProbAve(idx_test);
[~,dex]=sort(rate,'descend');
subplot(2,2,1);plot(rate(dex))
subplot(2,2,2);plot(links(dex),'*')
subplot(2,2,3);
[X,Y,T,AUCroc] = perfcurve(links,rate,1);
plot(X,Y);
axis([0 1 0 1]), grid on, xlabel('FPR'), ylabel('TPR'), hold on;
x = [0:0.1:1];plot(x,x,'b--'), hold off; title(['AUCroc = ', num2str(AUCroc)])
subplot(2,2,4)
[prec, tpr, fpr, thresh] = prec_rec(rate, links,  'numThresh',3000);
plot([0; tpr], [1 ; prec]); % add pseudo point to complete curve
xlabel('recall');
ylabel('precision');
title('precision-recall graph');
AUCpr = trapz([0;tpr],[1;prec]);
F1= max(2*tpr.*prec./(tpr+prec));
title(['AUCpr = ', num2str(AUCpr), '; F1 = ', num2str(F1)])


figure;
subplot(2,2,1); imagesc((A(Rankdex,Rankdex)).*ATrain_Mask(Rankdex,Rankdex));
subplot(2,2,2); imagesc(ProbAve(Rankdex,Rankdex).*ATrain_Mask(Rankdex,Rankdex))
subplot(2,2,3); imagesc((A(Rankdex,Rankdex)).*~ATrain_Mask(Rankdex,Rankdex));
subplot(2,2,4);imagesc((ProbAve(Rankdex,Rankdex)).*~ATrain_Mask(Rankdex,Rankdex))

