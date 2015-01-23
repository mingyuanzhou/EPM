function [AUCroc,AUCpr,F1,Phi,r,ProbAve,mi_dot_k,output,z]=GP_EPM(B,K, idx_train,idx_test,Burnin, Collections, IsDisplay, Datatype, Modeltype)
%Code for Gamma Process Edge Partition Model
%Mingyuan Zhou, Oct, 2014
%
%Input:
%B is an upper triagular matrix, the diagonal terms are not defined
%idx_train: training indices
%idx_test: test indices
%K: truncated number of atoms
%Datatype: 'Count' or 'Binary'. Use 'Count' for integer-weigted edges.
%
%Output:
%Phi: each row is a node's feature vector
%r: each elment indicates a community's popularity
%AUCroc: area under the ROC curve
%AUCpr: area under the precition-recall curve
%mi_dot_k: mi_dot_k(k,i) is the count that node $i$ sends to community $k$
%ProbeAve: ProbeAve(i,j) is the estimated probability for nodes $i$ and $j$
%to be linked
%z: hard community assignment

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
if ~exist('Datatype','var')
    Datatype = 'Binary';
    %Datatype = 'Count';
end
if ~exist('Modeltype','var')
    Modeltype = 'Infinite';
end


iterMax = Burnin+Collections;
N = size(B,2);

BTrain_Mask_triu = zeros(size(B));
BTrain_Mask_triu(idx_train) = 1;
%BTrain_Mask_triu = sparse(BTrain_Mask_triu);
BTrain_Mask=full(BTrain_Mask_triu+BTrain_Mask_triu');
BTrain_Mask_logic = BTrain_Mask>0;

BTrain = B;
BTrain(idx_test)= 0;

[ii,jj,M]=find(BTrain);
idx  = sub2ind([N,N],ii,jj);

[ii1,jj1]  = ind2sub([N,N],idx_test);
%Parameter Initialization
Phi = randg(1e-0*ones(N,K));
r = ones(K,1)/K;
c0=1;
%a_i = 1/N*ones(N,1);
a_i=0.01*ones(N,1);
c_i = ones(N,1);
IsMexOK=true;

e_0=1e-0;
f_0=1e-0;


gamma0=1;
AA = B+B'+eye(N);

output.K_positive = zeros(1,iterMax);
output.K_hardassignment= zeros(1,iterMax);
output.Loglike_Train = zeros(1,iterMax);
output.Loglike_Test = zeros(1,iterMax);
output.AUC = zeros(1,iterMax);

ProbSamples = zeros(N,N);
 Kmin=inf;
 EPS=0;
 
  if IsDisplay
     figure
 end
 
for iter=1:iterMax
    
    %draw a latent count for each edge
    if strcmp(Datatype, 'Binary')
        Rate = (Phi(ii,:).*Phi(jj,:))*r;
        M = truncated_Poisson_rnd(Rate);
                
    end
    
    %Sample m_ijk and update mi_dot_k
    if IsMexOK
        mi_dot_k = Multrnd_mijk(sparse(ii,jj,M,N,N),Phi,r);
    else
        mi_dot_k = zeros(K,N);
        for ij=1:length(idx)
            pmf = Phi(ii(ij),:)'.*r.*Phi(jj(ij),:)';
            mij_k = multrnd_histc(M(ij),pmf);
            mi_dot_k(:,ii(ij)) =  mi_dot_k(:,ii(ij)) + mij_k;
            mi_dot_k(:,jj(ij)) =  mi_dot_k(:,jj(ij)) + mij_k;
        end
    end
    

    %Sample a_i
    Phir = Phi*sparse(diag(r));
    ell = CRT_sum_mex_matrix(sparse(mi_dot_k),a_i')';
    
    if isempty(idx_test)
        a_i = randg(1e-2+ell)./(1e-2-sum(log(max(1./(1+ bsxfun(@minus,sum(Phir,1),Phir)./c_i(:,ones(1,K))),realmin)),2));
    else
        a_i = randg(1e-2+ell)./(1e-2-sum(log(max(1./(1+ BTrain_Mask*Phir./c_i(:,ones(1,K))),realmin)),2));
    end
    
    %Sample phi_ik
    if isempty(idx_test) %IsMexOK
        Phi_temp = randg(a_i(:,ones(1,K))+ mi_dot_k');
        temp = sum(Phir,1);
        for i=randperm(N)
            % subdex=~BTrain_Mask_logic(i,:);
            % temp = temp- subdex*Phir;
            temp = temp - Phir(i,:);
            Phi(i,:) = Phi_temp(i,:)./(c_i(i)+temp);
            Phir(i,:) = Phi(i,:).*r';
            % temp = temp+ sum(Phir(subdex,:),1);
            temp = temp + Phir(i,:);
        end
    else
        Phi_tempT = randg(a_i(:,ones(1,K))+ mi_dot_k')';
        PhiT=Phi';
        PhirT = Phir';
        for i=randperm(N)
            %Phik_minus_i = BTrain_Mask(i,:)*Phir;
            %a_i(i) = a_i_temp(i)./(1e-2-sum(log(max(c_i(i)./(c_i(i)+Phik_minus_i),realmin))));
            %Phi(i,:) = randg(a_i(i) + mi_dot_k(:,i)')./(c_i(i)+Phik_minus_i);
            %Phi(i,:) = Phi_temp(i,:)./(c_i(i)+BTrain_Mask(i,:)*Phir);
            % Phir(i,:) = Phi(i,:).*r';
            PhiT(:,i) = Phi_tempT(:,i)./(c_i(i)+PhirT*BTrain_Mask(:,i));
            %PhiT(:,i) = Phi_tempT(:,i)./(c_i(i)+sum(PhirT(:,BTrain_Mask_logic(:,i)),2));
            PhirT(:,i) = PhiT(:,i).*r;
        end
        Phi=PhiT';
        Phir = PhirT';
    end
    
        
    
    %         Phir = Phi*sparse(diag(r));
    %         for i=randperm(N)
    %             Phik_minus_i = BTrain_Mask(i,:)*Phir;
    %             ell = CRT_sum_mex(mi_dot_k(:,i),a_i(i));
    %             a_i(i) = randg(ell+1e-2)./(1e-2-sum(log(max(c_i(i)./(c_i(i)+Phik_minus_i),realmin))));
    %             Phi(i,:) = randg(a_i(i) + mi_dot_k(:,i)')./(c_i(i)+Phik_minus_i);
    %             Phir(i,:) = Phi(i,:).*r';
    %         end
    
    
    
    %Sample c_i
    c_i = randg(1e-0 + K*a_i)./(1e-0 +  sum(Phi,2));
    %c_i = gamrnd(1e-2 +sum(K*a_i),1./(1e-2 +  sum(sum(Phi,2))))*ones(N,1);
    
    
    if strcmp(Modeltype, 'Infinite')
        
        %Sample r_k
        if isempty(idx_test)
            temp=sum(Phi.*(bsxfun(@minus,sum(Phi,1),Phi)),1)'/2;
        else
            temp=sum(Phi.*(BTrain_Mask_triu*Phi),1)';
        end
        r = randg(gamma0/K+ sum(mi_dot_k,2)/2)./(c0+temp);
        
        %Sample gamma0
        ell = CRT_sum_mex(sum(mi_dot_k,2)/2,gamma0/K);
        gamma0 = randg(1e-0 + ell)/(1e-0-1/K* sum(log(max(c0./(c0+ temp),realmin))));
        
        %Sample c0
        %c0=1;
        c0 = randg(1e-0 + gamma0)/(1e-0+sum(r));
    end
    
    
    
    %Number of communities assigned with nonzero counts
    output.K_positive(iter) = nnz(sum(mi_dot_k,2));
    
    Prob =Phi*sparse(diag(r))*Phi'+EPS;
    Prob = 1-exp(-Prob);
    
    if iter>Burnin
        ProbSamples = ProbSamples +  Prob;
        ProbAve = ProbSamples/(iter-Burnin);
    else
        ProbAve = Prob;
    end
    % output.Loglike_Train(iter) = mean(B(idx_train).*log(max(ProbAve(idx_train),realmin))+(1-B(idx_train)).*log(max(1-ProbAve(idx_train),realmin)));
    % output.Loglike_Test(iter) = mean(B(idx_test).*log(max(ProbAve(idx_test),realmin))+(1-B(idx_test)).*log(max(1-ProbAve(idx_test),realmin)));
    
    %rate = 1-exp(-ProbAve(idx_test));
    rate= Prob(idx_test);
    links = double(B(idx_test)>0);
    output.AUC(iter) = aucROC(rate,links);
    %[~,~,~,output.AUC(iter)] = perfcurve(links,rate,1);
    
    [~,rdex]=sort(sum(mi_dot_k,2),'descend');
    if 0
        Phir =  Phi.*(sparse(BTrain_Mask)*Phi)*sparse(diag(r));
        Phir = Phir(:,rdex);
        [~,z] = max(Phir,[],2);
    else
        [~,z] = max(mi_dot_k(rdex,:),[],1);
    end
    
    output.K_hardassignment(iter) = length(unique(z));
    
    if mod(iter,100)==0 && IsDisplay
        
        
        [~,Rankdex]=sort(z);
        subplot(2,3,1);imagesc(AA(Rankdex,Rankdex));title(num2str(EPS));
        subplot(2,3,2);imagesc(ProbAve(Rankdex,Rankdex));
        subplot(2,3,3);imagesc(log(Phi(Rankdex,rdex)*sqrt(diag(r(rdex)))+1e-2));
        subplot(2,3,4);
        plot(r(rdex));
        subplot(2,3,5);plot(1:iter,output.K_positive(1:iter),1:iter,output.K_hardassignment(1:iter));
        %         if iter>Burnin
        %             subplot(2,3,5);plot(Burnin:iter,output.Loglike_Train(Burnin:iter),'b',Burnin:iter,output.Loglike_Test(Burnin:iter),'r');
        %         end
        subplot(2,3,6);plot(output.AUC(1:iter));title(num2str(gamma0));
        drawnow;
    end
    if mod(iter,1000)==0
        fprintf('Iter= %d, Number of Communities = %d \n',iter, output.K_positive(iter));
    end
    
%     if output.K_positive(iter)<=Kmin && iter>1000
%         if output.K_positive(iter)>1
%         Kmin = output.K_positive(iter);
%               Network_plot_PCM;
%         end
%     end

    
end


%%
rate = ProbAve(idx_test);


% zerorows = find((sum(BTrain_Mask.*(B+B'),2)==0&sum(~BTrain_Mask.*(B+B'),2)>0));
% %zerorows = find((sum(BTrain_Mask.*(B+B'),2)==0));
% for j=1:length(idx_test)
%     [i1,j1]=ind2sub([N,N],idx_test(j));
%     if nnz(i1==zerorows)>0 || nnz(j1==zerorows)>0
%         rate(j)=mean(rate);
%     end
% end

[~,rdex]=sort(sum(mi_dot_k,2),'descend');
[~,z] = max(mi_dot_k(rdex,:),[],1);
[~,Rankdex]=sort(z);

if isempty(idx_test)
    rate = ProbAve(idx_train);
    idx_test = idx_train;
end
figure;
subplot(1,2,1)
links = double(B(idx_test)>0);
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
subplot(3,2,1); imagesc((AA(Rankdex,Rankdex)).*BTrain_Mask(Rankdex,Rankdex));
subplot(3,2,2); imagesc((ProbAve(Rankdex,Rankdex)).*BTrain_Mask(Rankdex,Rankdex))
subplot(3,2,3); imagesc((AA(Rankdex,Rankdex)).*~BTrain_Mask(Rankdex,Rankdex));
subplot(3,2,4);imagesc((ProbAve(Rankdex,Rankdex)).*~BTrain_Mask(Rankdex,Rankdex))
subplot(3,2,5)
plot(sum(mi_dot_k(:,Rankdex)'>0,2),'*')
subplot(3,2,6);
imagesc(1-exp(-Phi(Rankdex,:)*(diag(r))*Phi(Rankdex,:)'))