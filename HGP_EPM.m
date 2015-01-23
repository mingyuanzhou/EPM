function [AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z]=HGP_EPM(B,K, idx_train,idx_test,Burnin, Collections, IsDisplay, Datatype, Modeltype)
%Code for Hierachical Gamma Process Edge Partition Model
%Mingyuan Zhou, Oct, 2014
%Input:
%B is an upper triagular matrix, the diagonal terms are not defined
%idx_train: training indices
%idx_test: test indices
%K: truncated number of atoms
%Datatype: 'Count' or 'Binary'. Use 'Count' for integer-weigted edges.


%Output:
%Phi: each row is a node's feature vector
%r_k: each elment indicates a community's popularity
%Lambda_KK: community-community interaction strengh rate matrix
%AUCroc: area under the ROC curve
%AUCpr: area under the precition-recall curve
%m_i_k_dot_dot: m_i_k_dot_dot(k,i) is the count that node $i$ sends to community $k$
%ProbeAve: ProbeAve(i,j) is the estimated probability for nodes $i$ and $j$ to be linked
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
    %Modeltype = 'Finite';
end

iterMax = Burnin+Collections;
N = size(B,2);

BTrain_Mask = zeros(size(B));
BTrain_Mask(idx_train) = 1;
BTrain_Mask=BTrain_Mask+BTrain_Mask';

BTrain = B;
BTrain(idx_test)= 0;

[ii,jj,M]=find(BTrain);
idx  = sub2ind([N,N],ii,jj);


Phi = gamrnd(1e-0*ones(N,K),1);


output.K_positive = zeros(1,iterMax);
output.K_hardassignment= zeros(1,iterMax);
output.Loglike_Train = zeros(1,iterMax);
output.Loglike_Test = zeros(1,iterMax);
AUC = zeros(1,iterMax);

ProbSamples = zeros(N,N);
r_k=ones(K,1)/K;



%Parameter Initialization
Epsilon = 1;
beta1=1;
beta2=1;

Lambda_KK=r_k*r_k';
Lambda_KK = triu(Lambda_KK,1)+triu(Lambda_KK,1)';
Lambda_KK(sparse(1:K,1:K,true))=Epsilon*r_k;
gamma0=1;
c0=1;
a_i=0.01*ones(N,1);
e_0=1e-0;
f_0=1e-0;
c_i = ones(N,1);
count=0;
IsMexOK=true;
AA=B+B'+eye(N);
LogLikeMax = -inf;
Kmin=inf;
EPS=0.01;
EPS=0;

if IsDisplay
    figure
end

for iter=1:iterMax
    
    %draw a latent count for each edge
    if strcmp(Datatype, 'Binary')
        Rate = sum((Phi(ii,:)*Lambda_KK).*Phi(jj,:),2);
        M = truncated_Poisson_rnd(Rate);
    end
    
    %Sample m_i_k1_k2_j and update m_i_k_dot_dot and m_dot_k_k_dot
    if IsMexOK
        [m_i_k_dot_dot, m_dot_k_k_dot] = Multrnd_mik1k2j(sparse(ii,jj,M,N,N),Phi,Lambda_KK);
    else
        m_i_k_dot_dot = zeros(K,N);
        m_dot_k_k_dot = zeros(K,K);
        for ij=1:length(idx)
            pmf = (Phi(ii(ij),:)'*Phi(jj(ij),:)).*Lambda_KK;
            mij_kk = reshape(multrnd_histc(M(ij),pmf(:)),K,K);
            m_i_k_dot_dot(:,ii(ij)) = m_i_k_dot_dot(:,ii(ij)) + sum(mij_kk,2);
            m_i_k_dot_dot(:,jj(ij)) = m_i_k_dot_dot(:,jj(ij)) + sum(mij_kk,1)';
            m_dot_k_k_dot = m_dot_k_k_dot + mij_kk + mij_kk';
        end
    end
    m_dot_k_k_dot(sparse(1:K,1:K,true))=m_dot_k_k_dot(sparse(1:K,1:K,true))/2;
    
    %Number of communities assigned with nonzero counts
    output.K_positive(iter) = nnz(sum(m_i_k_dot_dot,2));
    
    
    if 0 %iter>100
        %First sample all $a_i$ and then sample all $phi_ik$ is faster but
        %may provide worse performance
        %Sample a_i
        Phi_times_Lambda_KK = Phi*Lambda_KK;
        ell = CRT_sum_mex_matrix(sparse(m_i_k_dot_dot),a_i')';
        if isempty(idx_test)
            a_i = randg(1e-2+ell)./(1e-2- sum(log(max(1./(1+ bsxfun(@minus,sum(Phi_times_Lambda_KK,1),Phi_times_Lambda_KK)./c_i(:,ones(1,K))), realmin)),2));
        else
            a_i = randg(1e-2+ell)./(1e-2- sum(log(max(1./(1+ BTrain_Mask*Phi_times_Lambda_KK./c_i(:,ones(1,K))), realmin)),2));
        end
        
        %Sample phi_ik
        Phi_temp = randg(bsxfun(@plus,a_i, m_i_k_dot_dot'));
        if isempty(idx_test) %IsMexOK
            temp = sum(Phi_times_Lambda_KK,1);
            for i=randperm(N)
                temp = temp - Phi_times_Lambda_KK(i,:);
                Phi(i,:) = Phi_temp(i,:)./(c_i(i)+temp);
                Phi_times_Lambda_KK(i,:) = Phi(i,:)*Lambda_KK;
                temp = temp + Phi_times_Lambda_KK(i,:);
            end
        else
            for i=randperm(N)
                Phi(i,:) =  Phi_temp(i,:)./(c_i(i)+BTrain_Mask(i,:)*Phi_times_Lambda_KK);
                Phi_times_Lambda_KK(i,:) = Phi(i,:)*Lambda_KK;
            end
        end
        
    else
        
        %Sample a_i and phi_ik
        Phi_times_Lambda_KK = Phi*Lambda_KK;
        if isempty(idx_test)
            temp = sum(Phi_times_Lambda_KK,1);
            for i=randperm(N)
                temp = temp - Phi_times_Lambda_KK(i,:);
                ell = CRT_sum_mex(m_i_k_dot_dot(:,i),a_i(i));
                p_ik_prime_one_minus = c_i(i)./(c_i(i)+temp);
                a_i(i) = randg(ell+1e-2)/(1e-2-sum(log(max(p_ik_prime_one_minus,realmin))));
                Phi(i,:) =  randg(a_i(i) + m_i_k_dot_dot(:,i))'./(c_i(i)+temp);
                Phi_times_Lambda_KK(i,:) = Phi(i,:)*Lambda_KK;
                temp = temp + Phi_times_Lambda_KK(i,:);
            end
        else
            for i=randperm(N)
                ell = CRT_sum_mex(m_i_k_dot_dot(:,i),a_i(i));
                p_ik_prime_one_minus = c_i(i)./(c_i(i)+(BTrain_Mask(i,:))*Phi_times_Lambda_KK);
                a_i(i) = gamrnd(ell+1e-2,1./(1e-2-sum(log(max(p_ik_prime_one_minus,realmin)))));
                %Phi(i,:) =  randg(a_i(i) + m_i_k_dot_dot(:,i))'./(c_i(i)+(BTrain_Mask(i,:))*Phi_times_Lambda_KK);
                Phi(i,:) =  randg(a_i(i) + m_i_k_dot_dot(:,i))'.*p_ik_prime_one_minus/c_i(i);
                Phi_times_Lambda_KK(i,:) = Phi(i,:)*Lambda_KK;
            end
        end
    end
    %Sample c_i
    c_i = gamrnd(1e-0 + K*a_i,1./(1e-0 +  sum(Phi,2)));
    
    %Phi_KK(k_1,k_2) = 2^{-\delta(k_2=k_1)} \sum_{i}\sum_{j\neq i} \phi_{ik_1} \phi_{jk_2}
    if isempty(idx_test)
        Phi_KK = Phi'*bsxfun(@minus,sum(Phi,1),Phi);
    else
        Phi_KK = Phi'*BTrain_Mask*Phi;
    end
    Phi_KK(sparse(1:K,1:K,true)) = Phi_KK(sparse(1:K,1:K,true))/2;
    
    triu1dex = triu(true(K),1);
    diagdex = sparse(1:K,1:K,true);
    
    
    %Sample r_k
    L_KK=zeros(K,K);
    temp_p_tilde_k=zeros(K,1);
    p_kk_prime_one_minus = zeros(K,K);
    for k=randperm(K)
        R_KK=r_k';
        R_KK(k)=Epsilon;
        beta3=beta2*ones(1,K);
        beta3(k)=beta1;
        p_kk_prime_one_minus(k,:) = beta3./(beta3+ Phi_KK(k,:));
        
        if strcmp(Modeltype, 'Infinite')
            L_KK(k,:) = CRT_sum_mex_matrix(sparse(m_dot_k_k_dot(k,:)),r_k(k)*R_KK);
            temp_p_tilde_k(k) = -sum(R_KK.*log(max(p_kk_prime_one_minus(k,:), realmin)));
            r_k(k) = randg(gamma0/K+sum(L_KK(k,:)))./(c0+temp_p_tilde_k(k));
        end
    end
    
    if strcmp(Modeltype, 'Infinite')
        %Sample gamma0 with independence chain M-H
        ell_tilde = CRT_sum_mex(sum(L_KK,2),gamma0/K);
        sum_p_tilde_k_one_minus = -sum(log(c0./(c0+temp_p_tilde_k) ));
        gamma0new = randg(e_0 + ell_tilde)./(f_0 + 1/K*sum_p_tilde_k_one_minus);
        %AcceptProb1 = exp(sum(gampdfln(max(r_k,realmin),gamma0new/K,c0))+gampdfln(gamma0new,e_0,f_0) + gampdfln(gamma0,e_0 + ell_tilde,f_0-1/K* sum(log(c0./(c0+temp_p_tilde_k) ))  )...
        %         -(sum(sum(gampdfln(max(r_k,realmin),gamma0/K,c0))+gampdfln(gamma0,e_0,f_0) + gampdfln(gamma0new,e_0 + ell_tilde,f_0-1/K* sum(log(c0./(c0+temp_p_tilde_k) )) ) )   ));
        AcceptProb1 = CalAcceptProb1(r_k,c0,gamma0,gamma0new,ell_tilde,1/K*sum_p_tilde_k_one_minus,K);
        if AcceptProb1>rand(1)
            gamma0=gamma0new;
            count =count+1;
            %count/iter
        end
        %gamma0 =0.01;
        %Sample c0
        c0 = randg(1 + gamma0)/(1+sum(r_k));
    end
    % gamma0 = 0.1;
    %Sample Epsilon
    ell = sum(CRT_sum_mex_matrix( sparse(m_dot_k_k_dot(diagdex))',Epsilon*r_k'));
    %if iter>100
    Epsilon = randg(ell+1e-2)/(1e-2-sum(r_k.*log(max(p_kk_prime_one_minus(diagdex), realmin))));
    % Epsilon=1;
    %end
    
    %Sample lambda_{k_1 k_2}
    R_KK = r_k*(r_k');
    R_KK(sparse(1:K,1:K,true)) = Epsilon*r_k;
    Lambda_KK=zeros(K,K);
    Lambda_KK(diagdex) = randg(m_dot_k_k_dot(diagdex) + R_KK(diagdex))./(beta1+Phi_KK(diagdex));
    Lambda_KK(triu1dex) = randg(m_dot_k_k_dot(triu1dex) + R_KK(triu1dex))./(beta2+Phi_KK(triu1dex));
    Lambda_KK = Lambda_KK + triu(Lambda_KK,1)'; %Lambda_KK is symmetric
    
    %     if iter<100
    %         Lambda_KK=Lambda_KK-triu(Lambda_KK,2);
    %         Lambda_KK = triu(Lambda_KK,1)+triu(Lambda_KK,1)';
    %     end
    
    %Sample beta1 and beta2
    %     beta1 = randg(sum(R_KK(diagdex))+1e-2)./(1e-2+ sum(Lambda_KK(diagdex)));
    %     beta2 = randg(sum(R_KK(triu1dex))+1e-2)./(1e-2+ sum(Lambda_KK(triu1dex)));
    %
    beta1 = randg(sum(R_KK(diagdex))+sum(R_KK(triu1dex))+1e-0)./(1e-0+ sum(Lambda_KK(diagdex))+sum(Lambda_KK(triu1dex)));
    beta2 = beta1;
    
    
    
    Prob =Phi*(Lambda_KK)*Phi'+EPS;
    Prob = 1-exp(-Prob);
    if iter>Burnin
        %ProbSamples(:,:,iter-Burnin) = Prob;
        ProbSamples = ProbSamples +  Prob;
        ProbAve = ProbSamples/(iter-Burnin);
    else
        ProbAve = Prob;
    end
    %output.Loglike_Train(iter) = mean(B(idx_train).*log(max(ProbAve(idx_train),realmin))+(1-B(idx_train)).*log(max(1-ProbAve(idx_train),realmin)));
    %output.Loglike_Test(iter) = mean(B(idx_test).*log(max(ProbAve(idx_test),realmin))+(1-B(idx_test)).*log(max(1-ProbAve(idx_test),realmin)));
    
    
    %  rate = 1-exp(-ProbAve(idx_test));
    rate= Prob(idx_test);
    links = double(B(idx_test)>0);
    AUC(iter) = aucROC(rate,links);
    %[~,~,~,AUC(iter)] = perfcurve(links,rate,1);
    
    
    
    
    z=zeros(1,N);
    
    %if iter<Burnin
    % [~,rdex]=sort(sum(mi_dot_k,2));
    %  [~,rdex]=sort(sum(m_i_k_dot_dot,2));
    %   [~,rdex]=sort(r_k);
    [~,rdex]=sort(-sum(m_dot_k_k_dot,2));
    %rdex=1:K;
    %Phir=Phi*sqrt(diag(r));
    
    
    rrr = diag(Lambda_KK);
    
    
    
    
    
    %     n_k = sparse(z,1,1,K,1);
    %     [~,zdex]=sort(n_k);
    %     kkkk=1:K;
    %     kkkk=kkkk(zdex);
    %     yy=zeros(1,K);
    %     yy(kkkk)=1:K;
    %     z=yy(z);
    
    %n_k = sparse(z,1,1);
    [~,rdex]=sort(sum(m_i_k_dot_dot,2),'descend');
    [~,z] = max(m_i_k_dot_dot(rdex,:),[],1);
    [~,Rankdex]=sort(z);
    output.K_hardassignment(iter) = length(unique(z));
    
    %  end
    %  Rankdex = 1:N;
    if mod(iter,100)==0 && IsDisplay
        
        
        subplot(2,3,1);imagesc(AA(Rankdex,Rankdex));title(num2str(EPS));
        subplot(2,3,2);imagesc(ProbAve(Rankdex,Rankdex));
        subplot(2,3,3);imagesc(log(Phi(Rankdex,rdex)*Lambda_KK(rdex,rdex)+1e-2));
        subplot(2,3,4);
        imagesc(log(Lambda_KK(rdex,rdex)+0.001))
        subplot(2,3,5);plot(1:iter,output.K_positive(1:iter),1:iter,output.K_hardassignment(1:iter));title(num2str(Epsilon));
        %         if iter>Burnin
        %             subplot(2,3,5);plot(Burnin:iter,output.Loglike_Train(Burnin:iter),'b',Burnin:iter,output.Loglike_Test(Burnin:iter),'r');
        %         end
        subplot(2,3,6);plot(AUC(1:iter));title(num2str(gamma0));
        %           if mod(iter,100)==0
        %               Network_plot;
        %           end
        drawnow;
        
    end
    
    %%if output.Loglike_Train(iter)>LogLikeMax && iter>1000
    %% LogLikeMax = output.Loglike_Train(iter);
    %     if output.K_positive(iter)<=Kmin && iter>1000
    %
    %         Kmin = output.K_positive(iter);
    %               Network_plot;
    %
    %     end
    
    if mod(iter,1000)==0
        fprintf('Iter= %d, Number of Communities = %d \n',iter, output.K_positive(iter));
    end
end

rate = ProbAve(idx_test);
% zerorows = find((sum(BTrain_Mask.*(B+B'),2)==0&sum(~BTrain_Mask.*(B+B'),2)>0));
% %zerorows = find((sum(BTrain_Mask.*(B+B'),2)==0));
% for j=1:length(idx_test)
%     [i1,j1]=ind2sub([N,N],idx_test(j));
%     if nnz(i1==zerorows)>0 || nnz(j1==zerorows)>0
%         rate(j)=mean(rate);
%     end
% end

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
subplot(2,4,1); imagesc((AA(Rankdex,Rankdex)).*BTrain_Mask(Rankdex,Rankdex));
subplot(2,4,2); imagesc((1-exp(-ProbAve(Rankdex,Rankdex))).*BTrain_Mask(Rankdex,Rankdex));
subplot(2,4,3); imagesc((AA(Rankdex,Rankdex)).*~BTrain_Mask(Rankdex,Rankdex));
subplot(2,4,4);imagesc((1-exp(-ProbAve(Rankdex,Rankdex))).*~BTrain_Mask(Rankdex,Rankdex));
subplot(2,4,5);plot(sum(m_i_k_dot_dot(rdex,Rankdex)'>0,2),'*');
subplot(2,4,6);imagesc(log10(m_i_k_dot_dot(rdex,rdex)+0.1)');
subplot(2,4,7);imagesc(1-exp(-Phi(Rankdex,:)*(Lambda_KK-diag(diag(Lambda_KK)))*Phi(Rankdex,:)'));
subplot(2,4,8);imagesc(1-exp(-Phi(Rankdex,:)*(diag(diag(Lambda_KK)))*Phi(Rankdex,:)'));
