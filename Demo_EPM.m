%%Demo code for Infinite Edge Partition Models (EPMs)

% dataset = 'Toy';
% dataset = 'protein230';
% dataset = 'NIPS12';
% dataset = 'NIPS234';
% dataset = 'football';

%state = 0,1,2,3,4\\

dataset = 'protein230';
state = 0;

Burnin=1500;
Collections=1500;

TrainRatio =.8;

%for state = 0:4
if strcmp(dataset,'Toy')
    rng(state,'twister');
    
    Phi = zeros(70,4);
    Phi(1:20,1)=1.5;
    Phi(11:40,2)=1.0;
    Phi(30:50,3)=1.5;
    Phi(51:70,4)=2;
    
    Phi = randg(Phi);
    
    Lambda_KK1 = ...
        [1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1]*1;
    Lambda_KK2 = ...
        [1 0 0 0
        0 1 0 0
        0 0 0 1
        0 0 1 0];

    Lambda_KK = Lambda_KK2;
    Rate = Phi*Lambda_KK*Phi' + 0;
    
    B = poissrnd(triu(Rate,1))>0;
    figure;subplot(1,2,1);imagesc(1-exp(-triu(Rate,1)-triu(Rate,1)'));subplot(1,2,2); imagesc(B+B')
    
    N = size(B,2);
    B = triu(B,1);
    
    
    figure(100);subplot(2,3,1);imagesc(1-exp(-triu(Rate,1)-triu(Rate,1)'));title('(a) Ground truth')
    subplot(2,3,2); imagesc(B+B'); title('(b) Adjacency matrix')
    
    K=min(100,N);
elseif strcmp(dataset, 'protein230')
    data = load('data/Protein230.mat')
    B = data.B;
    N = size(B,2);
    B = triu(B,1);
    K=min(100,N);
elseif strcmp(dataset,'NIPS12')
    Data = load('data/nips12raw_str602.mat')
    DocAuthor  = zeros(Data.Np,Data.Na);
    for i=1:Data.Na
        DocAuthor(Data.apapers{i},i)=1;
    end
    B = zeros(size(DocAuthor,2));
    for i=1:size(DocAuthor,2)
        B(i,:) = double(sum( DocAuthor(DocAuthor(:,i)>0,:),1)>0);
    end
    N = size(B,2);
    B = triu(B,1);
    K=min(256,N);
    
elseif strcmp(dataset,'NIPS234')
    Data = load('data/nips_1-17.mat');
    DocAuthor = Data.docs_authors;
    B = zeros(size(DocAuthor,2));
    for i=1:size(DocAuthor,2)
        B(i,:) = sum( DocAuthor(DocAuthor(:,i)>0,:),1)>0;
    end
    Coauthor_times = sum(B,1);
    [~,idex]=sort(Coauthor_times,'descend');
    B = B(idex(1:234),idex(1:234));
    N = size(B,2);
    B = triu(B,1);
    K=min(100,N);
elseif strcmp(dataset,'football')
    Data = load('data/football_corrected.mat');
    B=Data.B;
    N = size(B,2);
    B = triu(B,1);
    K=min(100,N);
elseif strcmp(dataset,'yeast')
    Data = load('data/yeast.mat');
    B=Data.Problem.A;
    N = size(B,2);
    B = triu(B,1);
    K=min(256,N);
end


%% Save data to be used by the eigenmodel R package provided by Peter Hoff

%rand('state',state);
%randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
BB=full(B+B');
BBMask=full(BTrain_Mask);
save(['results/',dataset,'_B_',num2str(state),'.mat'], 'BBMask', 'BB');
%%%Run the R code: EigenModel.R
%%
%
%     load(['R',dataset,'_B_',num2str(state),'_',num2str(K),'.mat'])
%     rate = ProbAve(idx_test);
%     figure;
%     subplot(1,2,1)
%     links = double(B(idx_test)>0);
%     [~,dex]=sort(rate,'descend');
%     subplot(2,2,1);plot(rate(dex))
%     subplot(2,2,2);plot(links(dex),'*')
%     subplot(2,2,3);
%     [X,Y,T,AUCroc] = perfcurve(links,rate,1);
%     plot(X,Y);
%     axis([0 1 0 1]), grid on, xlabel('FPR'), ylabel('TPR'), hold on;
%     x = [0:0.1:1];plot(x,x,'b--'), hold off; title(['AUCroc = ', num2str(AUCroc)])
%     subplot(2,2,4)
%     [prec, tpr, fpr, thresh] = prec_rec(rate, links,  'numThresh',3000);
%     plot([0; tpr], [1 ; prec]); % add pseudo point to complete curve
%     xlabel('recall');
%     ylabel('precision');
%     title('precision-recall graph');
%     AUCpr = trapz(tpr,prec);
%     F1= max(2*tpr.*prec./(tpr+prec));
%     title(['AUCpr = ', num2str(AUCpr), '; F1 = ', num2str(F1)])
%     figure;imagesc(normcdf(ProbProbit))
%figure(100);subplot(2,3,4);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(d) AGM')


%%  Run the models

%Datatype='Count';
Datatype='Binary';
Modeltype = 'Infinite';
%Modeltype = 'Finite';
IsDisplay = true;

%% HGP_EPM: Hierachical gamma process edge partition model
% rand('state',state);
% randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
tic
[AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z]=HGP_EPM(B,K, idx_train,idx_test,Burnin, Collections, IsDisplay, Datatype, Modeltype);
fprintf('HGP_EPM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);

if state==0
    save(['results/',dataset,num2str(state),'HGP_EPM.mat'],'AUCroc','AUCpr','F1','Phi','Lambda_KK','r_k','ProbAve','m_i_k_dot_dot','output','z');
else
    save(['results/',dataset,num2str(state),'HGP_EPM.mat'],'AUCroc','AUCpr');
end
figure(100);subplot(2,3,6);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(f) HGP-EPM')
figure;imagesc(ProbAve)


%% GP_EPM: Gamma process edge partition model
%  rand('state',state);
%  randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);

tic
[AUCroc,AUCpr,F1,Phi,r,ProbAve,mi_dot_k,output,z] = GP_EPM(B,K,idx_train,idx_test,Burnin,Collections, IsDisplay, Datatype, Modeltype);
fprintf('GP_EPM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);

if state==0
    save(['results/',dataset,num2str(state),'GP_EPM.mat'],'AUCroc','AUCpr','F1','Phi','r','ProbAve','mi_dot_k','output','z');
else
    save(['results/',dataset,num2str(state),'GP_EPM.mat'],'AUCroc','AUCpr');
end
figure(100);subplot(2,3,5);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(e) GP-EPM')

%figure;imagesc(ProbAve)



%% GP_AGM: Gamma process affiliation graph model
%rand('state',state);
%randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
figure
tic
[AUCroc,AUCpr,F1,Phi,r,ProbAve,mi_dot_k,output,z] = GP_AGM(B,K,idx_train,idx_test,Burnin,Collections, IsDisplay, Datatype, Modeltype);
fprintf('GP_AGM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);

if state==0
    save(['results/',dataset,num2str(state),'GP_AGM.mat'],'AUCroc','AUCpr','F1','Phi','r','ProbAve','mi_dot_k','output','z');
else
    save(['results/',dataset,num2str(state),'GP_AGM.mat'],'AUCroc','AUCpr');
end
figure(100);subplot(2,3,4);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(d) AGM')



%% IRM: Infinite relational model
% rand('state',state);
%  randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
tic
[AUCroc,AUCpr,F1,ProbAve,z]=irm_CRP(B,K, idx_train,idx_test,Burnin, Collections, IsDisplay, Datatype, Modeltype);
fprintf('IRM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);

if state==0
    save(['results/',dataset,num2str(state),'IRM.mat'],'AUCroc','AUCpr','F1','ProbAve','z');
else
    save(['results/',dataset,num2str(state),'IRM.mat'],'AUCroc','AUCpr');
end

figure(100);subplot(2,3,3);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(c) IRM')

%end

if 0
    AUCroc_all = zeros(5,4);
    AUCpr_all = zeros(5,4);
    models = {'HGP_EPM','GP_EPM','GP_AGM','IRM'}
    for state=0:4
        for i=1:4
            load([dataset,num2str(state),models{i},'.mat'],'AUCroc','AUCpr')
            AUCroc_all(state+1,i) =AUCroc;
            AUCpr_all(state+1,i) =AUCpr;
        end
    end
    AUCroc_ave = mean(AUCroc_all,1)
    AUCroc_sigma = std(AUCroc_all,0,1)
    AUCpr_ave = mean(AUCpr_all,1)
    AUCpr_sigma = std(AUCpr_all,0,1)
    if 0
        set(gcf,'papersize',[30 10])
        print('-dpdf','Toy_results.pdf')
    end
    
    AUC_roc_eigenmodel = zeros(5,6);
    AUC_pr_eigenmodel= zeros(5,6);
    kk=[3,5,10,25,50,4]
    for i=1:6
        for state=0:4
            
            load ([dataset,'_B_',num2str(state),'.mat'], 'BBMask', 'BB');
            B = triu(BB,1);
            idx_test = find((triu(~BBMask,1)==1));
            
            K=kk(i);
            load(['R',dataset,'_B_',num2str(state),'_',num2str(K)])
            %ProbAve=normcdf(ProbProbit);
            rate = ProbAve(idx_test);
            links = double(B(idx_test)>0);
            [~,dex]=sort(rate,'descend');
            [X,Y,T,AUCroc] = perfcurve(links,rate,1);
            [prec, tpr, fpr, thresh] = prec_rec(rate, links,  'numThresh',3000);
            AUCpr = trapz(tpr,prec);
            F1= max(2*tpr.*prec./(tpr+prec));
            AUC_roc_eigenmodel(state+1,i)=AUCroc;
            AUC_pr_eigenmodel(state+1,i)=AUCpr;
        end
    end
    mean(AUC_roc_eigenmodel,1)
    std(AUC_roc_eigenmodel,0,1)
    mean(AUC_pr_eigenmodel,1)
    std(AUC_pr_eigenmodel,0,1)
    i=6;
    state=1;
    
    addpath results
    
    load ([dataset,'_B_',num2str(state),'.mat'], 'BBMask', 'BB');
    B = triu(BB,1);
    idx_test = find((triu(~BBMask,1)==1));
    
    K=kk(i);
    load(['R',dataset,'_B_',num2str(state),'_',num2str(K)])
    
    rate = ProbAve(idx_test);
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
    AUCpr = trapz(tpr,prec);
    F1= max(2*tpr.*prec./(tpr+prec));
    title(['AUCpr = ', num2str(AUCpr), '; F1 = ', num2str(F1)])
    figure;imagesc(normcdf(ProbProbit),[0,1])
end
