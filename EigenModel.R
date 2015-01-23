
#install.packages("eigenmodel")
#install.packages('R.matlab')

library(eigenmodel)
library(R.matlab)
# ##
# readMat('/data/Protein.mat')
# Len=max(max(data$train.i),max(data$train.j))
# Y = matrix(0, nrow = Len, ncol = Len)
# for (i in 1:length(data$train.i))
# {
#   Y[data$train.i[i],data$train.j[i]]=data$train.v[i]
# }
# Y = Y+t(Y)
# fit = eigenmodel_mcmc(Y, X = NULL, R = 2, S = 1500, seed = 1, burn = 500)
# 
# 
#for (strings in c('B1.mat', 'B2.mat',  'B3.mat',  'B6.mat',  'B7.mat'))
dataset = 'Toy1'
dataset = 'NIPS234'
dataset = 'Protein230'
dataset = 'NIPS12'
for (K in c(5,10,25))
{
  for (state in 0:4)
  #for (K in c(3,5,10,25,50))
    
  {
    strings = paste0(dataset,'_B_',state,'.mat')
    data = readMat(strings)
    for (i in 1:dim(data$BB)[1])
    {
      for (j in 1:dim(data$BB)[2])
      {
        if ( (i!=j) & (data$BBMask[i,j]==0 ))
        {
          data$BB[i,j]=NA
        }
        if (i==j)
        {
          data$BB[i,j]=NA
        }
      }
    }
    
    fit = eigenmodel_mcmc(data$BB, X = NULL, R = K, S = 3000, seed = 1, burn = 1500)
    
    writeMat(paste0('R',dataset,'_B_',state,'_',K,'.mat'),ProbAve=fit$Y_postmean, ProbProbit = fit$ULU_postmean)
  }
}
