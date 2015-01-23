/*==========================================================
 * Multrnd_mik1k2j_mex.c - 
 *
 *
 * The calling syntax is:
 *
 *		[m_i_k_dot_dot, m_dot_k_k_dot]=Multrnd_mik1k2j(sparse(ii,jj,M,N,N),Phi,Lambda_KK);
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2014 Mingyuan Zhou
 *
 *========================================================*/

#include "mex.h"
#include "string.h"
#include <math.h>
#include <stdlib.h>
/* //#include <math.h>
//#include "cokus.c"
//#define RAND_MAX_32 4294967295.0
//#include <stdio.h>
//#include "matrix.h"*/


/* //  The computational routine 
*/

mwIndex BinarySearch(double probrnd, double *prob_cumsum, mwSize Ksize) {
    mwIndex k, kstart, kend;
    if (probrnd <=prob_cumsum[0])
        return(0);
    else {
        for (kstart=1, kend=Ksize-1; ; ) {
            if (kstart >= kend) {
                /*//k = kend;*/
                return(kend);
            }
            else {
                k = kstart+ (kend-kstart)/2;
                if (prob_cumsum[k-1]>probrnd && prob_cumsum[k]>probrnd)
                    kend = k-1;
                else if (prob_cumsum[k-1]<probrnd && prob_cumsum[k]<probrnd)
                    kstart = k+1;
                else
                    return(k);
            }
        }
    }
    return(k);
}


void Multrnd_Matrix(double *m_i_k_dot_dot, double *m_dot_k_k_dot, double *Phi, double *Lambda_KK, mwIndex *ir, mwIndex *jc, double *pr,  mwSize Nsize, mwSize Ksize,  double *prob_cumsum) 
{    
  
    double cum_sum, probrnd;
    mwIndex k, k1, k2, j, i, token,total=0;
	/*//, ksave;*/
    mwIndex starting_row_index, stopping_row_index, current_row_index;
    
    
    for (j=0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                i = ir[current_row_index];   
                for (cum_sum=0, k=0, k1=0; k1<Ksize; k1++){
                    for (k2=0; k2<Ksize; k2++) { 
                        cum_sum += Phi[i+ k1*Nsize]*Phi[j+ k2*Nsize]*Lambda_KK[k1+k2*Ksize];
                        prob_cumsum[k] = cum_sum;
                        k++;
                    }
                }
                for (token=0;token< pr[total];token++) {
                    /*//probrnd = RND[ji]*cum_sum;*/
                    
                    
                    probrnd = (double)rand()/(double)RAND_MAX*cum_sum;
                  /* // probrnd = (double) randomMT()/RAND_MAX_32*cum_sum;
                  //  probrnd = (double) cum_sum * (double) randomMT() / (double) 4294967296.0;
                //    mexCallMATLAB(1, lhsPtr, 1,  rhsPtr, "rand");
                //  mexPrintf("%f\n",drand48());
                //    probrnd =  *mxGetPr(lhsPtr[0]) *cum_sum;*/
                    
                            
                    k = BinarySearch(probrnd, prob_cumsum, Ksize*Ksize);   
                    
                    
                    k1 = k/Ksize;
                    k2 = k%Ksize;
                    
                    /*//if(ksave!=k)
                    //  mexPrintf("%d,%d, %d\n",k,k1,k2);*/
                    m_i_k_dot_dot[k1+ i*Ksize]++;
                    m_i_k_dot_dot[k2+ j*Ksize]++;
                    m_dot_k_k_dot[k1+k2*Ksize]++;
                    m_dot_k_k_dot[k2+k1*Ksize]++;
                }
                total++;
            }
        }
    }
   
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *m_i_k_dot_dot, *m_dot_k_k_dot, *Phi, *Lambda_KK;
    double  *pr, *prob_cumsum;
    mwIndex *ir, *jc;
    mwIndex Nsize, Ksize;
    
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    Nsize = mxGetM(prhs[0]);
    Ksize = mxGetN(prhs[1]);
    Phi = mxGetPr(prhs[1]);
    Lambda_KK = mxGetPr(prhs[2]);
    
    
    plhs[0] = mxCreateDoubleMatrix(Ksize,Nsize,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Ksize,Ksize,mxREAL);
    m_i_k_dot_dot = mxGetPr(plhs[0]);
    m_dot_k_k_dot = mxGetPr(plhs[1]);
    
    prob_cumsum = (double *) mxCalloc(Ksize*Ksize,sizeof(double));

    Multrnd_Matrix(m_i_k_dot_dot, m_dot_k_k_dot, Phi, Lambda_KK, ir, jc, pr, Nsize, Ksize,  prob_cumsum);
    mxFree(prob_cumsum);
}