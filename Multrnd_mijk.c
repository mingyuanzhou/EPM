/*==========================================================
 * Multrnd_mijk.c - 
 *
 *
 * The calling syntax is:
 *
 *		
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2014 Mingyuan Zhou
 *
 *========================================================*/
/* $Revision: 0.1 $ */

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
//void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, mwIndex *ir, mwIndex *jc, double *pr, mwSize Nsize, mwSize Nsize, mwSize Ksize, double *RND, double *prob_cumsum) //, mxArray **lhsPtr, mxArray **rhsPtr)*/


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


void Multrnd_Matrix(double *m_i_dot_k,  double *Phi, double *r, mwIndex *ir, mwIndex *jc, double *pr,  mwSize Nsize, mwSize Ksize,  double *prob_cumsum) 
{    
  
    double cum_sum, probrnd;
    mwIndex k,  j, i, token,total=0;
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
                for (cum_sum=0, k=0; k<Ksize; k++){
                        cum_sum += Phi[i+ k*Nsize]*Phi[j+ k*Nsize]*r[k];
                        prob_cumsum[k] = cum_sum;
                }
                for (token=0;token< pr[total];token++) {
                    /*//probrnd = RND[ji]*cum_sum;*/
                    
                    
                    probrnd = (double)rand()/(double)RAND_MAX*cum_sum;
                  /* // probrnd = (double) randomMT()/RAND_MAX_32*cum_sum;
                  //  probrnd = (double) cum_sum * (double) randomMT() / (double) 4294967296.0;
                //    mexCallMATLAB(1, lhsPtr, 1,  rhsPtr, "rand");
                //  mexPrintf("%f\n",drand48());
                //    probrnd =  *mxGetPr(lhsPtr[0]) *cum_sum;*/
                    
                            
                    k = BinarySearch(probrnd, prob_cumsum, Ksize);   
                    
                    
                   
                    /*//if(ksave!=k)
                    //  mexPrintf("%d,%d, %d\n",k,k1,k2);*/
                    m_i_dot_k[k+ i*Ksize]++;
                    m_i_dot_k[k+ j*Ksize]++;
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
    double *m_i_dot_k, *m_dot_dot_k, *Phi, *r;
    double  *pr, *prob_cumsum;
    mwIndex *ir, *jc;
    mwIndex Nsize, Ksize;
    
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    Nsize = mxGetM(prhs[0]);
    Ksize = mxGetN(prhs[1]);
    Phi = mxGetPr(prhs[1]);
    r = mxGetPr(prhs[2]);
    
    
    plhs[0] = mxCreateDoubleMatrix(Ksize,Nsize,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Ksize,Ksize,mxREAL);
    m_i_dot_k = mxGetPr(plhs[0]);
    
    
    prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));

    Multrnd_Matrix(m_i_dot_k,  Phi, r, ir, jc, pr, Nsize, Ksize,  prob_cumsum);
    mxFree(prob_cumsum);
}