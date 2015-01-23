function AcceptProb1 = CalAcceptProb1(r_k,c,gamma0,gamma0new,a_new,c_new,K)
%r_k~Gamma(gamma0/K,1./c)
%Q(gamma0new)~Gamma(e_0 + anew,1./(f_0+c_new))

AcceptProb1 = exp(sum((gamma0new/K-gamma0/K)*log(c)-(gammaln(gamma0new/K)-gammaln(gamma0/K)) + (gamma0new/K-gamma0/K)*log(max(r_k,realmin)))...
            +(a_new)*(log(gamma0)-log(gamma0new))-  c_new*(gamma0-gamma0new));
 