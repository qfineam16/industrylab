data {
  int<lower=1> T;                   // number of observations (length)
  int<lower=1> K;                   // number of hidden states
  real y[T];                        // observations
  real hyperparam[K];               // hyperparameters for prior mean
}

parameters {
  // Discrete state model
  simplex[K] pi1;                   // initial state probabilities
  simplex[K] A[K];                  // transition probabilities
                                    // A[i][j] = p(z_t = j | z_{t-1} = i)

  // Continuous observation model
  real<lower=0> sigma[K];               // observation standard deviations

  // AR Parameters
  ordered[K] ksi0;             // Ordering prevents label switching
  real<lower=-1, upper=1> rho1[K];
}

transformed parameters {
  vector[K] logalpha[T];
  vector[K] mu_t[T];                   // observation means
  
  // Initialize at unconditional expectations
  mu_t[1, 1] = ksi0[1]/(1 - rho1[1]);         // mu for state 1
  mu_t[1, 2] = ksi0[2]/(1 - rho1[2]);         // mu for state 2
  mu_t[1, 3] = ksi0[3]/(1 - rho1[3]);         // mu for state 3
  

  // AR dynamics rolling forward
  for(t in 2:T){
    for(i in 1:K){
      mu_t[t, i] = ksi0[i] + rho1[i]*mu_t[t-1,i];
    }
  }


  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    logalpha[1] = log(pi1) + normal_lpdf(y[1] | mu_t[1], sigma);

    for (t in 2:T) {
      for (j in 1:K) { // j = current (t)
        for (i in 1:K) { // i = previous (t-1)
                         // Murphy (2012) Eq. 17.48
                         // belief state      + transition prob + local evidence at t
          accumulator[i] = logalpha[t-1, i] + log(A[i, j]) + normal_lpdf(y[t] | mu_t[t-1,i], sigma[j]);
        }
        logalpha[t, j] = log_sum_exp(accumulator);
      }
    }
  } // Forward
  

}

model {
  
  // Priors
  sigma ~ inv_gamma(0.1,0.1);
  for(j in 1:K){
    A[j] ~ beta(1,1);
  }
  
  // AR process
  ksi0 ~ normal(0,0.5);
  rho1 ~ normal(0,1);
  
  target += log_sum_exp(logalpha[T]); // Note: update based only on last logalpha
}


generated quantities{
  vector[K] alpha[T];

  for(t in 1:T){
    alpha[t] = softmax(logalpha[t]);
  }
}
