//Copyright (c) 2017 Jeff Alstott
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//
//Except as contained in this notice, the name of the authors shall not be used
//in advertising or otherwise to promote the sale, use or other dealings in this
//Software without prior written authorization from the authors.
data {

    int T; // number of time steps
    int K; // Number of time series
    int<lower=0,upper=T-1> P; // Number of lags for AR element
    int<lower=0,upper=T-1> Q; // Number of lags for MA element

    matrix[T, K] Y; // data to model

    // priors
    real mu_prior_location;
    real mu_prior_scale;

    real sigma_prior_location;
    real sigma_prior_scale;

    vector[P] phi_prior_location;
    vector[P] phi_prior_scale;

    real theta_prior_location;
    real theta_prior_scale;

    int<lower=0, upper=1> return_priors; //Whether to run the model at all, or just return samples from the priors
}

transformed data {
    int n_missing_observations;
    n_missing_observations = 0;
    for (k in 1:K){
        for (t in 1:T){
            if (is_nan(Y[t,k])){
                n_missing_observations = n_missing_observations + 1;
            }
        }
    }
}

parameters {
    vector[K] mu;
    vector<lower=0>[K] sigma;
    matrix[K,P] phi;
    matrix<lower=-1, upper=1>[K,Q] theta;

    vector[n_missing_observations] latent_data;
}

transformed parameters {
    matrix[T,K] Y_latent;

    {
    int latent_data_counter;
    latent_data_counter = 1;

    for (k in 1:K){
        for (t in 1:T){
            if (is_nan(Y[t,k])){
                Y_latent[t,k] = latent_data[latent_data_counter];
                latent_data_counter = latent_data_counter + 1;
            } else{
                Y_latent[t,k] = Y[t,k];
            }
        }
    }
    }
}

model {
    matrix[T,K] err;
    matrix[T,K] nu;
    matrix[T,P] Y_latent_lagged[K];

    for (k in 1:K){
        Y_latent_lagged[k] = rep_matrix(0,T,P);
        for (p in 1:P){
            Y_latent_lagged[k, p+1:T,p] = Y_latent[1:T-p,k];
        }
    }

    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);

    for (p in 1:P){
        phi[:,p] ~ normal(phi_prior_location[p], phi_prior_scale[p]);
    }

    for (q in 1:Q){
        theta[:,q] ~ normal(theta_prior_location, theta_prior_scale);
    }

    if (return_priors==0){
    for (k in 1:K) {
        nu[:,k] = rep_vector(mu[k], T);

        if (P>0){
            nu[:,k] = nu[:,k] + Y_latent_lagged[k]*phi[k]';
            }

        err[:,k] = Y_latent[:,k] - nu[:,k];

        if (Q>0){
            if (Q>1){
                for (t in 2:Q){
                    nu[t,k] = nu[t,k] + theta[k,1:t-1]*err[1:t-1, k];
                    err[t,k] = Y_latent[t,k] - nu[t,k];
                }
            }
            for (t in Q+1:T){
                nu[t,k] = nu[t,k] + theta[k]*err[t-Q:t-1,k];
                err[t,k] = Y_latent[t,k] - nu[t,k];
            }
        }

        err[max(P+1,Q+1):T,k] ~ normal(0, sigma[k]);
    }
    }
}
