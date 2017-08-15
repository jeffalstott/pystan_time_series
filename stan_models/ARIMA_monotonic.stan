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
functions {
    vector reverse(vector x, int N) {
        vector[N] reversed_x;
        for (i in 1:N){
            reversed_x[i] = x[N-(i-1)];
        }
        return(reversed_x);
    }
}

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
}

transformed data{
    int first_observation[K];
    int last_observation[K];
    int n_missing_updates_between_first_and_last;
    int n_missing_observations_before_first_and_after_last;

    n_missing_updates_between_first_and_last = 0;

    for (k in 1:K){
        first_observation[k] = 0;
        last_observation[k] = 0;
        {
        int t;
        t = 1;
        while (first_observation[k]<1){
            if (!is_nan(Y[t,k])){
                first_observation[k] = t;
            }
            t = t + 1;
         }

        t = T;
        while (last_observation[k]<1){
            if (!is_nan(Y[t,k])){
                last_observation[k] = t;
            }
            t = t - 1;
        }
        }

        {
        int latest_observation_ind;
        int n_unobserved;
        n_unobserved = 0;
        latest_observation_ind = first_observation[k];
        for (t in first_observation[k]+1:last_observation[k]){
            if (is_nan(Y[t,k])){
                n_unobserved = n_unobserved + 1;
            }
            else{
                if (n_unobserved>0){
                    n_missing_updates_between_first_and_last = n_missing_updates_between_first_and_last
                                                                    + n_unobserved + 1;
                    n_unobserved = 0;
                }
            }
        }
        }
    }
    n_missing_observations_before_first_and_after_last = sum(first_observation)-K + (T*K - sum(last_observation));
}


parameters {
    vector[K] mu;
    vector<lower=0>[K] sigma;
    matrix[K,P] phi;
    matrix<lower=-1, upper=1>[K,Q] theta;

    vector<lower=0>[n_missing_observations_before_first_and_after_last] unrestricted_updates;
    vector<lower=0, upper=1>[n_missing_updates_between_first_and_last] restricted_updates;

}
transformed parameters {
    matrix[T,K] Y_latent;
    matrix<lower=0>[T-1,K] Y_latent_change;

    // Fill the latent data within the observed data with either data values or restricted parameters
    {
    int restricted_update_counter;
    int gap_width;
    real previous_value;
    int previous_value_index;

    restricted_update_counter = 1;


    for (k in 1:K){
        Y_latent[first_observation[k],k] = Y[first_observation[k],k];
        previous_value = Y[first_observation[k],k];
        previous_value_index = first_observation[k];

        for (t in first_observation[k]+1:last_observation[k]){
            if (!is_nan(Y[t,k])){
                Y_latent[t,k] = Y[t,k];

                gap_width = t-previous_value_index;
                if (gap_width>1){
                    // These are the unobserved UPDATES between observed time steps.
                    // I.e. If Y_3 and Y_1 are observed, by Y_2 is not, these are (Y_3 - Y_2) and (Y_2-Y_1)
                    // We will say that these updates have to sum up to the observed difference between Y_3 and Y_1.
                    // The unobserved time steps then have values that are the cumulative sum of these updates.

                    Y_latent[previous_value_index+1:t-1, k] =
                    cumulative_sum(
                     restricted_updates[restricted_update_counter:restricted_update_counter+gap_width-1]
                     / sum(restricted_updates[restricted_update_counter:restricted_update_counter+gap_width-1])
                     * (Y[t,k] - previous_value)
                     )[1:gap_width-1] + previous_value;

                    // We don't include the last update in this sum, since we already explicitly grabbed the level
                    // we ultimately get to from the data itself.

                    restricted_update_counter = restricted_update_counter + gap_width-1;
                }
                previous_value = Y[t,k];
                previous_value_index = t;
            }
        }
    }
    }

    // Fill the latent data before and after the observed data with completely unrestricted parameters
    {
    int unrestricted_update_counter;
    unrestricted_update_counter = 1;

    for (k in 1:K){
        if (first_observation[k]>1){
            Y_latent[1:first_observation[k]-1, k] =
                        Y_latent[first_observation[k], k] -
                        reverse(cumulative_sum(unrestricted_updates[unrestricted_update_counter:unrestricted_update_counter-1+first_observation[k]-1]),
                                first_observation[k]-1);
            unrestricted_update_counter = unrestricted_update_counter + first_observation[k]-1;
        }
        if (last_observation[k]<T){
            Y_latent[last_observation[k]+1:T, k] =
                        cumulative_sum(unrestricted_updates[unrestricted_update_counter:unrestricted_update_counter+T-last_observation[k]-1])
                        + Y_latent[last_observation[k], k];
            unrestricted_update_counter = unrestricted_update_counter + T-last_observation[k];
        }
    }
    }

    Y_latent_change = Y_latent[2:T,:]-Y_latent[1:T-1,:];
}

model {
    matrix[T,K] err;
    matrix[T,K] nu;
    matrix[T,P] Y_latent_change_lagged[K];

    for (k in 1:K){
        Y_latent_change_lagged[k] = rep_matrix(0,T,P);
        for (p in 1:P){
            Y_latent_change_lagged[k, 1+p:T,p] = Y_latent_change[1:T-p,k];
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

    for (k in 1:K) {
        nu[:,k] = rep_vector(mu[k], T);

        if (P>0){
            nu[:,k] = nu[:,k] + Y_latent_change_lagged[k]*phi[k]';
            }

        err[:,k] = Y_latent_change[:,k] - nu[:,k];

        if (Q>0){
            if (Q>1){
                for (t in 2:Q){
                    nu[t,k] = nu[t,k] + theta[k,1:t-1]*err[1:t-1, k];
                    err[t,k] = Y_latent_change[t,k] - nu[t,k];
                }
            }
            for (t in 1+Q:T){
                nu[t,k] = nu[t,k] + theta[k]*err[t-Q:t-1,k];
                err[t,k] = Y_latent_change[t,k] - nu[t,k];
                }
        }

        err[1+max(P,Q):,k] ~ normal(0, sigma[k]);
    }
}
