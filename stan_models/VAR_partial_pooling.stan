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
    int D; // Number of dimensions per time series
    int<lower=0,upper=T-1> P; // Number of lags for AD element
    int P_lags[P]; //Lags for AD element
    int<lower=0,upper=T-1> Q; // Number of lags for MA element
    int Q_lags[Q]; //Lags for MA element



    matrix[T, D] Y[K]; // data to model

    int<lower=0,upper=1> use_partial_pooling; //Whether to use partial pooling of parameter estimation across the K time series
    int<lower=0,upper=1> use_student; //Whether to use the Student's T distribution for the noise

    int<lower=0,upper=1> monotonic_indices[D]; //Time series' dimensions that are limited to only increasing monotonically
    int<lower=0,upper=1> diff_indices[D]; //Time series' dimensions that are to be differenced before including in model

    // priors
    real mu_prior_location;
    real mu_prior_scale;

    real sigma_prior_location;
    real sigma_prior_scale;

    vector[P] phi_prior_location;
    vector[P] phi_prior_scale;

    real theta_prior_location;
    real theta_prior_scale;

    real tau_prior_location;
    real tau_prior_scale;

    real L_Omega_prior_scale;

    real nu_prior_location;
    real nu_prior_scale;

    int<lower=0,upper=1> return_priors; //Whether to skip fitting to data and just return samples from the priors
}

transformed data {
    int n_monotonic;
    int n_missing_observations;

    int first_observation[K,D];
    int last_observation[K,D];
    int n_missing_updates_between_first_and_last;
    int n_missing_observations_before_first_and_after_last;

    n_missing_observations = 0; //This is for the non-monotonic data
    n_monotonic = sum(monotonic_indices);
    n_missing_updates_between_first_and_last = 0;
    n_missing_observations_before_first_and_after_last = 0;

    for (k in 1:K){
        for (d in 1:D){
            if (monotonic_indices[d]==0){
                for (t in 1:T){
                    if (is_nan(Y[k,t,d])){
                        n_missing_observations = n_missing_observations + 1;
                    }
                }
            }
            else{
                first_observation[k,d] = 0;
                last_observation[k,d] = 0;
                {
                int t;
                t = 1;
                while (first_observation[k,d]<1){
                    if (!is_nan(Y[k,t,d])){
                        first_observation[k,d] = t;
                    }
                    t = t + 1;
                 }

                t = T;
                while (last_observation[k,d]<1){
                    if (!is_nan(Y[k,t,d])){
                        last_observation[k,d] = t;
                    }
                    t = t - 1;
                }
                }
                n_missing_observations_before_first_and_after_last = n_missing_observations_before_first_and_after_last +
                                                                    first_observation[k,d] - 1 +
                                                                    (T-last_observation[k,d]);


                {
                int latest_observation_ind;
                int n_unobserved;
                n_unobserved = 0;
                latest_observation_ind = first_observation[k,d];
                for (t in first_observation[k,d]+1:last_observation[k,d]){
                    if (is_nan(Y[k,t,d])){
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
        }
    }


}

parameters {
    matrix[K,D] mu;
    matrix<lower=0>[K,D] sigma;
    matrix[D,D] phi[K,P];
    matrix<lower=-1, upper=1>[Q,D] theta[K];
    matrix[K,D*use_student] nu;

    cholesky_factor_corr[use_partial_pooling*(2+P*D*D+Q+use_student)] L_Omega;
    vector<lower = 0>[use_partial_pooling*(2+P*D*D+Q+use_student)] tau;
    vector[D*use_partial_pooling] mu_mu;
    vector<lower = 0>[D*use_partial_pooling] mu_sigma;
    matrix[D,D] mu_phi[P*use_partial_pooling];
    matrix<lower=-1, upper=1>[Q,D*use_partial_pooling] mu_theta;
    vector[D*use_student*use_partial_pooling] mu_nu;

    vector[n_missing_observations] latent_data;
    vector<lower=0>[n_missing_observations_before_first_and_after_last] unrestricted_updates;
    vector<lower=0, upper=1>[n_missing_updates_between_first_and_last] restricted_updates;
}

transformed parameters {
    matrix[T,D] Y_latent[K];
    matrix<lower=0>[T-1,n_monotonic] Y_latent_monotonic_change[K];

    {
    int latent_data_counter;
    int restricted_update_counter;
    int gap_width;
    real previous_value;
    int previous_value_index;
    int monotonic_d_counter;
    int unrestricted_update_counter;
    unrestricted_update_counter = 1;

    restricted_update_counter = 1;
    latent_data_counter = 1;

    for (k in 1:K){
        monotonic_d_counter = 0;
        for (d in 1:D){
            if (monotonic_indices[d]==0){
                for (t in 1:T){
                    if (is_nan(Y[k,t,d])){
                        Y_latent[k,t,d] = latent_data[latent_data_counter];
                        latent_data_counter = latent_data_counter + 1;
                    } else{
                        Y_latent[k,t,d] = Y[k,t,d];
                    }
                }
            }
            else{
                Y_latent[k,first_observation[k,d],d] = Y[k, first_observation[k,d],d];
                previous_value = Y[k, first_observation[k,d],d];
                previous_value_index = first_observation[k,d];

                for (t in first_observation[k,d]+1:last_observation[k,d]){
                    if (!is_nan(Y[k,t,d])){
                        Y_latent[k,t,d] = Y[k,t,d];

                        gap_width = t-previous_value_index;
                        if (gap_width>1){
                            // These are the unobserved UPDATES between observed time steps.
                            // I.e. If Y_3 and Y_1 are observed, by Y_2 is not, these are (Y_3 - Y_2) and (Y_2-Y_1)
                            // We will say that these updates have to sum up to the observed difference between Y_3 and Y_1.
                            // The unobserved time steps then have values that are the cumulative sum of these updates.

                            Y_latent[k,previous_value_index+1:t-1, d] =
                            cumulative_sum(
                             restricted_updates[restricted_update_counter:restricted_update_counter+gap_width-1]
                             / sum(restricted_updates[restricted_update_counter:restricted_update_counter+gap_width-1])
                             * (Y[k,t,d] - previous_value)
                             )[1:gap_width-1] + previous_value;

                            // We don't include the last update in this sum, since we already explicitly grabbed the level
                            // we ultimately get to from the data itself.

                            restricted_update_counter = restricted_update_counter + gap_width-1;
                        }
                        previous_value = Y[k,t,d];
                        previous_value_index = t;
                    }
                }
                // Fill the latent data before and after the observed data with completely unrestricted parameters

                if (first_observation[k,d]>1){
                    Y_latent[k,1:first_observation[k,d]-1, d] =
                                Y_latent[k, first_observation[k,d], d] -
                                reverse(cumulative_sum(unrestricted_updates[unrestricted_update_counter:unrestricted_update_counter-1+first_observation[k,d]-1]),
                                        first_observation[k,d]-1);
                    unrestricted_update_counter = unrestricted_update_counter + first_observation[k,d]-1;
                }
                if (last_observation[k,d]<T){
                    Y_latent[k,last_observation[k,d]+1:T, d] =
                                cumulative_sum(unrestricted_updates[unrestricted_update_counter:unrestricted_update_counter+T-last_observation[k,d]-1])
                                + Y_latent[k,last_observation[k,d], d];
                    unrestricted_update_counter = unrestricted_update_counter + T-last_observation[k,d];
                }

                monotonic_d_counter = monotonic_d_counter + 1;
                Y_latent_monotonic_change[k,:,monotonic_d_counter] = Y_latent[k,2:T,d]-Y_latent[k,1:T-1,d];
            }
        }
    }
    }



}

model {
    matrix[T,D] Y_for_fitting[K];
    matrix[T,D] err;
    matrix[T,D] expected_value;

    //Define priors and model for partial pooling, if we're using it
    if (use_partial_pooling==1){
        tau ~ student_t(4,tau_prior_location, tau_prior_scale);
        L_Omega ~ lkj_corr_cholesky(L_Omega_prior_scale);
        mu_mu ~ normal(mu_prior_location, mu_prior_scale);
        mu_sigma ~ student_t(4, sigma_prior_location, sigma_prior_scale);

        for (p in 1:P){
            for (d in 1:D){
                mu_phi[p,d] ~ normal(phi_prior_location[p], phi_prior_scale[p]);
            }
        }

        for (q in 1:Q){
            mu_theta[q] ~ normal(theta_prior_location, theta_prior_scale);
        }

        if (use_student==1){
            1.0./mu_nu ~ cauchy(nu_prior_location, nu_prior_scale);
            target += -2*log(mu_nu);
        }

        {
        vector[2*D+P*D*D+Q*D] parameters_for_time_series[K];
        vector[2*D+P*D*D+Q*D] mu_parvec;

        mu_parvec[1:2*D] = append_row(mu_nu, append_row(mu_mu, log(mu_sigma)));
        for (p in 1:P){
            mu_parvec[1+2*D+(p-1)*D*D:2*D+(p-1)*D*D+D*D] = to_vector(mu_phi[p]);
        }
        mu_parvec[2*D+P*D*D+1:] = logit((to_vector(mu_theta)+1)/2);

        for (k in 1:K){
            parameters_for_time_series[k, :2*D] = append_col(nu[k], append_col(mu[k], log(sigma[k])))';
            for (p in 1:P){
                parameters_for_time_series[k,(1+2*D+(p-1)*D*D):(2*D+p*D*D)] = to_vector(phi[k,p]);
            }
            parameters_for_time_series[k,2*D+P*D*D+1:] = logit((to_vector(theta[k])+1)/2);
        }

        parameters_for_time_series ~ multi_normal_cholesky(mu_parvec, diag_pre_multiply(tau, L_Omega));
        }
    }
    if (use_partial_pooling==0){
        for(k in 1:K){
            mu[k] ~ normal(mu_prior_location, mu_prior_scale);
            sigma[k] ~ student_t(4, sigma_prior_location, sigma_prior_scale);

            for (p in 1:P){
                for (d in 1:D){
                    phi[k,p,d] ~ normal(phi_prior_location[p], phi_prior_scale[p]);
                }
            }

            for (q in 1:Q){
                theta[k,q] ~ normal(theta_prior_location, theta_prior_scale);
            }

            if (use_student==1){
                1.0./nu[k] ~ cauchy(nu_prior_location, nu_prior_scale);
                target += -2*log(nu[k]);
            }
        }
    }


    if (return_priors==0){

    //Some of the dimensions may be supposed to be modeled as changes, not levels. 
    //For those, we change the Y to the differences.
    Y_for_fitting = Y_latent;
    for (k in 1:K){
        for (d in 1:D){
            if(diff_indices[d]==1){
                Y_for_fitting[k,2:,d] = Y_latent[k,2:,d]-Y_latent[k,1:T-1,d];
                Y_for_fitting[k,1,d] = 0;
            }
        }
    }

    //Actually model something!
    for (k in 1:K) {
        expected_value = rep_matrix(mu[k], T);

        for (p in 1:P){
            expected_value[P_lags[p]+1:] = expected_value[P_lags[p]+1:] +
                                            Y_for_fitting[k,1:T-P_lags[p]]*phi[k,p];
        }

        err = Y_for_fitting[k] - expected_value;

        if (Q>0){
            for (t in 1:T){
                for (q in 1:Q){
                    if (Q_lags[q]<t){
                        expected_value[t] = expected_value[t] + theta[k,q].*err[t-Q_lags[q]];
                    }
                }
            }

//            if (Q>1){
//                for (t in 2:Q){
//                    expected_value[t] = expected_value[t] + columns_dot_product(theta[k, 1:t-1],err[1:t-1]);
//                    err[t] = Y_for_fitting[k,t] - expected_value[t];
//                }
//            }
//            for (t in Q+1:T){
//                expected_value[t] = expected_value[t] + columns_dot_product(theta[k],err[t-Q:t-1]);
//                err[t] = Y_for_fitting[k,t] - expected_value[t];
//                }
        }
        for (d in 1:D){
            if (use_student==1){
                err[1+max(P,Q):, d] ~ student_t(nu[k,d], 0, sigma[k,d]); //could change to multinormal later, so as to have the shocks for the different dimensions potentially have correlations
            }
            else{
                err[1+max(P,Q):, d] ~ normal(0, sigma[k,d]); //could change to multinormal later, so as to have the shocks for the different dimensions potentially have correlations
            }
        }
    }
    }
}
