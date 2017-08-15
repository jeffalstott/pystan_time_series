#Copyright (c) 2017 Jeff Alstott
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#Except as contained in this notice, the name of the authors shall not be used
#in advertising or otherwise to promote the sale, use or other dealings in this
#Software without prior written authorization from the authors.
parameter_priors = {
    'mu_prior_location': 0,
    'mu_prior_scale': 4,
    'sigma_prior_location': 0,
    'sigma_prior_scale': 4,
    'phi_prior_location': 0,
    'phi_prior_scale': 4,
    'theta_prior_location': 0,
    'theta_prior_scale': 4,
    'tau_prior_location': 0,
    'tau_prior_scale': 2,
    'L_Omega_prior_scale': 1,
    'nu_prior_location': 0,
    'nu_prior_scale': 2,
    }

def stan_data_creator(self, Y, p=0, q=0,
                      monotonic=None, difference=None,
                      use_student=False,
                      return_priors=False,
                      use_partial_pooling=False,
                      **kwargs):
    if Y.ndim==2:
        from numpy import expand_dims
        Y = expand_dims(Y.T,2)
    K = Y.shape[0]
    T = Y.shape[1]
    D = Y.shape[2]

    if monotonic is None:
        from numpy import zeros
        monotonic = zeros(D).astype('int')
    elif not hasattr(monotonic, '__len__') or len(monotonic)!=D:
        from numpy import zeros
        monotonic_temp = zeros(D).astype('int')
        monotonic_temp[monotonic] = 1
        monotonic = monotonic_temp

    if difference is None:
        from numpy import zeros
        difference = zeros(D).astype('int')
    elif not hasattr(difference, '__len__') or len(difference)!=D:
        from numpy import zeros
        difference_temp = zeros(D).astype('int')
        difference_temp[difference] = 1
        difference = difference_temp

    if hasattr(p, '__len__'):
        p_lags = p
        p = len(p_lags)
    else:
        from numpy import arange
        p_lags = arange(1,p+1).astype('int')

    if hasattr(q, '__len__'):
        q_lags = q
        q = len(q_lags)
    else:
        from numpy import arange
        q_lags = arange(1,q+1).astype('int')

    stan_data = {'Y':Y,
                 'D': D,
                 'T': T,
                 'K': K,
                 'P': p,
                 'P_lags': p_lags,
                 'Q': q,
                 'Q_lags': q_lags,
                 'monotonic_indices': monotonic,
                 'diff_indices': difference,
                 'use_student': int(use_student),
                 'return_priors': int(return_priors),
                 'use_partial_pooling': int(use_partial_pooling),
                }
    stan_data = {**stan_data, **self.parameter_priors}
    for k in kwargs.keys():
        stan_data[k] = kwargs[k]


    if not hasattr(stan_data['phi_prior_location'], '__len__') or len(stan_data['phi_prior_location'])==0:
        stan_data['phi_prior_location'] = [stan_data['phi_prior_location'] for i in range(p)]
    if not hasattr(stan_data['phi_prior_scale'], '__len__') or len(stan_data['phi_prior_scale'])==0:
        stan_data['phi_prior_scale'] = [stan_data['phi_prior_scale'] for i in range(p)]

    return stan_data
