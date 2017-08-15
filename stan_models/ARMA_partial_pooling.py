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
    'sigma_prior_scale': 2,
    'phi_prior_location': 0,
    'phi_prior_scale': 1,
    'theta_prior_location': 0,
    'theta_prior_scale': 1,
    'tau_prior_location': 0,
    'tau_prior_scale': 2,
    'L_Omega_prior_scale': 1,
    }

def stan_data_creator(self, Y, p=1, q=1, **kwargs):
    T = Y.shape[0]
    K = Y.shape[1]

    stan_data = {'Y':Y,
                 'T': T,
                 'K': K,
                 'P': p,
                 'Q': q,
                }
    stan_data = {**stan_data, **self.parameter_priors}
    for k in kwargs.keys():
        stan_data[k] = kwargs[k]


    if not hasattr(stan_data['phi_prior_location'], '__len__') or len(stan_data['phi_prior_location'])==0:
        stan_data['phi_prior_location'] = [stan_data['phi_prior_location'] for i in range(p)]
    if not hasattr(stan_data['phi_prior_scale'], '__len__') or len(stan_data['phi_prior_scale'])==0:
        stan_data['phi_prior_scale'] = [stan_data['phi_prior_scale'] for i in range(p)]

    return stan_data
