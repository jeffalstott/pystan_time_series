#The MIT License (MIT)
#
#Copyright (c) 2017 Jeff Alstott
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

import os
_stan_models_dir = os.path.join(os.path.dirname(__file__), 'stan_models')

model_names = [x[:-5] for x in os.listdir(_stan_models_dir) if x.endswith('.stan')]

class TimeSeriesModel(object):
    def __init__(self, name='VAR', **kwargs):

        name = name.replace(' ', '_').replace(',', '')
        self.name = name

        import os
        stan_models_dir = os.path.join(os.path.dirname(__file__), 'stan_models')
        try:
            self.model_code = open('%s/%s.stan'%(stan_models_dir, name), 'r').read()
        except FileNotFoundError:
            raise("This model is not defined.")

        from pickle import load
        self.model = load(open('%s/%s.pkl'%(stan_models_dir, name), 'rb'))

        import importlib.util
        spec = importlib.util.spec_from_file_location(name, '%s/%s.py'%(stan_models_dir,name))
        p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p)
        self.stan_data_creator = p.stan_data_creator
        self.parameter_priors = p.parameter_priors

        from numpy import unique
        self.model_parameters = unique([i.split('_prior')[0] for i in p.parameter_priors.keys()])

        self.stan_data = self.stan_data_creator(self, **kwargs)

    def sampling(self, **kwargs):
        if 'pars' not in kwargs.keys():
            kwargs['pars'] = [p for p in self.model_parameters]
            from numpy import isnan
            for k in self.stan_data.keys():
                if isnan(self.stan_data[k]).any():
                    kwargs['pars'] += [k+'_latent']
        self.fit = self.model.sampling(data=self.stan_data, **kwargs)


