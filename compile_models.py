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

from pystan import StanModel
from pickle import dump
from os import listdir

dirlist = listdir('stan_models/')

for f in dirlist:
    if f.endswith('VAR.stan'):
        print(f)
        compiled_model = StanModel(model_code=open('stan_models/%s'%f).read())
        dump(compiled_model, open('stan_models/%s.pkl'%f[:-5], 'wb'))


