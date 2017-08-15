# pystan_time_series
Basic time series modeling with [Stan](http://mc-stan.org/) and [Pystan](https://pystan.readthedocs.io/).

This is a small set of code to make it easy to do basic time series modeling with `stan`, and particularly with the `pystan` interface. In brief:

```
from pystan_time_series import TimeSeriesModel
Y = [your data here]
model = TimeSeriesModel(Y=Y)
model.sampling()
print(model.fit)
```

What makes `pystan_time_series` useful is that you can turn on options to modify to the model to do everything and the kitchen sink. [This Jupyter notebook](https://github.com/jeffalstott/pystan_time_series/blob/master/Examples_and_Tests.ipynb) shows the different types of models, how to use the Python interface to call them, and how the model correctly recovers the parameters of simulated data. Below is an overview of the tricks that `pystan_time_series` can do.



Things that `pystan_time_series` can model easily
====
- Many different time series
```
T = 100 #Number of time points
K = 10 #Number of time series
Y = data of shape (T,K)]

model = TimeSeriesModel(Y=Y) 
```

- missing data
```
Y[5:10] = nan #Missing data are passed as nans
model = TimeSeriesModel(Y=Y)
model.sampling()
model.fit['Y_latent'] #Includes all the observed points in Y, along with sampled estimates for what the missing observations are
```

- AR(p) models
```
model = TimeSeriesModel(Y=Y, p=2)
```


- MA(q) models
```
model = TimeSeriesModel(Y=Y, q=2)
```

- ARMA(p,q) models
```
model = TimeSeriesModel(Y=Y, p=1, q=2)
```

- Setting `p` and `q` to multiple, non-sequential lags
```
model = TimeSeriesModel(Y=Y, p=[1,5,10], q=[2,7])
```

- Multi-dimensional time series, which can be modeled as affecting are modeled together as a VAR model
```
D = 3 #Number of dimensions for each time series
Y = [data of shape (K,T,D)]
model = TimeSeriesModel(Y=Y, p=1, q=1) #All 3 dimensions affect each other at a lag of 1 (p). The moving average (q) element is separate for each dimension
```

- Setting one or more dimensions to be difference
```
D = 3
Y = [data of shape (K,T,D)]
model = TimeSeriesModel(Y=Y, difference=[0,1,0]) #Says the 2nd dimension should be differenced before modeling, making it an I(1) model
```

- Setting one or more dimensions to be monotonic
```
D = 3
Y = [data of shape (K,T,D)]
model = TimeSeriesModel(Y=Y, monotonic=[0,0,1]) #Says the 3rd dimension is monotonically increasing
```

- Partial pooling of parameter estimates across the K time series
```
model = TimeSeriesModel(Y=Y, use_partial_pooling=True)
```

- Using noise that is distributed not as a normal, but as a student's t distrbution
```
model = TimeSeriesModel(Y=Y, use_student=True)
```

- Do something crazy
```
K = 100
T = 200
D = 5
Y = [your data of shape (K,T,D)]
Y[:5] = nan
Y[100:105] = nan
Y[150:] = nan

model = TimeSeriesModel(Y=Y, p=[1,5,10], q=[1,5], partial_pooling=True, use_student=True, difference=[1,0,0,0,0], monotonic=[1,0,0,0,0])
```

Under the hood with the Stan models
====
[This directory](https://github.com/jeffalstott/pystan_time_series/tree/master/stan_models) has a collection of time series models as individual `.stan` files, each paired with a `.py` file that `TimeSeriesModel` calls for easier interfacing with the model. However, only one of these models actually matters: [VAR.stan](https://github.com/jeffalstott/pystan_time_series/blob/master/stan_models/VAR.stan). This model is fairly complex, but it implements all the capabilities of all the other models, which can then be turned on or off by passing options to the model as data. This is the `stan` model that `TimeSeriesModel` actually uses by default. The other `.stan` files are included for people that want to peruse the code of a simpler implementation, so they may better learn a model or `stan`.

_This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein._
