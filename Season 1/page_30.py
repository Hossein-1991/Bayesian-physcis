import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np

with pm.Model():
    f = pm.Uniform('f',0,1) # f = prior
    obs = pm.Binomial('obs',n=2127,p=f,observed = 2) # observed = r
    trace = pm.sample(20000)
sns.distplot(trace['f'])
plt.show()
