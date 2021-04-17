import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np

with pm.Model():
    s = pm.Uniform('s',0,10**7) # s = prior
    likelihood = pm.Poisson('likelihood',mu = s,observed=4)
    trace = pm.sample(20000)
tracee = trace ['s']
a = az.hdi(tracee,hdi_prob=0.95, round_to = 2)
b = [np.mean(tracee),np.median(tracee),np.std(tracee)]
print ('95% HD interval: ',a)
print ('mean:',b[0], 'median:',b[1],'std:',b[2])
sns.distplot(tracee)
plt.show()
