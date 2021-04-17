import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

with pm.Model():
    m = pm.Uniform('m',0,33) # m = prior
    obs = pm.Normal('obs',mu = m,sigma=3.3,observed=-5.4)
    trace = pm.sample(20000,step)
sns.distplot(trace['m'])
plt.hlines(1/33,0,33,colors='r')
plt.legend(['posterior','prior'])
plt.show()
