import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np

a = np.random.poisson(lam = 4 , size = (1,40))
d = a.ravel()
n = [1,4,10,40]
t = 1
for i in n:
    data = d[:i]
    with pm.Model():
        s = pm.Uniform('s',0,10**7) # s = prior
        likelihood = pm.Poisson('likelihood',mu = s,observed=data)
        trace = pm.sample(20000)
        plt.figure(figsize=(10,5))
        plt.subplot(2,2,t)
        plt.xlim(-1,20)
        plt.ylim(0,2)
        plt.yticks([])
        plt.axvline(4,0,1,color='r')
        sns.distplot(trace['s'])
        t = t + 1
plt.show()
