import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
from scipy import stats

ss = np.linspace(0,8,1000)
with pm.Model():
    b = pm.Pareto('b',alpha = 2 ,m = 1) # b = prior
    obs = pm.Uniform('obs',0,upper = b,observed = 1.138)
    trace = pm.sample(20000)
tracee = trace['b']
analytical = stats.pareto.pdf(ss,3,scale = 1.138) # m in pm.pareto is scale in stats.pareto
sns.distplot(tracee,kde_kws={'color':'k'})
plt.plot(ss,analytical)
plt.xlabel('b')
plt.ylabel('p(b)')
plt.legend(['Numerical','Analytical'])
plt.show()
