# simple AB
import matplotlib.pyplot as plt
import pymc as pm

p = pm.Uniform('p', lower = 0, upper = 1)

# setting constants

p_true = 0.05
N = 1500

# sampling N bern(0.05)

data = pm.rbernoulli(p_true, N)

print(data)
print(data.sum())

print("\nmean: ", data.mean())
print("\nObserved prop equal p_true.  ",  data.mean() == p_true)

obs = pm.Bernoulli("obs", p, value = data , observed = True)

# yay a quick mcmc

mcmc = pm.MCMC([p, obs])
mcmc.sample(20000, 1000)

plt.title("possible values for the true effectiveness of version A")
plt.vlines(p_true, 0, 90, linestyle = '--', label = "true $p_$A (not known)")
plt.hist( mcmc.trace("p")[:] , bins = 35, histtype = "stepfilled", normed = True)
plt.xlabel("value of $p_$A")
plt.ylabel("Density")
plt.show()






