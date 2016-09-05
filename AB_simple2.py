import pymc as pm
import matplotlib.pyplot as plt

true_p_A = 0.05
true_p_B = 0.04

N_A = 1500
N_B = 750

A_data = pm.rbernoulli(true_p_A, N_A)
B_data = pm.rbernoulli(true_p_B, N_B)

print("Mean of A: " , A_data.mean())
print("Mean of B: " , B_data.mean())

# priors
p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)

@pm.deterministic
def delta(p_A = p_A , p_B = p_B):
	return p_A - p_B

data_A = pm.Bernoulli("data_A", p_A , value = A_data, observed = True)
data_B = pm.Bernoulli("data_B", p_B , value = B_data, observed = True)

# mcmc
mcmc = pm.MCMC([p_A, p_B, delta, data_A, data_B])
mcmc.sample(25000, 5000)

p_A_samples = mcmc.trace("p_A")[:]
p_B_samples = mcmc.trace("p_B")[:]
delta_samples = mcmc.trace("delta")[:]

ax = plt.subplot(311)

# Ploting the posteriors

plt.xlim(0, .1)
plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_A$", color="#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right")
plt.show();
