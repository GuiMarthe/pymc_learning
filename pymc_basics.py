# pymc_basics.py

import pymc as pm
lambda_ = pm.Exponential("poisson_param", 1)
data_genarator = pm.Poisson("data_genarator", lambda_)

data_plus_one = data_genarator + 1


# understanding the realtionship between parantes and children variables 
# in pymc

print("Children of Lambda_:")
print(lambda_.children)
print("\nParants of data_genarator:")
print(data_genarator.parents)
print("Children of data_genarator:")
print(data_genarator.children)


# values (possibly random)

print("values of lambda_:", lambda_.value)
print("values of data_genarator.:",  data_genarator.value)
print("values of data_plus_one.:",  data_plus_one.value)