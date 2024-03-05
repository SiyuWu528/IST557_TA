#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pystan')


# In[2]:


import pandas as pd
import cmdstanpy
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./golden_retrievers.csv')

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Assuming the dataset has a single column with weights, let's prepare the data for Stan:
weights = df['weight'].values # Replace 'weight' with the actual column name if different
data = {'N': len(weights), 'weights': weights}


# In[3]:


stan_model_code = """
data {
  int<lower=0> N; // Number of observations
  vector[N] weights; // Observed weights
}

parameters {
  ordered[2] mean_weights;
  real<lower=0> sd_female; // Standard deviation for female dogs
  real<lower=0> sd_male; // Standard deviation for male dogs
  simplex[2] mix_proportions; // Mixing proportions for the two distributions
}

model {
  // Priors
  mean_weights[1] ~ normal(0, 50); // Prior for mean_female, now mean_weights[1]
  mean_weights[2] ~ normal(0, 50); // Prior for mean_male, now mean_weights[2]
  sd_female ~ normal(0, 50);
  sd_male ~ normal(0, 50);
  
  // Likelihood
  for (n in 1:N) {
    target += log_mix(mix_proportions[1],
                      normal_lpdf(weights[n] | mean_weights[1], sd_female),
                      normal_lpdf(weights[n] | mean_weights[2], sd_male));
  }
}
"""


# In[4]:


model_file_path = 'mixture_model.stan'
with open(model_file_path, 'w') as model_file:
    model_file.write(stan_model_code)


# In[5]:


# Compile the model
model = cmdstanpy.CmdStanModel(stan_file='./mixture_model.stan')


# In[6]:


fit = model.sample(data=data, chains=4, parallel_chains=4)


# In[8]:


print("Summary for Female Mean Weight:")
print(summary_stats.loc['mean_weights[1]'])

print("\nSummary for Male Mean Weight:")
print(summary_stats.loc['mean_weights[2]'])


# In[9]:


# Assuming `fit` is your Stan model fit object from cmdstanpy
samples = fit.draws_pd()


# In[10]:


import numpy as np
import pandas as pd

# Extracting ordered mean weights samples
mean_female_samples = samples['mean_weights[1]']
mean_male_samples = samples['mean_weights[2]']

# Calculating 95% Confidence Intervals
ci_female = np.percentile(mean_female_samples, [2.5, 97.5])
ci_male = np.percentile(mean_male_samples, [2.5, 97.5])

print(f"95% CI for female golden retrievers' weight: {ci_female}")
print(f"95% CI for male golden retrievers' weight: {ci_male}")


# In[11]:


# Assuming mix_proportions_samples correctly extracts the samples for the mixing proportions
# If the structure of mix_proportions has not changed, no modification is needed here
mix_proportions_samples = samples[['mix_proportions[1]', 'mix_proportions[2]']]
mix_proportions_mean = mix_proportions_samples.mean(axis=0)

# Calculating the estimated ratio of female to male dogs
ratio_female_to_male = mix_proportions_mean['mix_proportions[1]'] / mix_proportions_mean['mix_proportions[2]']

print(f"Estimated ratio of female to male dogs: {ratio_female_to_male:.2f}")


# In[ ]:




