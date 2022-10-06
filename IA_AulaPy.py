#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def f(x):
    return x**2 * np.sin(x) * np.cos(x)

x = np.linspace(0,20,1_000)

#print(x)


# In[3]:


import plotly.graph_objects as go 

fig = go.Figure()
fig.add_trace(go.Scatter(x=x,y=f(x)))

fig.show()


# In[4]:


from scipy.optimize import dual_annealing

summary = dual_annealing(f,[(0,20)])

summary.x


# In[5]:


def cost_func(x):
    return 0.35 * x[0] / x[2] +            0.64 * x[1] / x[3] +            0.75 * x[2] ** 2 / x[3] +            0.12 * x[2] ** 2 / x[1] +            0.28 * x[3] ** 2 / x[0] +            0.39 * x[0] ** 2 / x[1]


# In[6]:


x = [2, 1, 20, 10]

cost_func(x)


# In[7]:


bounds = [(2,50),(1,150),(20,80),(10,60)]

summary = dual_annealing(cost_func,bounds)
print(summary)


# In[8]:


100*(cost_func(x)-cost_func(summary.x)) / cost_func(x)


# In[9]:


summary.x


# In[10]:


cost_func([2, 1, 20, 10])


# In[11]:


import pandas as pd 

A_list = range(1,100)
B_list = range(1,100)
C_list = range(1,100)
D_list = range(1,100)

cost_struc = {"A":[],"B":[],"C":[],"D":[], "cost":[]}
for A in A_list:
    for B in B_list:
        for C in C_list:
            for D in D_list:
                cost_struc["A"].append(A)
                cost_struc["B"].append(B)
                cost_struc["C"].append(C)
                cost_struc["D"].append(D)
                cost_struc["cost"].append(cost_func([A,B,C,D]))

df_cost = pd.DataFrame(cost_struc)


# In[ ]:


import xgboost as xgb

cost_func_ia = xgb.XGBRegressor()
cost_func_ia.fit(df_cost[["A","B","C","D"]], df_cost[["cost"]])
suma= cost_func_ia
print(suma)


# In[ ]:


import xgboost as xgb


# In[ ]:




