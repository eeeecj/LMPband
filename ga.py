# %%
import numpy as np
from  sko.GA import GA
import math

print(np.e)
print(math.factorial(4))

capacity=3
q=1/30*100
t=400

def get_k_poison(s,m,k):
    return (np.power(s*m,k)*(np.power(np.e,-(s*m)))/math.factorial(k))
def get_obj(p):
    x=p[0]
    s=0
    for i in range(capacity+1):
        s+=get_k_poison(x,q,i)
    return s
def add_neq_cons(p):
    s=get_obj(p)
    return 0.95-s

ga=GA(func=get_obj,n_dim=1,size_pop=50,max_iter=800,prob_mut=0.001,lb=[0],ub=[0.5],constraint_ueq=[add_neq_cons],precision=1e-6)
best_x,best_y=ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

# %%
