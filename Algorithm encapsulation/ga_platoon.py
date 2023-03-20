# %%
import numpy as np
from sko.GA import GA

class max_dt:
    def __init__(self,c=100,dw=15,q1=60,q2=120,g=0.5,capacity=3,prop=0.9) -> None:
        self.dw=dw
        self.c=c
        self.q1=q1
        self.q2=q2
        self.g=g
        self.capacity=capacity
        self.prop=prop
    def get_k_patoon(self,k,dt):
        lam=self.q1+self.q2*self.g
        mu=1/(self.dw+dt)*3600/self.c
        rth=lam/mu
        # print(lam,mu,rth)
        return np.power(rth,k)*(1-rth)

    def get_obj(self,p):
        x=p[0]
        s=0
        for i in range(self.capacity+1):
            s+=self.get_k_patoon(i,x)
        return s
    def add_neq_cons(self,p):
        s=self.get_obj(p)
        return self.prop-s
    
    def solve(self):
        self.ga=GA(func=self.get_obj,n_dim=1,size_pop=50,max_iter=800,prob_mut=0.01,lb=[0],ub=[self.g],constraint_ueq=[self.add_neq_cons],precision=1e-7)
        best_x,best_y=self.ga.run()
        # print('best_x:', best_x, '\n', 'best_y:', best_y)
        return best_x

# if __name__=="_main_":
#     a=max_dt(dw=15/100,s=3600/120/100,g=0.5,capacity=3)
#     a.solve()

# %%
