# %%
import docplex.mp.model as md
from docplex.mp.conflict_refiner import ConflictRefiner
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.patches as pch
import matplotlib.pyplot as plt

from ga_platoon import max_dt
from scipy.optimize import curve_fit

class LMPband():
    def __init__(self,phase,cycle,vol,pv,pgt,d,sgt,ison,tau,taub,qb,qb_x,cap,ex,dwt,
                 lin,be,spc,spv,spcb,spvb,spd_on,spd_in) -> None:
        self.phase=phase
        self.cycle=cycle
        self.vol=vol
        self.pv=pv
        self.pgt=pgt
        self.d=d
        self.sgt=sgt
        self.ison=ison
        self.tau=tau
        self.taub=taub
        self.qb=qb
        self.ex=ex
        self.dwt=dwt
        self.spc=spc
        self.spv=spv
        self.spcb=spcb
        self.spvb=spvb
        self.linspace=lin
        self.be=be
        self.spd_on=spd_on
        self.spd_in=spd_in
        self.qb_x=qb_x
        self.cap=cap

        self.sg=np.array([(self.sgt[i]*self.phase).sum(axis=1) for i in range(len(self.sgt))])
        self.g=np.array([(self.pgt[i]*phase).sum(axis=1) for i in range(len(self.pgt))])
        
        
        self.rho=self.vol[0]/self.vol[1]
        self.num=len(vol[0])
        self.numr=len(self.pv)
        self.nump=len(self.phase[0])
        self.lin_num=len(self.linspace)
        self.M=1e6
        self.nx=1e-8

        self.model=md.Model("LMPband")
        self.mdl=md.Model("LMPband_variable")

        # self.GetProporation()
        self.dwm=np.array([[22.60083057,13.46496839,24.31,23.02457646,25.0084692,5.65743377,26.18040372,21.45,
                            21.30610731,15.91674887,10.69071851,15.91674887,21.15051345],
                           [27.90700128,24.76865382,26,33.1819511,33.15,10.60465754,31.59,21.45,30.94793897,
                            23.94176338,17.14793366,23.94176338,30.74566144]])
        
    def func(self,x,mu,sigma,N,D):
        return np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))*N
    
    def get_percent(self,data_x):
        area_list=np.array([[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50]])
        area_list=pd.DataFrame(area_list)
        area_list.columns=['ls','us']
        temp=list()
        for each in area_list.iterrows():
            temp.append(data_x[(data_x['speed']>each[1]['ls'])&(data_x['speed']<each[1]['us'])]['speed'].count())
        area_list['count']=np.array(temp)
        area_list['prop']=area_list['count']/area_list['count'].sum()
        area_list['ave']=(area_list['ls']+area_list['us'])/2/3.6
        return area_list
    def get_spd_proporation(self,crnl,on):
        spd=self.spd_on
        if on==False:
            spd=self.spd_in
        tag=False
        for k in crnl:
            tag|=spd["name"]==k
        dx=spd[tag]
        area_list=self.get_percent(dx)
        propt,_=curve_fit(self.func,area_list['ave'],area_list['prop'],bounds=[0,[15.,10,1000,1000.]],maxfev=100000)
        return propt.round(4)
    def get_subs(self,p):
        num=self.num
        p=np.array([p[i] for i in range(num)],dtype=int)
        pcum=p.cumsum()+1
        return pcum
    def get_subcrs(self):
        pcum=self.pcum
        return [np.where(pcum==k)[0] for k in np.unique(pcum) ]
    def get_rf(self, d, p):
        tmp = []
        for i, a in enumerate(p):
            idx = np.where(a != 0)[0]
            a = a & 0
            if len(idx) > 0:
                a[:idx[0]] = 1
            tmp.append((d[i]*a).sum())
        return tmp

    def get_grf(self,x,phase,idx,k,nump):
        f=sum([x[l,idx,k]*phase[k,l] for l in range(nump) if l!=idx])
        return f
    
    def _add_M1_car_variables(self):
        model,num,numr,nump,cycle=self.model,self.num,self.numr,self.nump,self.cycle

        Z_list = [(i) for i in range(num)]
        self.z = model.continuous_var_dict(Z_list, lb=1 /cycle[1], ub=1 / cycle[0], name="z")
        o_list = [(i) for i in range(num)]
        self.o = model.continuous_var_dict(o_list, lb=0, ub=1, name="o")
        t_list = [(i, k) for i in range(2) for k in range(num-1)]
        self.t = model.continuous_var_dict(t_list, lb=0, name="t")

        p_list = [(i) for i in range(num)]
        self.p = model.binary_var_dict(p_list, name="p")
        w_list = [(i, k) for i in range(numr) for k in range(num)]
        self.w = model.continuous_var_dict(w_list, lb=0, ub=1, name="w")
        b_list = [(i, k) for i in range(numr) for k in range(num)]
        self.b = model.continuous_var_dict(b_list, lb=0, ub=1, name="b")
        n_list = [(i, k) for i in range(numr) for k in range(num)]
        self.n = model.integer_var_dict(n_list, lb=0, ub=10, name="n")
        u_list = [(i, k) for i in range(numr) for k in range(num)]
        self.u = model.continuous_var_dict(u_list, lb=0, ub=1, name="u")
        y_list = [(i, k) for i in range(numr) for k in range(num)]
        self.y = model.binary_var_dict(y_list, name="y")
        x_list = [(l, m, k) for l in range(nump) for m in range(nump) for k in range(num)]
        self.x = model.binary_var_dict(x_list, name="x")
        r_list = [(i, k) for i in range(numr) for k in range(num)]
        self.r = model.continuous_var_dict(r_list, lb=0, ub=1, name="r")
        rb_list = [(i, k) for i in range(numr) for k in range(num)]
        self.rb = model.continuous_var_dict(rb_list, lb=0, ub=1, name="rb")

    def _add_over_phase(self,l,m,n):
        model,x,num,nump=self.model,self.x,self.num,self.nump
        for k in range(num):
            model.add_constraint(x[l,m,k]+x[n,m,k]==1)
            model.add_constraints([x[l,nk,k]+x[nk,m,k]==1 for nk in range(nump) if nk!=l and nk!=m])
            model.add_constraints([x[m,nk,k]+x[nk,n,k]==1 for nk in range(nump) if nk!=m and nk!=n])

    def _add_M1_car_constraints(self):
        model,num,t,z,d,spc,spv,x,numr,nump=self.model,self.num,self.t,self.z,self.d,self.spc,self.spv,self.x,self.numr,self.nump
        b,w,g,r,rb,pgt,phase,p,u=self.b,self.w,self.g,self.r,self.rb,self.pgt,self.phase,self.p,self.u
        M,nx,ison,be,y,rho=self.M,self.nx,self.ison,self.be,self.y,self.rho
        
        for k in range(num-1):
            model.add_constraint(d[k] /spc[1] * z[k] <= t[0, k])
            model.add_constraint(t[0, k] <= d[k]/ spc[0] * z[k])
            model.add_constraint(d[k] /spc[1] * z[k] <= t[1, k])
            model.add_constraint(t[1, k] <= d[k] /spc[0] * z[k])

        for k in range(num-2):
            model.add_constraint(d[k] /spv[0] * z[k] <= d[k] / d[k+1] * t[0, k+1] - t[0, k])
            model.add_constraint(d[k] / d[k+1] * t[0, k + 1] - t[0, k] <= d[k] / spv[1]* z[k])
            model.add_constraint(d[k] / spv[0] * z[k] <= d[k] / d[k+1] * t[1, k + 1] - t[1, k])
            model.add_constraint(d[k] / d[k] * t[1, k + 1] - t[1, k] <= d[k] / spv[1] * z[k])

        for k in range(num):
            model.add_constraints([x[l, l, k] == 1 for l in range(nump)])
            model.add_constraints([x[l, m, k] + x[m, l, k] == 1 for m in range(nump) for l in range(nump) if l != m])
            model.add_constraints([x[l, n, k] >= x[l, m, k] + x[m, n, k] - 1 for l in range(nump) for m in range(nump)
                    for n in range(nump) if l != m and l != n and n != m ] )
            
            model.add_constraint(x[2,3,k]==1)
            model.add_constraints([x[2,nk,k]+x[nk,3,k]==1 for nk in range(nump) if nk!=2 and nk!=3])

        self._add_over_phase(0,1,2)
        self._add_over_phase(5,6,7)

        for i in range(numr):
            for k in range(num):
                model.add_constraint(b[i, k] / 2 <= w[i, k])
                model.add_constraint(w[i, k]<= g[i,k]- b[i, k]/2)
        
        for i in range(numr):
            for k in range(num):
                model.add_constraints(
                    [r[i, k]<= model.sum([pgt[i, k, m]* x[l, m, k]* phase[k, l] for l in range(nump) if l != m ])
                    + M * (1 - pgt[i, k, m]) for m in range(nump) ])
                model.add_constraints(
                    [rb[i, k]<= model.sum([ pgt[i, k, m] * x[m, l, k] * phase[k, l] for l in range(nump) if l != m])
                        + M * (1 - pgt[i, k , m]) for m in range(nump) ])
                model.add_constraint(r[i, k]+ rb[i, k]+ model.sum([self.pgt[i, k, l] * phase[k, l] for l in range(nump)])== 1)

        for k in range(num-1):
            model.add_constraint(p[k] + p[k + 1] <= 1)

        for k in range(num):
            for i in range(numr):
                model.add_constraints([nx * p[k] <= u[i, k], u[i, k] <= p[k]])

        for i in range(numr):
            if ison[i]==0:
                self._add_M1_split_on_cons(i)
            else:
                self._add_M1_split_in_cons(i)

        for k in range(num):
            for i in range(numr):
                model.add_constraint(be * z[k] - M * (1 - y[i, k]) <= b[i, k])
                model.add_constraint(b[i, k] <= y[i, k])
        for k in range(num):
            model.add_constraint((1 - rho[k]) * model.sum([b[i, k]*(1-ison[i]) for i in range(numr)])>=\
                (1 - rho[k])* rho[k]* model.sum([b[i, k]*ison[i] for i in range(numr)]))
        for k in range(num-1):
            model.add_constraint(-M * p[k + 1] <= z[k + 1] - z[k])
            model.add_constraint(z[k + 1] - z[k] <= M * p[k + 1])

    def _add_M1_split_on_cons(self,i):
        model,num,o,r,w,t,u,n,y,M,p,b,g,tau=self.model,self.num,self.o,self.r,self.w,self.t,self.u,self.n,self.y,\
        self.M,self.p,self.b,self.g,self.tau
        for k in range(num-1):
            model.add_constraint(o[k] + r[i, k] + w[i, k] + t[0, k] + u[i, k+1] >=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] - M * (1 - y[i, k+1]))
            model.add_constraint(o[k] + r[i, k] + w[i, k] + t[0, k] + u[i, k+1] <=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] + M * (1 - y[i, k+1]))

            model.add_constraint(-M * p[k+1] <= y[i, k + 1] - y[i, k])
            model.add_constraint(y[i, k + 1] - y[i, k] <= M * p[k+1])

            model.add_constraints([b[i,k]/2-M*p[k+1]<=w[i,k+1],w[i,k+1]<=g[i,k+1]-b[i,k]/2+M*p[k+1]])

    def _add_M1_split_in_cons(self,i):
        model,num,o,r,w,t,u,n,y,M,p,b,g,tau=self.model,self.num,self.o,self.r,self.w,self.t,self.u,self.n,self.y,\
        self.M,self.p,self.b,self.g,self.tau
        for k in range(num-1):
            model.add_constraint(o[k] + r[i, k] + w[i, k] + n[i, k]+tau[1,k] >=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] - M * (1 - y[i, k+1]))
            model.add_constraint(o[k] + r[i, k] + w[i, k] + n[i, k]+tau[1,k] <=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] + M * (1 - y[i, k+1]))

            model.add_constraint(-M * p[k+1] <= y[i, k + 1] - y[i, k])
            model.add_constraint(y[i, k + 1] - y[i, k] <= M * p[k+1])
            
            model.add_constraints([b[i,k+1]/2-M*p[k+1]<=w[i,k],w[i,k]<=g[i,k]-b[i,k+1]/2+M*p[k+1]])
    
    def _add_M1_bus_variables(self):
        model,num=self.model,self.num
        wb_list = [(i, k) for i in range(2) for k in range(num)]
        self.wb = model.continuous_var_dict(wb_list, lb=0, ub=1, name="wb")
        bb_list = [(i, k) for i in range(2) for k in range(num)]
        self.bb = model.continuous_var_dict(bb_list, lb=0, ub=1, name="bb")
        nb_list = [(i, k) for i in range(2) for k in range(num)]
        self.nb = model.integer_var_dict(nb_list, lb=0, ub=10, name="nb")
        dw_list=[(i, k) for i in range(2) for k in range(num-1)]
        self.dw=model.continuous_var_dict(dw_list,lb=0,ub=1,name="dw")
        tb_list=[(i, k) for i in range(2) for k in range(num-1)]
        self.tb=model.continuous_var_dict(tb_list,lb=0,name="tb")
        rt_list=[(i, k) for i in range(2) for k in range(num)]
        self.rt=model.continuous_var_dict(rt_list,lb=0,ub=1,name="rt")
        rtb_list=[(i, k) for i in range(2) for k in range(num)]
        self.rtb=model.continuous_var_dict(rtb_list,lb=0,ub=1,name="rtb")
    
    def _add_M1_bus_constraints(self):
        model,num,d,spcb,spvb,tb,z,ex,dw,dwt=self.model,self.num,self.d,self.spcb,self.spvb,self.tb,\
        self.z,self.ex,self.dw,self.dwt
        rt,sgt,phase,nump,M,x,rtb,sg=self.rt,self.sgt,self.phase,self.nump,self.M,self.x,self.rtb,self.sg
        wb,bb,be,o,nb,p,taub=self.wb,self.bb,self.be,self.o,self.nb,self.p,self.taub

        for k in range(num-1):
            model.add_constraint(d[k] / spcb[1] * z[k] <= tb[0, k]-ex[k]*dwt*z[k]-dw[0,k])
            model.add_constraint(tb[0, k]-ex[k]*dwt*z[k]-dw[0,k] <= d[k] / spcb[0] * z[k])

            model.add_constraint(d[k] / spcb[1] * z[k] <= tb[1, k]-ex[k]*dwt*z[k]-dw[1,k])
            model.add_constraint(tb[1, k]-ex[k]*dwt*z[k]-dw[1,k] <= d[k] / spcb[0] * z[k])

        for k in range(num-2):
            model.add_constraint(d[k] / spvb[0] * z[k] <= 
            d[k]/d[k+1]*(tb[0, k + 1]-ex[k+1]*dwt*z[k]-dw[0,k+1])- (tb[0, k]-ex[k]*dwt*z[k]-dw[0,k]))
            model.add_constraint(d[k]/d[k+1]*(tb[0, k + 1]-ex[k+1]*dwt*z[k]-dw[0,k+1])- (tb[0, k]-ex[k]*dwt*z[k]-dw[0,k]) <=
            d[k] / spvb[1] * z[k])

            model.add_constraint(d[k] / spvb[0] * z[k] <=
            d[k]/d[k+1]*(tb[1, k + 1]-ex[k+1]*dwt*z[k]-dw[1,k+1])- (tb[1, k]-ex[k]*dwt*z[k]-dw[1,k]))
            model.add_constraint(d[k]/d[k+1]*(tb[1, k + 1]-ex[k+1]*dwt*z[k]-dw[1,k+1])-(tb[1, k]-ex[k]*dwt*z[k]-dw[1,k])<=
            d[k] / spvb[1] * z[k])

        for i in range(2):
            for k in range(num):
                model.add_constraints(
                    [rt[i, k]<= model.sum([sgt[i, k, m]* x[l, m, k]* phase[k, l] for l in range(nump) if l != m ])
                    + M * (1 - sgt[i, k, m]) for m in range(nump) ])
                model.add_constraints(
                    [rtb[i, k]<= model.sum([ sgt[i, k, m] * x[m, l, k] * phase[k, l] for l in range(nump) if l != m])
                        + M * (1 - sgt[i, k , m]) for m in range(nump) ])
                model.add_constraint(rt[i, k]+ rtb[i, k]+ sg[i,k]== 1)
        for k in range(num):
            for i in range(2):
                model.add_constraint(bb[i,k]/2<=wb[i,k]) 
                model.add_constraint(wb[i,k]<=sg[i,k]-bb[i,k]/2)      

        # for i in range(2):
        #     model.add_constraints([bb[i,k]>=be*z[k] for k in range(num)])

        for k in range(num-1):
            for i in range(2):
                model.add_constraints([dw[i,k]<=ex[k]*15*z[k]])

        for k in range(num-1):
            model.add_constraint(o[k] + rt[0, k] + wb[0, k] + tb[0, k] <=
                                    o[k + 1] + rt[0, k+1] + wb[0, k + 1] +taub[0,k+1]+ nb[0, k + 1]+M*p[k+1])
            model.add_constraint(o[k] + rt[0, k] + wb[0, k] + tb[0, k]  >=
                                    o[k + 1] + rt[0, k+1] + wb[0, k + 1] +taub[0,k+1]+ nb[0, k + 1] - M*p[k+1])

            model.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])

            model.add_constraint(o[k] + rt[1, k] + wb[1, k] + nb[1, k] +taub[1,k]>=
                                    o[k + 1] + rt[1, k+1] + wb[1, k + 1] + tb[1, k] - M * p[k+1] )
            model.add_constraint(o[k] + rt[1, k] + wb[1, k] + nb[1, k]+taub[1,k] <=
                                    o[k + 1] + rt[1, k+1] + wb[1, k + 1] + tb[1, k] + M * p[k+1])
                
            model.add_constraints([bb[1,k+1]/2-M*p[k+1]<=wb[1,k],wb[1,k]<=sg[1,k]-bb[1,k+1]/2+M*p[k+1]])

    def _add_M1_obj(self):
        self.sum_b = self.model.sum([self.pv[i] * self.b[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_u = self.model.sum([self.pv[i] * self.u[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_p = self.model.sum([self.p[k] * (self.vol[0, k] + self.vol[1, k]) for k in range(self.num)])
        self.sum_bb= self.model.sum_vars(self.bb)*self.qb[0]
        # self.sum_bb=0

    def _M1_solve(self):
        self._add_M1_car_variables()
        self._add_M1_bus_variables()
        self._add_M1_car_constraints()
        self._add_M1_bus_constraints()
        self._add_M1_obj()
        sum_b,sum_u,sum_p,sum_bb=self.sum_b,self.sum_u,self.sum_p,self.sum_bb
        model=self.model

        refiner=ConflictRefiner()
        res=refiner.refine_conflict(model)
        print(res.display())

        model.set_multi_objective("max",[5*(sum_b+sum_bb)-3*sum_u-2*sum_p])
        # model.maximize(sum_b * 5 - sum_u * 4 - sum_p * 1)
        self.sol = model.solve(log_output=True)
        print(self.sol.solve_details)
        print("object value",self.sol.objective_value)

    def _get_M1_result(self):
        sol,p,t,y,z,r,x,rt,dw,tb=self.sol,self.p,self.t,self.y,self.z,self.r,self.x,self.rt,self.dw,self.tb
        p = sol.get_value_dict(p)
        t = sol.get_value_dict(t)
        y = sol.get_value_dict(y)
        z = sol.get_value_dict(z)
        r = sol.get_value_dict(r)
        x = sol.get_value_dict(x)
        rt=sol.get_value_dict(rt)
        dw=sol.get_value_dict(dw)
        tb=sol.get_value_dict(tb)
        return p,t,y,z,r,x,rt,dw,tb
    
    def _add_M2_car_variables(self):
        mdl,num,lin_num,numr=self.mdl,self.num,self.lin_num,self.numr

        o_list = [(i) for i in range(num)]
        self.o2 = mdl.continuous_var_dict(o_list, lb=0, ub=1, name="o")
        ### 速度波动变量
        yp_list=[(i,j,k) for i in range(2) for j in range(num) for k in range(lin_num)]
        self.yp=mdl.binary_var_dict(yp_list,name="yp")

        nt_list = [(i, j, k) for i in range(2) for j in range(num) for k in range(lin_num)]
        self.nt = mdl.integer_var_dict(nt_list, lb=0, ub=10, name="nt")

        C_list=[(i,j,k) for i in range(2) for j in range(num) for k in range(lin_num)]
        self.C=mdl.continuous_var_dict(C_list,lb=0,ub=1,name="C")

        pc_list=[(i,j,k) for i in range(2) for j in range(num) for k in range(lin_num)]
        self.pc=mdl.binary_var_dict(pc_list,name="pc")

        w_list = [(i, k) for i in range(numr) for k in range(num)]
        self.w2 = mdl.continuous_var_dict(w_list, lb=0, ub=1, name="w")

        b_list = [(i, k) for i in range(numr) for k in range(num)]
        self.b2 = mdl.continuous_var_dict(b_list, lb=0, ub=1, name="b")

        n_list = [(i, k) for i in range(numr) for k in range(num)]
        self.n2 = mdl.integer_var_dict(n_list, lb=0, ub=10, name="n")

        u_list = [(i, k) for i in range(numr) for k in range(num)]
        self.u2 = mdl.continuous_var_dict(u_list, lb=0, ub=1, name="u")

    def _add_M2_car_constraints(self):
        mdl,num,b,w,g,numr,o=self.mdl,self.num,self.b2,self.w2,self.g,self.numr,self.o2
        nx,u,ison,be,M,rho=self.nx,self.u2,self.ison,self.be,self.M,self.rho
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()

        for k in range(num):
            for i in range(numr):
                mdl.add_constraint(b[i, k] / 2<= w[i, k])
                mdl.add_constraint(w[i, k] <= g[i][k] - b[i, k] / 2)

        for k in range(num):
            for i in range(numr):
                mdl.add_constraints([nx * p[k] <= u[i, k], u[i, k] <= p[k]])
        
        mdl.add_constraint(o[0]==0)
        for i in range(numr):
            if ison[i]==0:
                self._add_M2_split_on_cons(i)
            elif ison[i]==1:
                self._add_M2_split_in_cons(i)

        for k in range(num):
            for i in range(numr):
                mdl.add_constraint(be * z[k] - M * (1 - y[i, k]) <= b[i, k])
                mdl.add_constraint(b[i, k] <= y[i, k])

        for k in range(num):
            mdl.add_constraint((1 - rho[k]) * mdl.sum([b[i, k]*(1-ison[i]) for i in range(numr)])>=\
                (1 - rho[k])* rho[k]* mdl.sum([b[i, k]*ison[i] for i in range(numr)]))

    def _add_M2_split_on_cons(self,i):
        mdl,num,o,w,u,n,M,b,g,tau=self.mdl,self.num,self.o2,self.w2,self.u2,self.n2,self.M,self.b2,self.g,self.tau
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()

        for k in range(num-1):
            mdl.add_constraint(o[k] + r[i, k] + w[i, k] + t[0, k] + u[i, k+1] >=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] - M * (1 - y[i, k+1]))
            mdl.add_constraint(o[k] + r[i, k] + w[i, k] + t[0, k] + u[i, k+1] <=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] + M * (1 - y[i, k+1]))

            mdl.add_constraints([b[i,k]/2-M*p[k+1]<=w[i,k+1],w[i,k+1]<=g[i,k+1]-b[i,k]/2+M*p[k+1]])

    def _add_M2_split_in_cons(self,i):
        mdl,num,o,w,u,n,M,b,g,tau=self.mdl,self.num,self.o2,self.w2,self.u2,self.n2,self.M,self.b2,self.g,self.tau
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()
        for k in range(num-1):
            mdl.add_constraint(o[k] + r[i, k] + w[i, k] + n[i, k]+tau[1,k] >=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] - M * (1 - y[i, k+1]))
            mdl.add_constraint(o[k] + r[i, k] + w[i, k] + n[i, k] +tau[1,k]<=
                                o[k + 1] + r[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] + M * (1 - y[i, k+1]))
            mdl.add_constraints([b[i,k+1]/2-M*p[k+1]<=w[i,k],w[i,k]<=g[i,k]-b[i,k+1]/2+M*p[k+1]])
    def _add_M2_bus_variables(self):
        mdl,num=self.mdl,self.num
        wb_list = [(i, k) for i in range(2) for k in range(num)]
        self.wb2 = mdl.continuous_var_dict(wb_list, lb=0, ub=1, name="wb")
        bb_list = [(i, k) for i in range(2) for k in range(num)]
        self.bb2 = mdl.continuous_var_dict(bb_list, lb=0, ub=1, name="bb")
        nb_list = [(i, k) for i in range(2) for k in range(num)]
        self.nb2 = mdl.integer_var_dict(nb_list, lb=0, ub=10, name="nb")

    def _add_M2_bus_constraints(self):
        mdl,num,bb,wb,sg,be=self.mdl,self.num,self.bb2,self.wb2,self.sg,self.be
        o,nb,M,taub=self.o2,self.nb2,self.M,self.taub
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()

        for k in range(num):
            for i in range(2):
                mdl.add_constraint(bb[i,k]/2<=wb[i,k]) 
                mdl.add_constraint(wb[i,k]<=sg[i,k]-bb[i,k]/2)      

        # for i in range(2):
        #     mdl.add_constraints([bb[i,k]>=be*z[k] for k in range(num)])

        for k in range(num-1):
            mdl.add_constraint(o[k] + rt[0, k] + wb[0, k] + tb[0, k] <=
                                    o[k + 1] + rt[0, k+1] + wb[0, k + 1]+taub[0,k+1] + nb[0, k + 1]+M*p[k+1])
            mdl.add_constraint(o[k] + rt[0, k] + wb[0, k] + tb[0, k]  >=
                                    o[k + 1] + rt[0, k+1] + wb[0, k + 1] +taub[0,k+1] + nb[0, k + 1] - M*p[k+1])

            mdl.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])

            mdl.add_constraint(o[k] + rt[1, k] + wb[1, k] + nb[1, k]+taub[1,k] >=
                                    o[k + 1] + rt[1, k+1] + wb[1, k + 1] + tb[1, k] - M * p[k+1] )
            mdl.add_constraint(o[k] + rt[1, k] + wb[1, k] + nb[1, k]+taub[1,k]  <=
                                    o[k + 1] + rt[1, k+1] + wb[1, k + 1] + tb[1, k] + M * p[k+1])
                
            mdl.add_constraints([bb[1,k+1]/2-M*p[k+1]<=wb[1,k],wb[1,k]<=sg[1,k]-bb[1,k+1]/2+M*p[k+1]])


    def get_dw_max(self):
        num,qb,qb_x,dwt,sg,cap,cycle=self.num,self.qb,self.qb_x,self.dwt,self.sg,self.cap,self.cycle
        a_max=[[],[]]
        for i in range(num):
            cc=cycle.mean()
            dtm=max_dt(c=cc,dw=dwt/cc,q1=qb_x[0,i],q2=qb[0],g=sg[0,i],capacity=cap[0,i])
            tmp=dtm.solve()[0]
            a_max[0].append(tmp*cc)
            dtm=max_dt(dw=dwt/120,q1=qb_x[1,i],q2=qb[0],g=sg[1,i],capacity=cap[1,i])
            tmp=dtm.solve()[0]
            a_max[1].append(tmp*cc)
        self.dwm=np.array(a_max)
        print(self.dwm)

    def getprop(self,linspace1,linspace2,mu,sigma):
        t1=stats.norm(mu,sigma).cdf(linspace1)
        t2=stats.norm(mu,sigma).cdf(linspace2)
        return t2-t1

    def GetProporation(self,mu,sigma):
        return self.getprop(self.linspace-0.25,self.linspace+0.25,mu,sigma).round(4)

    def _add_var_on_cons(self,A, B, o, r, g, t, n, k, end, yp, px, p,pc,C,onbound,z):
        if k>=end:
            self.mdl.add_constraint(pc[k-1]==yp[k-1])
            return onbound
        else:
            if p[k]==1:
                A1=o[k]+r[k]+n[k]
                B1=o[k]+r[k]+n[k]+g[k]
            else:
                A1=self.mdl.max(A+t[k-1]-px[k-1],o[k]+r[k]+n[k])
                B1=self.mdl.min(B+t[k-1], o[k]+r[k]+n[k]+g[k])

            self.mdl.add_if_then(pc[k]==1,C[k]==B1-A1)
            self.mdl.add_if_then(pc[k]==0,C[k]==0)

            self.mdl.add_constraints([self.be*z[k]-self.M*(1-yp[k]) <= B1-A1, B1-A1 <= g[k]+self.M*(1-yp[k])])
            self.mdl.add_constraints([p[k]>=pc[k-1],yp[k-1]>=pc[k-1],pc[k-1]>=p[k]+yp[k-1]-1])
            onbound.append([A1,B1,B1-A1])
            return self._add_var_on_cons(A1, B1, o, r, g, t, n, k+1, end, yp, px, p,pc,C,onbound,z)

    def _add_var_in_cons(self,A, B, o, r, g, t, n, k, end, yp, px, p,pc,C,inbound,z):
        if k<=end:
            self.mdl.add_constraints([p[k+1]>=pc[k+1],yp[k+1]>=pc[k+1],p[k+1]+yp[k+1]-1<=pc[k+1]])
            return inbound
        else:
            A1=self.mdl.max(A+t[k]-px[k],o[k]+r[k]+n[k])
            B1=self.mdl.min(B+t[k], o[k]+r[k]+n[k]+g[k])
            
            self.mdl.add_constraints([self.be*z[k]-self.M*(1-yp[k]) <= B1-A1, B1-A1 <= g[k]+self.M*(1-yp[k])])
            self.mdl.add_constraints([p[k+1]>=pc[k+1],yp[k+1]>=pc[k+1],p[k+1]+yp[k+1]-1<=pc[k+1]])

            self.mdl.add_if_then(pc[k]==1,C[k]==B1-A1)
            self.mdl.add_if_then(pc[k]==0,C[k]==0)
            inbound.append([A1,B1,B1-A1])
            if p[k]==1:
                A1=o[k]+r[k]+n[k]
                B1=o[k]+r[k]+n[k]+g[k]
            return self._add_var_in_cons(A1, B1, o, r, g, t, n, k-1, end, yp, px, p,pc,C,inbound,z)
        
    def _add_M2_var_constraints(self):
        mdl,linspace,o,phase,nump,sg,pc,C=self.mdl,self.linspace,self.o2,self.phase,self.nump,self.sg,self.pc,self.C
        num,vol,d,nt,yp,tau=self.num,self.vol,self.d,self.nt,self.yp,self.tau
        M,lin_num=self.M,self.lin_num
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()
        self.sum_on=0
        self.sum_in=0
        onbound_x=[]
        inbound_x=[]

        self.pcum=self.get_subs(p)
        subcrs=self.get_subcrs()
        props=[[],[]]
        for i in range(len(subcrs)):
            propt=self.get_spd_proporation(subcrs[i],True)
            pro=self.GetProporation(propt[0],propt[1])
            props[0].append(pro) 

            propt=self.get_spd_proporation(subcrs[i],False)
            pro=self.GetProporation(propt[0],propt[1])
            props[1].append(pro) 
        print(props)

        for i,v in enumerate(linspace):
            A_on_0=o[0]+min(self.get_grf(x,phase,2,0,nump),self.get_grf(x,phase,3,0,nump))
            B_on_0=A_on_0+sg[0,0]
            mdl.add_if_then(pc[0,0,i]==1,C[0,0,i]==B_on_0-A_on_0)
            mdl.add_if_then(pc[0,0,i]==0,C[0,0,i]==0)
            onb=self._add_var_on_cons(
                A=A_on_0,
                B=B_on_0,
                o=o,
                r=[min(self.get_grf(x,phase,2,k,nump),self.get_grf(x,phase,3,k,nump)) for k in range(num)],
                g=sg[0],
                t=np.array([d[j]/v*z[j] for j in range(num-1)]),
                n=np.array([nt[0,j,i] for j in range(num)]),
                k=1,
                end=num,
                yp=np.array([yp[0,j,i] for j in range(num)]),
                px=tau[0],
                p=np.array([p[k] for k in range(num)]),
                pc=np.array([pc[0,j,i] for j in range(num)]),
                C=[C[0,k,i] for k in range(num)],
                onbound=[[A_on_0,B_on_0,o[0]+sg[0,0]]],
                z=z
            )
            self.sum_on+=mdl.sum([C[0,k,i]*vol[0,k]*props[0][self.pcum[k]-1][i] for k in range(num)])
            onbound_x.append(onb)
            
            A_in_0=o[num-1]+min(self.get_grf(x,phase,3,num-1,nump),self.get_grf(x,phase,4,num-1,nump))
            B_in_0=A_in_0+sg[1,num-1]
            mdl.add_if_then(pc[1,num-1,i]==1,C[1,num-1,i]==B_on_0-A_on_0)
            mdl.add_if_then(pc[1,num-1,i]==0,C[1,num-1,i]==0)

            inb=self._add_var_in_cons(
                A=A_in_0,
                B=B_in_0,
                o=o,
                r=[min(self.get_grf(x,phase,3,k,nump),self.get_grf(x,phase,4,k,nump)) for k in range(num)],
                g=sg[1],
                t=np.array([d[j]/v*z[j] for j in range(num-1)]),
                n=np.array([nt[1,j,i] for j in range(num)]),
                k=num-2,
                end=-1,
                yp=np.array([yp[1,j,i] for j in range(num)]),
                px=tau[1],
                p=p,
                pc=np.array([pc[1,j,i] for j in range(num)]),
                C=[C[1,k,i] for k in range(num)],    
                inbound=[[A_in_0,B_in_0,o[0]+sg[1,num-1]]],
                z=z
            )
            self.sum_in+=mdl.sum([C[1,k,i]*vol[1,k]*props[1][self.pcum[k]-1][i] for k in range(num)])
            inbound_x.append(inb)

        for k in range(num-1):
            for v in range(lin_num):
                mdl.add_constraints([-M*p[k+1] <= yp[0, k, v]-yp[0, k+1, v], yp[0, k, v]-yp[0, k+1, v] <= M*p[k+1]])
                mdl.add_constraints([-M*p[k+1] <= yp[1, k, v]-yp[1, k+1, v], yp[1, k, v]-yp[1, k+1, v] <= M*p[k+1]])

    def _add_M2_obj(self):
        mdl=self.mdl
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()
        self.sum_b2 = mdl.sum([self.pv[i] * self.b2[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_u2 = mdl.sum([self.pv[i] * self.u2[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_bb2= mdl.sum_vars(self.bb2)*self.qb[0]
        self.sum_v = self.sum_in+self.sum_on
        self.sum_p2 = mdl.sum([p[k] * (self.vol[0, k] + self.vol[1, k]) for k in range(self.num)])


    def _M2_solve(self):
        self._add_M2_car_variables()
        self._add_M2_bus_variables()
        self._add_M2_car_constraints()
        self._add_M2_bus_constraints()
        self._add_M2_var_constraints()
        self._add_M2_obj()
        mdl,sum_b,sum_u,sum_bb,sum_v,sum_p=self.mdl,self.sum_b2,self.sum_u2,self.sum_bb2,self.sum_v,self.sum_p2
        mdl=self.mdl

        refiner=ConflictRefiner()
        res=refiner.refine_conflict(mdl)
        print(res.display())

        mdl.set_multi_objective("max",[5*(sum_b+sum_bb)-3*sum_u-2*sum_p,sum_v],priorities=[2,1],weights=[1,1])
        # mdl.set_multi_objective("max",[sum_b+sum_bb,sum_u],weights=[5,-4])

        self.solution = mdl.solve(log_output=True)
        print(self.solution.solve_details)
        print("object value",self.solution.objective_value)
        
    def get_dataframe(self):
        sol,o,w,n,u,b,yp,pc,nt,C,bb,wb,nb=self.solution,self.o2,self.w2,self.n2,self.u2,self.b2,self.yp,self.pc,\
        self.nt,self.C,self.bb2,self.wb2,self.nb2
        num,numr,dwt,dw,d=self.num,self.numr,self.dwt,self.dw,self.d
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()
        o = sol.get_value_dict(o)
        w = sol.get_value_dict(w)
        n = sol.get_value_dict(n)
        u = sol.get_value_dict(u)
        b = sol.get_value_dict(b)
        yp = sol.get_value_dict(yp)
        pc=sol.get_value_dict(pc)
        nt=sol.get_value_dict(nt)
        C=sol.get_value_dict(C)
        bb=sol.get_value_dict(bb)
        wb=sol.get_value_dict(wb)
        nb=sol.get_value_dict(nb)

        Df=[[i for i in range(1,num+1)]]
        Df+=[[d[i] for i in range(num-1)] + [np.nan]]
        Df+=[[b[i,k] for k in range(num)] for i in range(numr)]
        Df+=[[o[k] for k in range(num)]]
        Df+=[[p[k] for k in range(num)]]
        Df+=[[t[i,k] for k in range(num-1)]+[np.nan] for i in range(2)]
        Df+=[[y[i,k] for k in range(num)] for i in range(numr)]
        Df+=[[1/z[k] for k in range(num)]]
        Df+=[[u[i,k] for k in range(num)] for i in range(numr)]
        # Df+=[[yp[i,k,j] for k in range(num)] for j in range(lin_num) for i in range(2)]
        Df+=[[bb[i,k] for k in range(num)] for i in range(2)]
        Df+=[[(dw[i,k])/z[k]+dwt for k in range(num-1)]+[np.nan] for i in range(2)]
        Df+=[[tb[i,k]/z[k] for k in range(num-1)]+[np.nan] for i in range(2)]
        Df=np.array(Df)
        Df=Df.T
        Df=pd.DataFrame(Df)
        cols=["cross_number"]
        cols+=["distance"]
        cols+=["b"+str(i) for i in range(1,numr+1)]
        cols+=["offset","p"]
        cols+=["t"+str(i) for i in range(1,3)]
        cols+=["y"+str(i) for i in range(1,numr+1)]
        cols+=["z"]
        cols+=["u"+str(i) for i in range(1,numr+1)]

        cols+=["bb"+str(i) for i in range(1,3)]
        cols+=["dw"+str(i) for i in range(1,3)]
        cols+=["tb"+str(i) for i in range(1,3)]

        Df.columns=cols

        Df["offset"] = Df.offset * Df.z
        Df["t1"] = Df.t1 * Df.z
        Df["t2"] = Df.t2 * Df.z
        for i in range(numr):
            Df["b"+str(i+1)]=Df.loc[:,"b"+str(i+1)]*Df.z
        Df["bb1"]=Df.bb1*Df.z
        Df["bb2"]=Df.bb2*Df.z
        Df.round(2)
        # self.Df=Df
        return Df
    def get_gst(self,phase,x):
        tmp=[phase[np.where(x<x[k])].sum() for k in range(len(x))]
        return tmp
    def get_fphase(self):
        phase,nump,num=self.phase,self.nump,self.num
        df=self.get_dataframe()
        phase=np.array([phase[i]*df.z[i] for i in range(num)])
        x=self.sol.get_value_dict(self.x)
        x_list = np.array(
            [x[l, m, k] for l in range(nump) for m in range(nump) for k in range(num)], dtype=int
        ).reshape(nump, nump, num)
        xl=np.array([x_list[:, :, i].sum(axis=0) for i in range(num)])

        tmp=np.array([self.get_gst(phase[i],xl[i]) for i in range(num)])
        return tmp
        
    
    def get_draw_dataframe(self):
        Df=self.get_dataframe()
        num,numr=self.num,self.numr
        w,wb,u,d=self.w2,self.wb2,self.u2,self.d
        p,t,y,z,r,x,rt,dw,tb=self._get_M1_result()
        Df2 = Df.copy()
        for i in range(numr):
            Df2["w"+str(i+1)]=[w[i, k] for k in range(num)]
            Df2["u"+str(i+1)]=np.array([u[i, k] for k in range(num)]) * Df.z
            Df2["car_t"+str(i+1)]=Df2.offset + r[i] * Df2.z  +Df2.loc[:,"w"+str(i+1)] * Df2.z - Df2.loc[:,"b"+str(i+1)]/ 2 

        Df2["wb1"]=[wb[0, k] for k in range(num)]
        Df2["wb2"]=[wb[1, k] for k in range(num)]
        
 
# %%
