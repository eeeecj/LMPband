import pandas as pd
import cvxpy as cp
import numpy as np
import docplex.mp.model as md
from docplex.mp.conflict_refiner import ConflictRefiner
import scipy.stats as stats
import seaborn as sns 
import matplotlib.patches as pch 
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import curve_fit
from ga_platoon import max_dt


class LMband():
    def __init__(self, phase, cycle, vol, pv, pg, d, sgt, ison, tau, taub, qb,qb_x,cap, low, up, linspace, be, lowv, upv,
                 ex, dwt,lowb,upb,lowbv,upbv,spd_on,spd_in) -> None:
        self.d = d  # 交叉口间距
        self.phase = phase  # 相位时长
        self.cycle = np.array(cycle)  # 信号周期
        self.vol = vol  # 直行流量
        self.pv = pv  # 路径流量
        self.pg = pg  # 路径通行权
        self.sgt = sgt  # 直行通行权
        self.ison = ison  # 路径是否为上行方向
        self.tau = tau  # 左转偏移量
        self.taub = taub  # 公交左转偏移量
        self.qb = qb  # 公交流量
        self.qb_x=qb_x #红灯时间时支路转入车辆
        self.cap=cap # 公交车站的容量
        self.ex = ex  # 公交车站
        self.dwt = dwt  # 平均停靠时间
        self.lin = linspace  # 速度求解空间
        self.spd_on=spd_on
        self.spd_in=spd_in

        self.rho = self.vol[0]/self.vol[1]
        self.num = len(self.vol[0])
        self.numr = len(self.pv)
        self.nump = len(self.phase[0])
        self.lin_num = len(self.lin)

        self.M = 1e6
        self.nx = 1e-8
        self.be = be
        self.spc = np.array([low, up])
        self.spv = np.array([lowv, upv])
        self.spcb=np.array([lowb,upb])
        self.spvb=np.array([lowbv,upbv])

        self.g = np.array([(self.pg[i]*self.phase).sum(axis=1) for i in range(len(self.pg))])
        self.r = 1-self.g
        self.rf = np.array([self.get_rf(self.phase, self.pg[i]) for i in range(self.numr)])
        self.sg = np.array([(self.sgt[i]*phase).sum(axis=1) for i in range(len(self.sgt))])
        self.srf = np.array([self.get_rf(self.phase, self.sgt[i]) for i in range(len(self.sgt))])

        self.model = md.Model("LMBand")
        self.md2=md.Model("variable_LMband")

        # self.GetProporation()
        # self.get_dw_max()
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

    def _add_M1_car_variables(self):
        model, num, numr, cycle = self.model, self.num, self.numr, self.cycle
        #   公共变量
        Z_list = [(i) for i in range(num)]
        self.z = model.continuous_var_dict(
            Z_list, lb=1/cycle[1], ub=1/cycle[0], name="z")
        o_list = [(i) for i in range(num)]
        self.o = model.continuous_var_dict(o_list, lb=0, ub=1, name="o")
        t_list = [(i, k) for i in range(2) for k in range(num-1)]
        self.t = model.continuous_var_dict(t_list, lb=0, name="t")

        # 长干道变量
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

    def _add_M1_car_constraints(self):
        z, o, t, p, w, b, n, u, y = self.z, self.o, self.t, self.p, self.w, self.b, self.n, self.u, self.y
        model, g, d, r, num, numr, nump, scp, spv, lin, lin_num, be, M, tau = self.model, self.g, self.d, self.r, self.num,\
            self.numr, self.nump, self.spc, self.spv, self.lin, self.lin_num, self.be, self.M, self.tau
        ison, nx, M,rho = self.ison, self.nx, self.M,self.rho

        for k in range(num):
            for i in range(numr):
                model.add_constraint(b[i, k] / 2 <= w[i, k])
                model.add_constraint(w[i, k] <= g[i][k] - b[i, k] / 2)
        # 速度上下限约束
        for k in range(num-1):
            model.add_constraint(d[k] / scp[1] * z[k] <= t[0, k])
            model.add_constraint(t[0, k] <= d[k] / scp[0] * z[k])
            model.add_constraint(d[k] / scp[1] * z[k] <= t[1, k])
            model.add_constraint(t[1, k] <= d[k] / scp[0] * z[k])
        #速度波动约束
        for k in range(num-2):
            model.add_constraint(d[k] / spv[0] * z[k] <=d[k] / d[k+1] * t[0, k + 1] - t[0, k])
            model.add_constraint(d[k]/d[k+1] * t[0, k + 1] - t[0, k] <= d[k] / spv[1] * z[k])
            model.add_constraint(d[k] / spv[0] * z[k] <=d[k]/d[k+1] * t[1, k + 1] - t[1, k])
            model.add_constraint(d[k]/d[k+1] * t[1, k + 1] - t[1, k] <= d[k] / spv[1] * z[k])
        # 子区内至少两个交叉口
        for k in range(num-1):
            model.add_constraint(p[k] + p[k + 1] <= 1)
        # 子区分割等待时间
        for k in range(num):
            for i in range(numr):
                model.add_constraints([nx * p[k] <= u[i, k], u[i, k] <= p[k]])
        # 
        for k in range(num):
            for i in range(numr):
                model.add_constraint(be * z[k] - M * (1 - y[i, k]) <= b[i, k])
                model.add_constraint(b[i, k] <= y[i, k])

        model.add_constraint(o[0]==0)
        for i in range(numr):
            if ison[i]==0:
                self.add_split_on_cons(i)
            elif ison[i]==1:
                self.add_split_in_cons(i)
        # 根据流量分配
        for k in range(num):
            model.add_constraint((1 - rho[k]) * model.sum([b[i, k]*(1-ison[i]) for i in range(numr)])>=\
                (1 - rho[k])* rho[k]* model.sum([b[i, k]*ison[i] for i in range(numr)]))
        # 相同子区内周期相等
        for k in range(num-1):
            model.add_constraint(-M * p[k + 1] <= z[k + 1] - z[k])
            model.add_constraint(z[k + 1] - z[k] <= M * p[k + 1])

    def add_split_on_cons(self, i):
        model, num, rf, M,tau,g= self.model, self.num, self.rf, self.M,self.tau,self.g
        o, w, t, u, y, p,n,b = self.o, self.w, self.t, self.u, self.y, self.p,self.n,self.b

        for k in range(num-1):
            model.add_constraint(o[k] + rf[i, k] + w[i, k] + t[0, k] + u[i, k+1] >=
                                 o[k + 1] + rf[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] - M * (1 - y[i, k+1]))
            model.add_constraint(o[k] + rf[i, k] + w[i, k] + t[0, k] + u[i, k+1] <=
                                 o[k + 1] + rf[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] + M * (1 - y[i, k+1]))

            model.add_constraint(-M * p[k+1] <= y[i, k + 1] - y[i, k])
            model.add_constraint(y[i, k + 1] - y[i, k] <= M * p[k+1])

            model.add_constraints([b[i, k]/2-M*p[k+1] <= w[i, k+1], w[i, k+1] <= g[i, k+1]-b[i, k]/2+M*p[k+1]])

    def add_split_in_cons(self,i):
        model, num, rf, M,tau,g= self.model, self.num, self.rf, self.M,self.tau,self.g
        o, w, t, u, y, p,n,b = self.o, self.w, self.t, self.u, self.y, self.p,self.n,self.b

        for k in range(num-1):
            model.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[1,k] >=
                                 o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k+1] - M * (1 - y[i, k+1]))
            model.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[1,k] <=
                                 o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k+1] + M * (1 - y[i, k+1]))


            model.add_constraint(-M * p[k+1] <= y[i, k + 1] - y[i, k])
            model.add_constraint(y[i, k + 1] - y[i, k] <= M * p[k+1])

            model.add_constraints([b[i, k+1]/2-M*p[k+1] <= w[i, k], w[i, k] <= g[i, k]-b[i, k+1]/2+M*p[k+1]])

    def _add_M1_bus_variables(self):
        model, num = self.model, self.num
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
        ub_list=[(i,k) for i in range(2) for k in range(num)]
        self.ub=model.continuous_var_dict(ub_list,lb=0,ub=1,name="ub")
    
    def _add_M1_bus_constaraints(self):
        z, o, tb, p, wb, bb, nb,dw = self.z, self.o, self.tb, self.p, self.wb, self.bb, self.nb,self.dw
        model, sg, d, srf, num, spcb, spvb, be, M, taub = self.model, self.sg, self.d, self.srf, self.num,\
             self.spcb, self.spvb,  self.be, self.M, self.taub
        ison, nx, M,rho,ex ,dwt,ub,dwm= self.ison, self.nx, self.M,self.rho,self.ex,self.dwt,self.ub,self.dwm

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

        for k in range(num):
            for i in range(2):
                model.add_constraint(bb[i,k]/2<=wb[i,k]) 
                model.add_constraint(wb[i,k]<=sg[i,k]-bb[i,k]/2)    

        for i in range(2):
            model.add_constraints([bb[i,k]>=be*z[k] for k in range(num)])

        for k in range(num-1):
            for i in range(2):
                model.add_constraint(dw[i,k]<=ex[k]*dwm[i,k]*z[k])

        for k in range(num):
            for i in range(2):
                model.add_constraints([nx * p[k] <= ub[i, k], ub[i, k] <= p[k]])

        for k in range(num-1):
            model.add_constraint(o[k]+srf[0,k]+wb[0,k]+tb[0,k]+ub[0,k+1]==o[k+1]+srf[0,k+1]+wb[0,k+1]+nb[0,k+1]+taub[0,k+1])
            model.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k]==o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k]+ub[1,k+1])

            model.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])
            model.add_constraints([bb[1,k+1]/2-M*p[k+1]<=wb[1,k],wb[1,k]<=sg[1,k]-bb[1,k+1]/2+M*p[k+1]])

    def _add_M1_obj(self):
        self.sum_b = self.model.sum([self.pv[i] * self.b[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_u = self.model.sum([self.pv[i] * self.u[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_p = self.model.sum([self.p[k] * (self.vol[0, k] + self.vol[1, k]) for k in range(self.num)])
        self.sum_bb = self.model.sum_vars(self.bb)*self.qb[0]
    
    def _M1_solve(self):
        self._add_M1_car_variables()
        self._add_M1_car_constraints()
        self._add_M1_bus_variables()
        self._add_M1_bus_constaraints()
        self._add_M1_obj()
        model,sum_b,sum_u,sum_p,sum_bb=self.model,self.sum_b,self.sum_u,self.sum_p,self.sum_bb
        model.set_multi_objective("max",[5*(sum_b+sum_bb)-4*(sum_u)-1*sum_p])
        self.sol = model.solve(log_output=True)
        print(self.sol.solve_details)
        print("object value:",self.sol.objective_value)

    def _get_M1_result(self):
        sol,z,t,p,y,dw,tb=self.sol,self.z,self.t,self.p,self.y,self.dw,self.tb
        z = sol.get_value_dict(z)
        t = sol.get_value_dict(t)
        p = sol.get_value_dict(p)
        y = sol.get_value_dict(y)
        dw=sol.get_value_dict(dw)
        tb=sol.get_value_dict(tb)
        print("M1",sol.get_value(self.sum_b),sol.get_value(self.sum_bb))
        
        return z,t,p,y,dw,tb
    
    def _add_M2_car_variables(self):
        mdl,num,lin_num,numr=self.md2,self.num,self.lin_num,self.numr
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
        mdl,num,numr,b,w,g,u,o=self.md2,self.num,self.numr,self.b2,self.w2,self.g,self.u2,self.o2
        nx,M,ison,rho,be=self.nx,self.M,self.ison,self.rho,self.be
        z,t,p,y,dw,tb=self._get_M1_result()
        
        for k in range(num):
            for i in range(numr):
                mdl.add_constraint(b[i, k] / 2<= w[i, k])
                mdl.add_constraint(w[i, k] <= g[i][k] - b[i, k] / 2)

        for k in range(num):
            for i in range(numr):
                mdl.add_constraints([nx * p[k] <= u[i, k], u[i, k] <= p[k]])

        for k in range(num):
            for i in range(numr):
                mdl.add_constraint(be * z[k] - M * (1 - y[i, k]) <= b[i, k])
                mdl.add_constraint(b[i, k] <= y[i, k])

        mdl.add_constraint(o[0]==0)
        for i in range(numr):
            if ison[i]==0:
                self._add_M2_split_on_cons(i)
            elif ison[i]==1:
                self._add_M2_split_in_cons(i)

        for k in range(num):
            mdl.add_constraint((1 - rho[k]) * mdl.sum([b[i, k]*(1-ison[i]) for i in range(numr)])>=\
                (1 - rho[k])* rho[k]* mdl.sum([b[i, k]*ison[i] for i in range(numr)]))

    def _add_M2_split_on_cons(self,i):
        mdl,num,o,w,u,n,b=self.md2,self.num,self.o2,self.w2,self.u2,self.n2,self.b2
        rf,M,tau,g=self.rf,self.M,self.tau,self.g
        z,t,p,y,dw,tb=self._get_M1_result()

        for k in range(num-1):
            mdl.add_constraint(o[k] + rf[i, k] + w[i, k] + t[0, k] + u[i, k+1] >=
                                o[k + 1] + rf[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] - M * (1 - y[i, k+1]))
            mdl.add_constraint(o[k] + rf[i, k] + w[i, k] + t[0, k] + u[i, k+1] <=
                                o[k + 1] + rf[i, k+1] + w[i, k + 1] + n[i, k + 1]+tau[0,k+1] + M * (1 - y[i, k+1]))

            mdl.add_constraints([b[i,k]/2-M*p[k+1]<=w[i,k+1],w[i,k+1]<=g[i,k+1]-b[i,k]/2+M*p[k+1]])

    def _add_M2_split_in_cons(self,i):
        mdl,num,o,w,u,n,b=self.md2,self.num,self.o2,self.w2,self.u2,self.n2,self.b2
        rf,M,g,tau=self.rf,self.M,self.g,self.tau
        z,t,p,y,dw,tb=self._get_M1_result()

        for k in range(num-1):
            mdl.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[1,k] >=
                                o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k+1] - M * (1 - y[i, k+1]))
            mdl.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[1,k] <=
                                o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k+1] + M * (1 - y[i, k+1]))
            
            mdl.add_constraints([b[i,k+1]/2-M*p[k+1]<=w[i,k],w[i,k]<=g[i,k]-b[i,k+1]/2+M*p[k+1]])

    def _add_M2_bus_variables(self):
        mdl,num=self.md2,self.num
        wb_list = [(i, k) for i in range(2) for k in range(num)]
        self.wb2 = mdl.continuous_var_dict(wb_list, lb=0, ub=1, name="wb")

        bb_list = [(i, k) for i in range(2) for k in range(num)]
        self.bb2 = mdl.continuous_var_dict(bb_list, lb=0, ub=1, name="bb")

        nb_list = [(i, k) for i in range(2) for k in range(num)]
        self.nb2 = mdl.integer_var_dict(nb_list, lb=0, ub=10, name="nb")

        ub_list=[(i,k) for i in range(2) for k in range(num)]
        self.ub2=mdl.continuous_var_dict(ub_list,lb=0,ub=1,name="ub")

    def _add_M2_bus_constraints(self):
        mdl,num,bb,wb,sg,be,o,nb=self.md2,self.num,self.bb2,self.wb2,self.sg,self.be,self.o2,self.nb2
        z,t,p,y,dw,tb=self._get_M1_result()
        srf,M,taub,ub,nx=self.srf,self.M,self.taub,self.ub2,self.nx
        
        for k in range(num):
            for i in range(2):
                mdl.add_constraint(bb[i,k]/2<=wb[i,k]) 
                mdl.add_constraint(wb[i,k]<=sg[i,k]-bb[i,k]/2)      

        for i in range(2):
            mdl.add_constraints([bb[i,k]>=be*z[k] for k in range(num)])
        
        for k in range(num):
            for i in range(2):
                mdl.add_constraints([nx * p[k] <= ub[i, k], ub[i, k] <= p[k]])

        for k in range(num-1):
            mdl.add_constraint(o[k]+srf[0,k]+wb[0,k]+tb[0,k]+ub[0,k+1]==o[k+1]+srf[0,k+1]+wb[0,k+1]+nb[0,k+1]+taub[0,k+1])
            mdl.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])

            mdl.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k]==o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k]+ub[1,k+1])
                
            mdl.add_constraints([bb[1,k+1]/2-M*p[k+1]<=wb[1,k],wb[1,k]<=sg[1,k]-bb[1,k+1]/2+M*p[k+1]])

    def _add_M2_spd_constraints(self):
        linspace,pc,C,d,nt,yp,tau,vol,lin_num=self.lin,self.pc,self.C,self.d,self.nt,self.yp,self.tau,\
        self.vol,self.lin_num
        srf,sg,M,taub,num=self.srf,self.sg,self.M,self.taub,self.num
        mdl,o=self.md2,self.o2
        z,t,p,y,dw,tb=self._get_M1_result()

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

        self.sum_on=0
        self.sum_in=0
        onbound_x=[]
        inbound_x=[]
        for i,v in enumerate(linspace):
            A_on_0=o[0]+srf[0,0]
            B_on_0=o[0]+srf[0,0]+sg[0,0]
            mdl.add_if_then(pc[0,0,i]==1,C[0,0,i]==B_on_0-A_on_0)
            mdl.add_if_then(pc[0,0,i]==0,C[0,0,i]==0)
            onb=self._add_var_on_cons(
                A=A_on_0,
                B=B_on_0,
                o=o,
                r=srf[0],
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
                onbound=[[o[0]+srf[0,0],o[0]+srf[0,0]+sg[0,0],o[0]+sg[0,0]]],
                z=z
            )
            self.sum_on+=mdl.sum([C[0,k,i]*vol[0,k]*props[0][self.pcum[k]-1][i] for k in range(num)])
            onbound_x.append(onb)
            
            A_in_0=o[num-1]+srf[1,num-1]
            B_in_0=o[num-1]+srf[1,num-1]+sg[1,num-1]
            mdl.add_if_then(pc[1,num-1,i]==1,C[1,num-1,i]==B_in_0-A_in_0)
            mdl.add_if_then(pc[1,num-1,i]==0,C[1,num-1,i]==0)
            mdl.add_constraints([C[1,num-1,i]<=(B_in_0-A_in_0)+M*(1-pc[1,num-1,i]),C[1,num-1,i]>=(B_in_0-A_in_0)-M*(1-pc[1,num-1,i]),
            C[1,num-1,i]>=-M*pc[1,num-1,i],C[1,num-1,i]<=M*pc[1,num-1,i]])
            inb=self._add_var_in_cons(
                A=A_in_0,
                B=B_in_0,
                o=o,
                r=srf[1],
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
    

    def _add_var_on_cons(self,A, B, o, r, g, t, n, k, end, yp, px, p,pc,C,onbound,z):
        if k>=end:
            self.md2.add_constraint(pc[k-1]==yp[k-1])
            return onbound
        else:
            if p[k]==1:
                A1=o[k]+r[k]+n[k]
                B1=o[k]+r[k]+n[k]+g[k]
            else:
                A1=self.md2.max(A+t[k-1]-px[k-1],o[k]+r[k]+n[k])
                B1=self.md2.min(B+t[k-1], o[k]+r[k]+n[k]+g[k])

            self.md2.add_if_then(pc[k]==1,C[k]==B1-A1)
            self.md2.add_if_then(pc[k]==0,C[k]==0)

            self.md2.add_constraints([self.be*z[k]-self.M*(1-yp[k]) <= B1-A1, B1-A1 <= g[k]+self.M*(1-yp[k])])
            self.md2.add_constraints([p[k]>=pc[k-1],yp[k-1]>=pc[k-1],pc[k-1]>=p[k]+yp[k-1]-1])
            onbound.append([A1,B1,B1-A1])
            return self._add_var_on_cons(A1, B1, o, r, g, t, n, k+1, end, yp, px, p,pc,C,onbound,z)

    def _add_var_in_cons(self,A, B, o, r, g, t, n, k, end, yp, px, p,pc,C,inbound,z):
        if k<=end:
            self.md2.add_constraints([p[k+1]>=pc[k+1],yp[k+1]>=pc[k+1],p[k+1]+yp[k+1]-1<=pc[k+1]])
            return inbound
        else:
            A1=self.md2.max(A+t[k]-px[k],o[k]+r[k]+n[k])
            B1=self.md2.min(B+t[k], o[k]+r[k]+n[k]+g[k])

            self.md2.add_constraints([self.be*z[k]-self.M*(1-yp[k]) <= B1-A1, B1-A1 <= g[k]+self.M*(1-yp[k])])
            self.md2.add_constraints([p[k+1]>=pc[k+1],yp[k+1]>=pc[k+1],p[k+1]+yp[k+1]-1<=pc[k+1]])

            self.md2.add_if_then(pc[k]==1,C[k]==B1-A1)
            self.md2.add_if_then(pc[k]==0,C[k]==0)
            inbound.append([A1,B1,B1-A1])
            if p[k]==1:
                A1=o[k]+r[k]+n[k]
                B1=o[k]+r[k]+n[k]+g[k]
            return self._add_var_in_cons(A1, B1, o, r, g, t, n, k-1, end, yp, px, p,pc,C,inbound,z)
    

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
        return self.getprop(self.lin-0.25,self.lin+0.25,mu,sigma).round(4)

    def _add_M2_obj(self):
        z,t,p,y,dw,tb=self._get_M1_result()
        mdl=self.md2
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
        self._add_M2_spd_constraints()
        self._add_M2_obj()

        mdl,sum_b,sum_u,sum_bb,sum_v,sum_p=self.md2,self.sum_b2,self.sum_u2,self.sum_bb2,self.sum_v,self.sum_p2
    
        refiner=ConflictRefiner()
        res=refiner.refine_conflict(mdl)
        res.display()

        mdl.set_multi_objective("max",[5*(sum_b+sum_bb)-4*(sum_u)-2*sum_p,sum_v],priorities=[2,1],weights=[1,1])
        # mdl.set_multi_objective("max",[sum_b+sum_bb,sum_u],weights=[5,-4])
        self.solution = mdl.solve(log_output=True)
        print(self.solution.solve_details)
        print("object value",self.solution.objective_value)
    def set_M2_time_limit(self,time):
        mdl=self.md2
        mdl.set_time_limit(time)
    def solve(self):
        self._M1_solve()
        self._M2_solve()

    def _get_M2_result(self):
        sol=self.solution
        b,o,u,n,yp,pc,nt,C,bb,wb,nb,w=self.b2,self.o2,self.u2,self.n2,self.yp,self.pc,self.nt,self.C,\
        self.bb2,self.wb2,self.nb2,self.w2
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
        # print("M2",sol.get_value(self.sum_bb),sol.get_value(self.sum_bb2))
        return b,o,u,n,yp,pc,nt,C,bb,wb,nb,w
    def get_dataframe(self):
        num,numr,d,dwt=self.num,self.numr,self.d,self.dwt
        b,o,u,n,yp,pc,nt,C,bb,wb,nb,w=self._get_M2_result()
        z,t,p,y,dw,tb=self._get_M1_result()
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
        # cols+=["yp_on_"+str(i) for i in range(1,lin_num+1)]
        # cols+=["yp_in_"+str(i) for i in range(1,lin_num+1)]

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
        return Df

    def get_draw_dataframe(self):
        Df=self.get_dataframe()
        num,d,rf,srf,numr=self.num,self.d,self.rf,self.srf,self.numr
        b,o,u,n,yp,pc,nt,C,bb,wb,nb,w=self._get_M2_result()
        Df2 = Df.copy()
        for i in range(numr):
            Df2["w"+str(i+1)]=[w[i, k] for k in range(num)]
            Df2["u"+str(i+1)]=np.array([u[i, k] for k in range(num)]) * Df.z
            Df2["car_t"+str(i+1)]=Df2.offset + rf[i] * Df2.z  +Df2.loc[:,"w"+str(i+1)] * Df2.z - Df2.loc[:,"b"+str(i+1)]/ 2 

        Df2["wb1"]=[wb[0, k] for k in range(num)]
        Df2["wb2"]=[wb[1, k] for k in range(num)]
        Df2["bus_t1"] =Df2.offset + srf[0] * Df2.z + Df2.wb1 * Df2.z - Df2.bb1 / 2
        Df2["bus_t2"] =Df2.offset + srf[1] * Df2.z + Df2.wb2 * Df2.z - Df2.bb2 / 2
        Df2['on_bus_v1']=[d[i]/(Df.tb1[i]-Df.dw1[i]) for i in range(num-1)]+[np.nan]
        Df2['in_bus_v1']=[d[i]/(Df.tb2[i]-Df.dw2[i]) for i in range(num-1)]+[np.nan]
        return Df2
    def data_formater(self,data, last_data, cross_num,z,g,t):
        while data < last_data+t[cross_num]-g[cross_num]/2*z[cross_num]:
            data += z[cross_num]
        return data

    def onbound(self,b, car_t, t, dis, distance):
        zip_x = [car_t,car_t + b,car_t + t + b,car_t + t,]
        zip_y = [dis, dis, dis + distance, dis + distance]
        return zip_x, zip_y

    def inbound(self,b, car_t, t, dis, distance):
        zip_x = [  car_t + t, car_t + t + b, car_t + b, car_t]
        zip_y = [dis - distance, dis - distance, dis, dis]
        return zip_x, zip_y
    
    def draw_car_bound(self,filepath,colors,legends,idx,linestyles):
        Df2=self.get_draw_dataframe()
        phase,numr,num,pgt,ison,nump,g=self.phase,self.numr,self.num,self.pg,self.ison,self.nump,self.g
        green_time=np.array([phase[:, j] * Df2.z for j in range(nump)])

        fig1 = plt.figure(figsize=(16, 16), dpi=300)
        ax1 = fig1.add_subplot()
        legends=[0 for i in range(nump)]
        color = colors
        on_count=0
        in_count=0
        for i in range(1,numr+1):
            tmpstr="car_t"+str(i)
            if ison[i-1]==0:
                Df2.loc[:,tmpstr]+=Df2.z*on_count
                on_count+=1
                for j in range(1, num):
                    Df2.loc[j,tmpstr]=self.data_formater(Df2.loc[j,tmpstr], Df2.loc[j-1,tmpstr], j,Df2.z,g[i-1],Df2.t1)
            else:
                Df2.loc[:,tmpstr]+=Df2.z*in_count
                in_count+=1
                for j in range(num - 1, 0, -1):
                    Df2.loc[j-1,tmpstr] =self.data_formater(Df2.loc[j-1,tmpstr], Df2.loc[j,tmpstr], j - 1,Df2.z,g[i-1],Df2.t2)
        
        max_width =max(Df2[["car_t"+str(i) for i in range(1,numr+1)]].max())
        max_hight = sum(Df2.distance[0 : num - 1]) + 100
        for i in range(0, num):
            offset_r = Df2.offset[i] - Df2.z[i]
            sum_dis = sum(Df2.distance[0:i])
            while offset_r < max_width:
                for j in range(nump):
                    if green_time[j,i] == 0:
                        continue
                    else:
                        legend=ax1.add_patch(
                           plt.Rectangle(
                                (offset_r, sum(Df2.distance[0:i])),
                                green_time[j, i],
                                20,
                                facecolor=color[j]["color"],
                                hatch=color[j]["hatch"],
                                fill=color[j]["fill"],
                                edgecolor='black',
                                linewidth=0.5
                            )
                        )
                        if legends[j]==0:
                            legends[j]=legend
                        offset_r += green_time[j,i]
            # ax1.text(10, sum(Df2.distance[0:i]) + 25, "S"+str(i + 1), fontsize=16)
        # plt.plot([0, 0], [0, max_hight])

        for idx in range(1,numr+1):
            if ison[idx-1]==0:
                for i in range(0, num):
                    dis = sum(Df2.distance[0:i])
                    if Df2.loc[i,"b"+str(idx)]== 0:
                        continue
                    else:
                        bstr,carstr,tstr="b"+str(idx),"car_t"+str(idx),"t1"
                        zip_x, zip_y = self.onbound(Df2.loc[i,bstr], Df2.loc[i,carstr], Df2.loc[i,tstr], dis, Df2.distance[i])
                        onbound1 = ax1.add_patch(pch.Polygon(xy=list(zip(zip_x, zip_y)), fill=False,linewidth=1,linestyle=linestyles[idx-1]["linestyle"]))
            else:
                for i in range(1, num):
                    dis = sum(Df2.distance[0:i])
                    if Df2.loc[i,"b"+str(idx)] == 0:
                        continue
                    else:
                        bstr,carstr,tstr="b"+str(idx),"car_t"+str(idx),"t2"
                        zip_x, zip_y = self.inbound(Df2.loc[i,bstr], Df2.loc[i,carstr], Df2.loc[i-1,tstr], dis, Df2.distance[i - 1])
                        inbound2 = ax1.add_patch(pch.Polygon(xy=list(zip(zip_x, zip_y)),fill=False,linewidth=1,linestyle=linestyles[idx-1]["linestyle"]))

        plt.xlim([0,max_width])
        plt.ylim(0, sum(Df2.distance[0 : num - 1]) + 100)
        xticks=np.arange(0,max_width,Df2.loc[0,"z"])
        yticks=[0]+Df2["distance"].cumsum().tolist()
        plt.xticks(xticks,[i for i in range(len(xticks))],fontsize=20)
        plt.yticks(yticks,["S"+str(i + 1) for i in range(len(yticks))],fontsize=20)

        ax1.legend(
            handles=legends,
            labels=[" "*10 for i in range(nump)],
            fontsize=20,
            loc="center right",
        )
        fig1.savefig(filepath, bbox_inches="tight")

    def draw_bus_bound(self,filepath,colors):
        Df2=self.get_draw_dataframe()
        ex,num,nump,sgt,phase,sg=self.ex,self.num,self.nump,self.sgt,self.phase,self.sg
        font1 = {'family': 'SimSun', 'size': 18, 'weight': 'normal'}
        bus_stop=[ex[i]*(sum(Df2.distance[0:i])+Df2.distance[i]*0.5) for i in range(0,num-1)]
        green_time=np.array([phase[:, j] * Df2.z for j in range(nump)])
        fig1 = plt.figure(figsize=(16,16), dpi=300)
        ax1 = fig1.add_subplot()
        legends=[0 for i in range(nump)]

        Df2.bus_t1 += Df2.z
        for i in range(1, num):
            Df2.loc[i, "bus_t1"] = self.data_formater(Df2.bus_t1[i], Df2.bus_t1[i - 1], i,Df2.z,sg[0],Df2.tb1)

        for i in range(num - 1, 0, -1):
            Df2.loc[i - 1, "bus_t2"] = self.data_formater(Df2.bus_t2[i - 1], Df2.bus_t2[i], i - 1,Df2.z,sg[1],Df2.tb2)

        max_width =max(Df2[["bus_t1", "bus_t2"]].max())+Df2.z.max()

        max_hight = sum(Df2.distance[0 : num - 1]) + 100
        print(max_width,max_width/Df2.z[0])
        legendc = dict()
        for i in range(0, num):
            offset_r = Df2.offset[i] - Df2.z[i]
            sum_dis = sum(Df2.distance[0:i])
            while offset_r < max_width:
                for j in range(nump):
                    if green_time[j,i] == 0:
                        continue
                    else:
                        legend=ax1.add_patch(
                            plt.Rectangle(
                                (offset_r, sum(Df2.distance[0:i])),
                                green_time[j, i],
                                20,
                                facecolor=colors[j]["color"],
                                hatch=colors[j]["hatch"],
                                fill=colors[j]["fill"],
                                edgecolor='black',
                                linewidth=0.5
                            )
                        )
                        if legends[j]==0:
                            legends[j]=legend
                        offset_r += green_time[j,i]
                        legends.append(legend)

        for i in range(0,num-1):
            dis=sum(Df2.distance[0:i])
            if Df2.bb1[i]==0:
                continue
            else:
                if bus_stop[i]>0:
                    bus_dis=(bus_stop[i]-dis)
                    zip_x1,zip_y1=self.onbound(Df2.bb1[i],Df2.bus_t1[i],bus_dis/Df2.on_bus_v1[i],dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x1,zip_y1)),fill=False,linewidth=1))

                    zip_x2,zip_y2=self.onbound(Df2.bb1[i],Df2.bus_t1[i]+bus_dis/Df2.on_bus_v1[i]+Df2.dw1[i],
                    bus_dis/Df2.on_bus_v1[i],dis+bus_dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x2,zip_y2)),fill=False,linewidth=1))
                    plt.plot([zip_x1[3],zip_x2[1]],[zip_y1[3],zip_y2[1]],color='coral',linewidth=3)
                else:
                    zip_x,zip_y=self.onbound(Df2.bb1[i],Df2.bus_t1[i],Df2.tb1[i],dis,Df2.distance[i])
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))

        for i in range(1,num):
            dis=sum(Df2.distance[0:i])
            if Df2.bb2[i]==0:
                continue
            else:
                if bus_stop[i-1]>0:
                    bus_dis=dis-bus_stop[i-1]
                    zip_x1,zip_y1=self.inbound(Df2.bb2[i],Df2.bus_t2[i],bus_dis/Df2.in_bus_v1[i-1],dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x1,zip_y1)),fill=False,linewidth=1))

                    zip_x2,zip_y2=self.inbound(Df2.bb2[i],Df2.bus_t2[i]+bus_dis/Df2.in_bus_v1[i-1]+Df2.dw2[i-1],
                    bus_dis/Df2.in_bus_v1[i-1],dis-bus_dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x2,zip_y2)),fill=False,linewidth=1))
                    legendx,=plt.plot([zip_x1[0],zip_x2[2]],[zip_y1[0],zip_y2[2]],color='coral',linewidth=3)
                else:
                    zip_x,zip_y=self.inbound(Df2.bb2[i],Df2.bus_t2[i],Df2.tb2[i-1],dis,Df2.distance[i-1])
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))


        plt.xlim([0,max_width,])
        plt.ylim(0, sum(Df2.distance[0 : num - 1]) + 100)
        xticks=np.arange(0,max_width,Df2.loc[0,"z"])
        yticks=[0]+Df2["distance"].cumsum().tolist()
        plt.xticks(xticks,[i for i in range(len(xticks))],fontsize=font1["size"])
        plt.yticks(yticks,["S"+str(i + 1) for i in range(len(yticks))],fontsize=font1["size"])
        legends.append(legendx)
        ax1.legend(
            handles=legends,
            labels=[" "*10 for i in range(nump)]+["bus station"],
            fontsize=20,
            loc="center right",
        )
        fig1.savefig(filepath, bbox_inches="tight"
        )