import pandas as pd
import cvxpy as cp
import numpy as np
import docplex.mp.model as md
from docplex.mp.conflict_refiner import ConflictRefiner
import scipy.stats as stats
import seaborn as sns 
import matplotlib.patches as pch
import matplotlib.pyplot as plt


class LMband():
    def __init__(self, phase, cycle, vol, pv, pg, d, sgt, srlgt, ison, tau, taub, qb, low, up, linspace, be, lowv, upv,
                 ex, dwt,lowb,upb,lowbv,upbv,mu,sigma) -> None:
        self.d = d  # 交叉口间距
        self.phase = phase  # 相位时长
        self.cycle = cycle  # 信号周期
        self.vol = vol  # 直行流量
        self.pv = pv  # 路径流量
        self.pg = pg  # 路径通行权
        self.sgt = sgt  # 直行通行权
        self.srlgt = srlgt  # 直行绿前红灯通行权
        self.ison = ison  # 路径是否为上行方向
        self.tau = tau  # 左转偏移量
        self.taub = taub  # 公交左转偏移量
        self.qb = qb  # 公交流量
        self.ex = ex  # 公交车站
        self.dwt = dwt  # 平均停靠时间
        self.lin = linspace  # 速度求解空间
        self.mu=mu
        self.sigma=sigma

        self.rho = self.vol[0]/self.vol[1]
        self.num = len(self.vol[0])
        self.numr = len(self.pv)
        self.nump = len(self.pg)
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

        self.GetProporation()

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
            model.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[0,k] >=
                                 o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] - M * (1 - y[i, k+1]))
            model.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[0,k] <=
                                 o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] + M * (1 - y[i, k+1]))


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
        self.tb=model.continuous_var_dict(tb_list,lb=0,ub=1,name="tb")
    
    def _add_M1_bus_constaraints(self):
        z, o, tb, p, wb, bb, nb,dw = self.z, self.o, self.tb, self.p, self.wb, self.bb, self.nb,self.dw
        model, sg, d, srf, num, spcb, spvb, be, M, taub = self.model, self.sg, self.d, self.srf, self.num,\
             self.spcb, self.spvb,  self.be, self.M, self.taub
        ison, nx, M,rho,ex ,dwt= self.ison, self.nx, self.M,self.rho,self.ex,self.dwt

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
                model.add_constraints([dw[i,k]<=ex[k]*15*z[k]])

        for k in range(num-1):
            model.add_constraint(o[k] + srf[0, k] + wb[0, k] + tb[0, k] <=
                                    o[k + 1] + srf[0, k+1] + wb[0,k+1] + nb[0,k+1]+taub[0,k+1]+M*p[k+1])
            model.add_constraint(o[k] + srf[0, k] + wb[0, k] + tb[0, k]  >=
                                    o[k + 1] + srf[0, k+1] + wb[0, k + 1] + nb[0, k + 1]+taub[0,k+1]-M*p[k+1])

            model.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])

            model.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k] >=
                                    o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k] - M * p[k+1] )
            model.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k] <=
                                    o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k] + M * p[k+1])     
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
        model.set_multi_objective("max",[sum_b+sum_bb,sum_u,sum_p],weights=[5,-4,-2])
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
                                o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] - M * (1 - y[i, k+1]))
            mdl.add_constraint(o[k] + rf[i, k] + w[i, k] + n[i, k]+tau[1,k] <=
                                o[k + 1] + rf[i, k+1] + w[i, k + 1] + t[1, k] + u[i, k] + M * (1 - y[i, k+1]))
            
            mdl.add_constraints([b[i,k+1]/2-M*p[k+1]<=w[i,k],w[i,k]<=g[i,k]-b[i,k+1]/2+M*p[k+1]])

    def _add_M2_bus_variables(self):
        mdl,num=self.md2,self.num
        wb_list = [(i, k) for i in range(2) for k in range(num)]
        self.wb2 = mdl.continuous_var_dict(wb_list, lb=0, ub=1, name="wb")

        bb_list = [(i, k) for i in range(2) for k in range(num)]
        self.bb2 = mdl.continuous_var_dict(bb_list, lb=0, ub=1, name="bb")

        nb_list = [(i, k) for i in range(2) for k in range(num)]
        self.nb2 = mdl.integer_var_dict(nb_list, lb=0, ub=10, name="nb")

    def _add_M2_bus_constraints(self):
        mdl,num,bb,wb,sg,be,o,nb=self.md2,self.num,self.bb2,self.wb2,self.sg,self.be,self.o2,self.nb2
        z,t,p,y,dw,tb=self._get_M1_result()
        srf,M,taub=self.srf,self.M,self.taub
        
        for k in range(num):
            for i in range(2):
                mdl.add_constraint(bb[i,k]/2<=wb[i,k]) 
                mdl.add_constraint(wb[i,k]<=sg[i,k]-bb[i,k]/2)      

        for i in range(2):
            mdl.add_constraints([bb[i,k]>=be*z[k] for k in range(num)])

        for k in range(num-1):
            mdl.add_constraint(o[k] + srf[0, k] + wb[0, k] + tb[0, k] <=
                                    o[k + 1] + srf[0, k+1] + wb[0, k + 1]+taub[0,k+1]+nb[0,k+1]+M*p[k+1])
            mdl.add_constraint(o[k] + srf[0, k] + wb[0, k] + tb[0, k]  >=
                                    o[k + 1] + srf[0, k+1] + wb[0, k + 1]+taub[0,k+1]+nb[0, k+1]- M*p[k+1])

            mdl.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])

            mdl.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k] >=
                                    o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k] - M * p[k+1] )
            mdl.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k] <=
                                    o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k] + M * p[k+1])
                
            mdl.add_constraints([bb[1,k+1]/2-M*p[k+1]<=wb[1,k],wb[1,k]<=sg[1,k]-bb[1,k+1]/2+M*p[k+1]])

    def _add_M2_spd_constraints(self):
        linspace,pc,C,d,nt,yp,tau,vol,ProDistribution,lin_num=self.lin,self.pc,self.C,self.d,self.nt,self.yp,self.tau,\
        self.vol,self.ProDistribution,self.lin_num
        srf,sg,M,taub,num=self.srf,self.sg,self.M,self.taub,self.num
        mdl,o=self.md2,self.o2
        z,t,p,y,dw,tb=self._get_M1_result()

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
            self.sum_on+=mdl.sum([C[0,k,i]*vol[0,k] for k in range(num)])*ProDistribution[i]
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
            self.sum_in+=mdl.sum([C[1,k,i]*vol[1,k] for k in range(num)])*ProDistribution[i]
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
                A1=o[k]+r[k]+n[k]+px[k]
                B1=o[k]+r[k]+n[k]+g[k]
            return self._add_var_in_cons(A1, B1, o, r, g, t, n, k-1, end, yp, px, p,pc,C,inbound,z)
    
    
    def getprop(self,linspace1,linspace2,mu,sigma):
        t1=stats.norm(mu,sigma).cdf(linspace1)
        t2=stats.norm(mu,sigma).cdf(linspace2)
        return t2-t1

    def GetProporation(self):
        mu=self.mu
        sigma=np.sqrt(self.sigma)
        self.ProDistribution=self.getprop(self.lin-0.25,self.lin+0.25,mu,sigma)

    def _add_M2_obj(self):
        mdl=self.md2
        self.sum_b2 = mdl.sum([self.pv[i] * self.b2[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_u2 = mdl.sum([self.pv[i] * self.u2[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_bb2= mdl.sum_vars(self.bb2)*self.qb[0]
        self.sum_v = self.sum_in+self.sum_on
    
    def _M2_solve(self):
        self._add_M2_car_variables()
        self._add_M2_bus_variables()
        self._add_M2_car_constraints()
        self._add_M2_bus_constraints()
        self._add_M2_spd_constraints()
        self._add_M2_obj()

        mdl,sum_b,sum_u,sum_bb,sum_v=self.md2,self.sum_b2,self.sum_u2,self.sum_bb2,self.sum_v
    
        refiner=ConflictRefiner()
        res=refiner.refine_conflict(mdl)
        res.display()

        mdl.set_multi_objective("max",[sum_b+sum_bb,sum_u,sum_v],priorities=[2,2,1],weights=[5,-4,1])
        # mdl.set_multi_objective("max",[sum_b+sum_bb,sum_u],weights=[5,-4])
        solution = mdl.solve(log_output=True)
        print(solution.solve_details)
        print("object value",solution.objective_value)

    def solve(self):
        self._M1_solve()
        self._M2_solve()