import docplex.mp.model as md
from docplex.mp.conflict_refiner import ConflictRefiner
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.patches as pch
import matplotlib.pyplot as plt

class MPbandTool():
    def __init__(self,phase, cycle, vol, pv, pg, d, sgt, ison, tau, taub, qb, low, up, be, lowv, upv,
                 ex, dwt,lowb,upb,lowbv,upbv,p,o,z,t) -> None:
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
        self.ex = ex  # 公交车站
        self.dwt = dwt  # 平均停靠时间
        self.p=p
        self.o=o
        self.z=z
        self.t=t

        self.rho = self.vol[0]/self.vol[1]
        self.num = len(self.vol[0])
        self.numr = len(self.pv)
        self.nump = len(self.phase[0])

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
        
    def get_rf(self, d, p):
        tmp = []
        for i, a in enumerate(p):
            idx = np.where(a != 0)[0]
            a = a & 0
            if len(idx) > 0:
                a[:idx[0]] = 1
            tmp.append((d[i]*a).sum())
        return tmp
    def _add_variables(self):
        model, num, numr, cycle = self.model, self.num, self.numr, self.cycle
        #   公共变量
        # t_list = [(i, k) for i in range(2) for k in range(num-1)]
        # self.t = model.continuous_var_dict(t_list, lb=0, name="t")

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
    
    def _add_car_constraints(self):
        z, o, t, p, w, b, n, u, y = self.z, self.o, self.t, self.p, self.w, self.b, self.n, self.u, self.y
        model, g, d, r, num, numr, nump, scp, spv, be, M, tau = self.model, self.g, self.d, self.r, self.num,\
            self.numr, self.nump, self.spc, self.spv, self.be, self.M, self.tau
        ison, nx, M,rho = self.ison, self.nx, self.M,self.rho

        for k in range(num):
            for i in range(numr):
                model.add_constraint(b[i, k] / 2 <= w[i, k])
                model.add_constraint(w[i, k] <= g[i][k] - b[i, k] / 2)

        for k in range(num):
            for i in range(numr):
                model.add_constraints([nx * p[k] <= u[i, k], u[i, k] <= p[k]])
        # 
        for k in range(num):
            for i in range(numr):
                model.add_constraint(be * z[k] - M * (1 - y[i, k]) <= b[i, k])
                model.add_constraint(b[i, k] <= y[i, k])

        for i in range(numr):
            if ison[i]==0:
                self.add_split_on_cons(i)
            elif ison[i]==1:
                self.add_split_in_cons(i)
        # 根据流量分配
        for k in range(num):
            model.add_constraint((1 - rho[k]) * model.sum([b[i, k]*(1-ison[i]) for i in range(numr)])>=\
                (1 - rho[k])* rho[k]* model.sum([b[i, k]*ison[i] for i in range(numr)]))

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

    def _add_bus_variables(self):
        model, num = self.model, self.num
        wb_list = [(i, k) for i in range(2) for k in range(num)]
        self.wb = model.continuous_var_dict(wb_list, lb=0, ub=1, name="wb")
        bb_list = [(i, k) for i in range(2) for k in range(num)]
        self.bb = model.continuous_var_dict(bb_list, lb=0, ub=1, name="bb")
        nb_list = [(i, k) for i in range(2) for k in range(num)]
        self.nb = model.integer_var_dict(nb_list, lb=0, ub=10, name="nb")
        tb_list=[(i, k) for i in range(2) for k in range(num-1)]
        self.tb=model.continuous_var_dict(tb_list,lb=0,name="tb")
        ub_list=[(i,k) for i in range(2) for k in range(num)]
        self.ub=model.continuous_var_dict(ub_list,lb=0,ub=1,name="ub")
        yb_list=[(i,k) for i in range(2) for k in range(num)]
        self.yb=model.binary_var_dict(yb_list,name="yb")
    
    def _add_bus_constaraints(self):
        z, o, tb, p, wb, bb, nb= self.z, self.o, self.tb, self.p, self.wb, self.bb, self.nb
        model, sg, d, srf, num, spcb, spvb, be, M, taub = self.model, self.sg, self.d, self.srf, self.num,\
             self.spcb, self.spvb,  self.be, self.M, self.taub
        ison, nx, M,rho,ex ,dwt,ub,yb= self.ison, self.nx, self.M,self.rho,self.ex,self.dwt,self.ub,self.yb

        for k in range(num-1):
            model.add_constraint(d[k] / spcb[1] * z[k] <= tb[0, k]-ex[k]*dwt*z[k])
            model.add_constraint(tb[0, k]-ex[k]*dwt*z[k] <= d[k] / spcb[0] * z[k])

            model.add_constraint(d[k] / spcb[1] * z[k] <= tb[1, k]-ex[k]*dwt*z[k])
            model.add_constraint(tb[1, k]-ex[k]*dwt*z[k]<= d[k] / spcb[0] * z[k])

        for k in range(num-2):
            model.add_constraint(d[k] / spvb[0] * z[k] <= 
            d[k]/d[k+1]*(tb[0, k + 1]-ex[k+1]*dwt*z[k])- (tb[0, k]-ex[k]*dwt*z[k]))
            model.add_constraint(d[k]/d[k+1]*(tb[0, k + 1]-ex[k+1]*dwt*z[k])- (tb[0, k]-ex[k]*dwt*z[k]) <=
            d[k] / spvb[1] * z[k])

            model.add_constraint(d[k] / spvb[0] * z[k] <=
            d[k]/d[k+1]*(tb[1, k + 1]-ex[k+1]*dwt*z[k])- (tb[1, k]-ex[k]*dwt*z[k]))
            model.add_constraint(d[k]/d[k+1]*(tb[1, k + 1]-ex[k+1]*dwt*z[k])-(tb[1, k]-ex[k]*dwt*z[k])<=
            d[k] / spvb[1] * z[k])

        for k in range(num):
            for i in range(2):
                model.add_constraint(bb[i,k]/2<=wb[i,k]) 
                model.add_constraint(wb[i,k]<=sg[i,k]-bb[i,k]/2)    

        for i in range(2):
            model.add_constraints([bb[i,k]>=be*z[k]-M*(1-yb[i,k]) for k in range(num)])


        for k in range(num):
            for i in range(2):
                model.add_constraints([nx * p[k] <= ub[i, k], ub[i, k] <= p[k]])
        for i in range(2):
            for k in range(num-1):
                model.add_constraint(-M * p[k+1] <= yb[i, k + 1] - yb[i, k])
                model.add_constraint(yb[i, k + 1] - yb[i, k] <= M * p[k+1])

        for k in range(num-1):
            model.add_constraint(o[k]+srf[0,k]+wb[0,k]+tb[0,k]+ub[0,k+1]>=
                                 o[k+1]+srf[0,k+1]+wb[0,k+1]+nb[0,k+1]+taub[0,k+1]-M*(1-yb[0,k+1]))
            model.add_constraint(o[k]+srf[0,k]+wb[0,k]+tb[0,k]+ub[0,k+1]<=
                                 o[k+1]+srf[0,k+1]+wb[0,k+1]+nb[0,k+1]+taub[0,k+1]+M*(1-yb[0,k+1]))
            model.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k]>=
                                 o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k]+ub[1,k+1]-M*(1-yb[1,k+1]))
            model.add_constraint(o[k] + srf[1, k] + wb[1, k] + nb[1, k]+taub[1,k]<=
                                 o[k + 1] + srf[1, k+1] + wb[1, k + 1] + tb[1, k]+ub[1,k+1]+M*(1-yb[1,k+1]))

            model.add_constraints([bb[0,k]/2-M*p[k+1]<=wb[0,k+1],wb[0,k+1]<=sg[0,k+1]-bb[0,k]/2+M*p[k+1]])
            model.add_constraints([bb[1,k+1]/2-M*p[k+1]<=wb[1,k],wb[1,k]<=sg[1,k]-bb[1,k+1]/2+M*p[k+1]])

    def _add_obj(self):
        self.sum_b = self.model.sum([self.pv[i] * self.b[i, k] for i in range(self.numr) for k in range(self.num)])
        self.sum_bb = self.model.sum_vars(self.bb)*self.qb[0]
    
    def _solve(self):
        self._add_variables()
        self._add_car_constraints()
        self._add_bus_variables()
        self._add_bus_constaraints()
        self._add_obj()
        model,sum_b,sum_bb=self.model,self.sum_b,self.sum_bb
        model.set_multi_objective("max",[sum_b+sum_bb],weights=[1])

        refiner=ConflictRefiner()
        res=refiner.refine_conflict(model)
        res.display()

        self.sol = model.solve(log_output=True)
        print(self.sol.solve_details)
        print("object value:",self.sol.objective_value)


    def _get_result(self):
        sol,z,t,p,y,tb=self.sol,self.z,self.t,self.p,self.y,self.tb
        o,w,n,u,b,bb,wb,nb=self.o,self.w,self.n,self.u,self.b,self.bb,self.wb,self.nb
        # z = sol.get_value_dict(z)
        # t = sol.get_value_dict(t)
        y = sol.get_value_dict(y)
        # dw=sol.get_value_dict(dw)
        tb=sol.get_value_dict(tb)
        # o = sol.get_value_dict(o)
        w = sol.get_value_dict(w)
        n = sol.get_value_dict(n)
        u = sol.get_value_dict(u)
        b = sol.get_value_dict(b)
        bb=sol.get_value_dict(bb)
        wb=sol.get_value_dict(wb)
        nb=sol.get_value_dict(nb)
        return z,t,p,y,tb,o,w,n,u,b,bb,wb,nb
    
    def get_dataframe(self):
        num,numr,d,dwt=self.num,self.numr,self.d,self.dwt
        z,t,p,y,tb,o,w,n,u,b,bb,wb,nb=self._get_result()
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
        # Df+=[[(dw[i,k])/z[k]+dwt for k in range(num-1)]+[np.nan] for i in range(2)]
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
        # cols+=["dw"+str(i) for i in range(1,3)]
        cols+=["tb"+str(i) for i in range(1,3)]

        Df.columns=cols

        Df["offset"] = Df.offset * Df.z
        Df["t1"] = Df.t1 * Df.z
        Df["t2"] = Df.t2 * Df.z
        # for i in range(numr):
        #     Df["b"+str(i+1)]=Df.loc[:,"b"+str(i+1)]*Df.z
        # Df["bb1"]=Df.bb1*Df.z
        # Df["bb2"]=Df.bb2*Df.z

        return Df

    def get_draw_dataframe(self):
        Df=self.get_dataframe()
        num,d,rf,srf,numr=self.num,self.d,self.rf,self.srf,self.numr
        z,t,p,y,tb,o,w,n,u,b,bb,wb,nb=self._get_result()
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

        fig1 = plt.figure(figsize=(20, 20), dpi=300)
        ax1 = fig1.add_subplot()
        legends = []
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
        
        max_width =max(Df2[["car_t"+str(i) for i in range(1,numr+1)]].max())+Df2.z.max()
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
                        legends.append(legend)
                        offset_r += green_time[j,i]
            ax1.text(10, sum(Df2.distance[0:i]) + 25, "S"+str(i + 1), fontsize=16)
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

        plt.xlim([0,max_width,])
        plt.ylim(0, sum(Df2.distance[0 : num - 1]) + 100)
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
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
        fig1 = plt.figure(figsize=(20, 20), dpi=300)
        ax1 = fig1.add_subplot()
        legends=[]

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
                        offset_r += green_time[j,i]
                        legends.append(legend)
            ax1.text(10, sum(Df2.distance[0:i]) + 25, "S"+str(i + 1), fontsize=18)
            
        plt.plot([0, 0], [0, max_hight])
            #绘制公交车站
        for i in range(0,num-1):
            flag=bus_stop[i]
            if flag!=0: 
                plt.plot([-100,max_width],[bus_stop[i],bus_stop[i]],color=sns.color_palette('Greys',5)[3],linewidth=1)
                ax1.text(0,bus_stop[i],'公交车站',fontdict=font1)

        for i in range(0,num-1):
            dis=sum(Df2.distance[0:i])
            if Df2.bb1[i]==0:
                continue
            else:
                if bus_stop[i]>0:
                    bus_dis=(bus_stop[i]-dis)
                    zip_x,zip_y=self.onbound(Df2.bb1[i],Df2.bus_t1[i],bus_dis/Df2.on_bus_v1[i],dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))

                    zip_x,zip_y=self.onbound(Df2.bb1[i],Df2.bus_t1[i]+bus_dis/Df2.on_bus_v1[i]+Df2.dw1[i],
                    bus_dis/Df2.on_bus_v1[i],dis+bus_dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))
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
                    zip_x,zip_y=self.inbound(Df2.bb2[i],Df2.bus_t2[i],bus_dis/Df2.in_bus_v1[i-1],dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))

                    zip_x,zip_y=self.inbound(Df2.bb2[i],Df2.bus_t2[i]+bus_dis/Df2.in_bus_v1[i-1]+Df2.dw2[i-1],
                    bus_dis/Df2.in_bus_v1[i-1],dis-bus_dis,bus_dis)
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))
                else:
                    zip_x,zip_y=self.inbound(Df2.bb2[i],Df2.bus_t2[i],Df2.tb2[i-1],dis,Df2.distance[i-1])
                    ax1.add_patch(pch.Polygon(xy=list(zip(zip_x,zip_y)),fill=False,linewidth=1))


        plt.xlim([0,max_width,])
        plt.ylim(0, sum(Df2.distance[0 : num - 1]) + 100)
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.legend(
            handles=legends,
            labels=[" "*10 for i in range(nump)],
            fontsize=20,
            loc="center right",
        )
        fig1.savefig(filepath, bbox_inches="tight"
        )