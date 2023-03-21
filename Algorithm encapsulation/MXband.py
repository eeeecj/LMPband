import docplex.mp.model as md
from docplex.mp.conflict_refiner import ConflictRefiner
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.patches as pch
import matplotlib.pyplot as plt

class MXband():
    def __init__(self,phase,cycle,sgt,d,low,up,lowv, upv,vol,tau) -> None:
        self.model=md.Model("Multiband")
        self.phase=phase
        self.sgt=sgt
        self.cycle=cycle
        self.d=d
        self.low=low
        self.up=up
        self.lowv=lowv
        self.upv=upv
        self.vol=vol
        self.spc = np.array([low, up])
        self.spv = np.array([lowv, upv])
        self.tau=tau
        self.M=1e6
        self.nx=1e-6
        self.be=8

        self.rho = self.vol[0]/self.vol[1]
        self.num=self.num = len(self.vol[0])
        self.sg = np.array([(self.sgt[i]*phase).sum(axis=1) for i in range(len(self.sgt))])
        self.srf = np.array([self.get_rf(self.phase, self.sgt[i]) for i in range(len(self.sgt))])

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
        model,num,cycle=self.model,self.num,self.cycle
        Z_list = [(i) for i in range(num)]
        self.z = model.continuous_var_dict(
        Z_list, lb=1/cycle[1], ub=1/cycle[0], name="z")
        
        var_list_w=[(i,j) for i in range(2)  for j in range(num)]
        self.w=model.continuous_var_dict(var_list_w,lb=0,ub=1,name='w')

        var_list_b=[(i,j) for i in range(2) for j in range(num)]
        self.b=model.continuous_var_dict(var_list_b,lb=0,ub=1,name='b')

        var_list_n=[(i,j) for i in range(2) for j in range(num)]
        self.n=model.integer_var_dict(var_list_n,lb=0,ub=10,name='n')

        var_list_o=[(i) for i in range(num)]
        self.o=model.continuous_var_dict(var_list_o,lb=0,ub=1,name='o')

        var_list_t=[(i,j) for i in range(2) for j in range(num-1)]
        self.t=model.continuous_var_dict(var_list_t,lb=0,name='t')

        var_list_u=[(i,j) for i in range(2) for j in range(num)]
        self.u=model.continuous_var_dict(var_list_u,lb=0,ub=1,name='u')

        var_list_p=[(i) for i in range(num)]
        self.p=model.binary_var_dict(var_list_p,name="p")
    def _add_constraints(self):
        model,num,w,t,d,scp,spv,z=self.model,self.num,self.w,self.t,self.d,self.spc,self.spv,self.z
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

        nx,u,p,b,sg,be,o,srf,n,tau=self.nx,self.u,self.p,self.b,self.sg,self.be,self.o,self.srf,self.n,self.tau
        M=self.M

        for k in range(num-1):
            model.add_constraint(p[k] + p[k + 1] <= 1)

        for k in range(num):
            for i in range(2):
                model.add_constraints([nx * p[k] <= u[i, k], u[i, k] <= p[k]])


        for i in range(num-1):
            model.add_constraints([b[0,i]/2-M*p[i+1]<=w[0,i+1],w[0,i+1]<=sg[0,i+1]-b[0,i]/2+M*p[i+1]])
            model.add_constraints([b[1,i+1]/2-M*p[i+1]<=w[1,i],w[1,i]<=sg[1,i]-b[1,i+1]/2+M*p[i+1]])
            
        for k in range(2):
            for i in range(num):
                model.add_constraint(b[k,i]/2<=w[k,i])
                model.add_constraint(w[k,i]<=sg[k,i]-b[k,i]/2)
    
        # for i in range(num):
        #     model.add_constraints([b[j,i]>=be*z[i] for j in range(2)])
        
        model.add_constraint(o[0]==0)    
        for i in range(num-1):   
            model.add_constraint(o[i]+srf[0,i]+w[0,i]+t[0,i]+u[0,i+1]==o[i+1]+srf[0,i+1]+w[0,i+1]+n[0,i+1]+tau[0,i+1])           
            model.add_constraint(o[i]+srf[1,i]+w[1,i]+n[1,i]+tau[1,i]==o[i+1]+srf[1,i+1]+u[1,i+1]+w[1,i+1]+t[1,i])                  

        for k in range(num-1):
            model.add_constraint(-M * p[k + 1] <= z[k + 1] - z[k])
            model.add_constraint(z[k + 1] - z[k] <= M * p[k + 1])

    def _add_obj(self):
        model=self.model
        self.sum_b=model.sum([self.vol[i,k] * self.b[i, k] for i in range(2) for k in range(self.num)])
        self.sum_u = self.model.sum([self.vol[i,k] * self.u[i, k] for i in range(2) for k in range(self.num)])
        self.sum_p = self.model.sum([self.p[k] * (self.vol[0, k] + self.vol[1, k])/2 for k in range(self.num)])
        
    def _solve(self):
        self._add_variables()
        self._add_constraints()
        self._add_obj()

        refiner=ConflictRefiner()
        res=refiner.refine_conflict(self.model)
        res.display()

        model,sum_b,sum_u,sum_p=self.model,self.sum_b,self.sum_u,self.sum_p
        model.set_multi_objective("max",[sum_b,sum_u,sum_p],weights=[0.6,-0.35,-0.05])
        self.sol = model.solve(log_output=True)
        print(self.sol.solve_details)
        print("object value:",self.sol.objective_value)

    def _get_result(self):
        sol,o,w,n,t,b,u,z,p=self.sol,self.o,self.w,self.n,self.t,self.b,self.u,self.z,self.p
        o=sol.get_value_dict(o)
        w=sol.get_value_dict(w)
        n=sol.get_value_dict(n)
        t=sol.get_value_dict(t)
        b=sol.get_value_dict(b)
        u=sol.get_value_dict(u)
        z=sol.get_value_dict(z)
        p=sol.get_value_dict(p)
        return o,w,n,t,b,u,z,p
    
    def get_dataframe(self):
        num,d,sg,srf=self.num,self.d,self.sg,self.srf
        o,w,n,t,b,u,z,p=self._get_result()
        Df=[[i for i in range(1,num+1)]]
        Df+=[[d[i] for i in range(num-1)] + [np.nan]]
        Df+=[[sg[i,k] for k in range(num)] for i in range(2) ]
        Df+=[[b[i,k] for k in range(num)] for i in range(2)]
        Df+=[[o[k] for k in range(num)]]
        Df+=[[p[k] for k in range(num)]]
        Df+=[[t[i,k] for k in range(num-1)]+[np.nan] for i in range(2)]
        Df+=[[1/z[k] for k in range(num)]]
        Df+=[[u[i,k] for k in range(num)] for i in range(2)]
        Df=np.array(Df)
        Df=Df.T
        Df=pd.DataFrame(Df)
        cols=["cross_number"]
        cols+=["distance"]
        cols+=["sg"+str(i) for i in range(1,3)]
        cols+=["b"+str(i) for i in range(1,3)]
        cols+=["offset","p"]
        cols+=["t"+str(i) for i in range(1,3)]
        cols+=["z"]
        cols+=["u"+str(i) for i in range(1,3)]
        Df.columns=cols
        Df["offset"] = Df.offset * Df.z
        Df["t1"] = Df.t1 * Df.z
        Df["t2"] = Df.t2 * Df.z
        # for i in range(2):
        #     Df["b"+str(i+1)]=Df.loc[:,"b"+str(i+1)]*Df.z
        for i in range(2):
            Df["w"+str(i+1)]=[w[i, k] for k in range(num)]
            Df["car_t"+str(i+1)]=Df.offset + srf[i] * Df.z  +Df.loc[:,"w"+str(i+1)] * Df.z - Df.loc[:,"b"+str(i+1)]/ 2 
        return Df
    
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
        Df=self.get_dataframe()
        phase,num,sg=self.phase,self.num,self.sg
        nump=len(self.phase[0])
        green_time=np.array([phase[:, j] * Df.z for j in range(nump)])

        fig1 = plt.figure(figsize=(20, 20), dpi=300)
        ax1 = fig1.add_subplot()
        legends = []
        color = colors
        on_count=0
        in_count=0
        for i in range(1,3):
            tmpstr="car_t"+str(i)
            if i==1:
                for j in range(1, num):
                    Df.loc[j,tmpstr]=self.data_formater(Df.loc[j,tmpstr], Df.loc[j-1,tmpstr], j,Df.z,sg[i-1],Df.t1)
            else:
                for j in range(num - 1, 0, -1):
                    Df.loc[j-1,tmpstr] =self.data_formater(Df.loc[j-1,tmpstr], Df.loc[j,tmpstr], j - 1,Df.z,sg[i-1],Df.t2)
        
        max_width =max(Df[["car_t"+str(i) for i in range(1,3)]].max())+Df.z.max()
        max_hight = sum(Df.distance[0 : num - 1]) + 100
        for i in range(0, num):
            offset_r = Df.offset[i] - Df.z[i]
            sum_dis = sum(Df.distance[0:i])
            while offset_r < max_width:
                for j in range(nump):
                    if green_time[j,i] == 0:
                        continue
                    else:
                        if self.sgt[0,i,j]==1:
                            legend=ax1.add_patch(
                            plt.Rectangle(
                                    (offset_r, sum(Df.distance[0:i])),
                                    green_time[j, i],
                                    20,
                                    facecolor="g",
                                    edgecolor='black',
                                    linewidth=0.5
                                )
                            )
                            legends.append(legend)
                        # legend=ax1.add_patch(
                        #    plt.Rectangle(
                        #         (offset_r, sum(Df.distance[0:i])),
                        #         green_time[j, i],
                        #         20,
                        #         facecolor=color[j]["color"],
                        #         hatch=color[j]["hatch"],
                        #         fill=color[j]["fill"],
                        #         edgecolor='black',
                        #         linewidth=0.5
                        #     )
                        # )
                        # legends.append(legend)
                        offset_r += green_time[j,i]
            ax1.text(10, sum(Df.distance[0:i]) + 25, "S"+str(i + 1), fontsize=16)
            
        for idx in range(1,3):
            if idx==1:
                for i in range(0, num):
                    dis = sum(Df.distance[0:i])
                    if Df.loc[i,"b"+str(idx)]== 0:
                        continue
                    else:
                        bstr,carstr,tstr="b"+str(idx),"car_t"+str(idx),"t1"
                        zip_x, zip_y = self.onbound(Df.loc[i,bstr], Df.loc[i,carstr], Df.loc[i,tstr], dis, Df.distance[i])
                        onbound1 = ax1.add_patch(pch.Polygon(xy=list(zip(zip_x, zip_y)), fill=False,linewidth=1,linestyle=linestyles[idx-1]["linestyle"]))
            else:
                for i in range(1, num):
                    dis = sum(Df.distance[0:i])
                    if Df.loc[i,"b"+str(idx)] == 0:
                        continue
                    else:
                        bstr,carstr,tstr="b"+str(idx),"car_t"+str(idx),"t2"
                        zip_x, zip_y = self.inbound(Df.loc[i,bstr], Df.loc[i,carstr], Df.loc[i-1,tstr], dis, Df.distance[i - 1])
                        inbound2 = ax1.add_patch(pch.Polygon(xy=list(zip(zip_x, zip_y)),fill=False,linewidth=1,linestyle=linestyles[idx-1]["linestyle"]))

        plt.xlim([0,max_width,])
        plt.ylim(0, sum(Df.distance[0 : num - 1]) + 100)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.legend(
            handles=legends,
            labels=[" "*10 for i in range(nump)],
            fontsize=20,
            loc="center right",
        )
        fig1.savefig(filepath, bbox_inches="tight")