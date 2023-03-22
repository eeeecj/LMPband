import os 
import sys
import pandas as pd
from Simulation import Simulation
import numpy as np


class SimAnalysis():
    def __init__(self,df,phase,ex,fileath1,layout,filepath2,fg=None) -> None:
        vissim=Simulation(fileath1,layout,filepath2)
        self.vissim=vissim
        vissim.set_dwelltime(np.array([df.dw1,df.dw2]))
        vissim.init_vissim()
        scs=vissim.scs
        for sc in scs:
            id=int(sc.ID)
            vissim.set_offset(sc,df.offset[id-1])
            vissim.set_cycle(sc,df.z[id-1])
            sgs=sc.SignalGroups
            g=phase[id-1]*df.z[id-1]
            if fg is not None:
                fgt=fg[id-1]
            else:
                fgt=None
            vissim.Set_SignalGroups(sgs,g,fgt)
        df["d"]=df.distance
        car_t=np.array([df.d/df.t1,df.d/df.t2])+3
        bus_t=np.array([df.d/(df.tb1-ex*df.dw1[:12]),df.d/(df.tb2-ex*df.dw2[:12])])
        vissim.set_desiredspeed(car_t,bus_t)
        vissim.set_vehInput(1)
        vissim.set_period(3600)
        vissim.set_randomseed(42)
    def set_saturation(self,idx):
        vissim=self.vissim
        vissim.set_vehInput(idx)
        
    def start(self):
        self.vissim.start()

    def analysis_speed(self,filepath):
        vissim=self.vissim
        tz= pd.read_table(filepath, encoding='gbk')
        index=np.where(tz[tz.columns[0]].str.startswith(' Link;'))[0]
        data=tz[index[0]:]
        data.columns = ['data']
        data = pd.DataFrame([var.split(';') for var in data['data']])
        data.columns = data.iloc[0, :].apply(lambda x :x.strip())
        data = data.drop([0], axis=0)
        data=data.drop('',axis=1)
        data=data.reset_index(drop=True)
        t=data[data.columns].astype(float)
        data[t.columns]=t

        links=vissim.vnet.Links
        linkName=dict()
        for link in links:
            linkName[link.ID]=link.Name
        data["LinkName"]=data["Link"].apply(lambda x:linkName[int(x)])
        data=data[-(data["LinkName"]=="")]

        data["speed"]=data["v"]//10*10
        dgp=data.groupby(["speed","LinkName"])["TQDelay"].aggregate(np.mean)
        return dgp
    def analysis_traveltime(self,filepath):
        traveltime = pd.read_table(filepath, encoding='gbk')
        traveltime = traveltime[9:].reset_index(drop=True)
        traveltime.columns = ['data']
        traveltime1 = pd.DataFrame([var.split(';') for var in traveltime.data])
        traveltime1 = traveltime1.drop(columns=[len(traveltime1.columns)-1], axis=1)[1:]
        traveltime1.columns = ["time"]+["traveltime", "vehicles"]*8
        traveltime1=traveltime1[3:].reset_index(drop=True)
        traveltime1.replace('\s+','',regex=True,inplace=True)
        traveltime1.columns=traveltime1.columns+traveltime1.loc[0,:]
        traveltime1=traveltime1[2:].reset_index(drop=True)
        return traveltime1
    def analysis_delay(self,filepath):
        delay = pd.read_table(filepath, encoding='gbk')
        delay = delay[9:].reset_index(drop=True)
        delay.columns = ['data']
        delay1 = pd.DataFrame([var.split(';') for var in delay.data])
        delay1 = delay1.drop(columns=[len(delay1.columns)-1], axis=1)[1:]
        delay1.columns = ["time"]+["delay", "stoped", "stops", "vehicle", "per delay", "pers"]*8
        delay1.replace('\s+','',regex=True,inplace=True)
        delay1.columns=delay1.columns+delay1.loc[4,:]
        delay1=delay1.drop(11,axis=0)
        delay1=delay1[4:].reset_index(drop=True)
        return delay1
    