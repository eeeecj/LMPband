import win32com.client as com
import numpy as np
from copy import copy
import re

class Simulation():
    def __init__(self, filepath,layout,filepath2) -> None:
        self.filepath=filepath
        self.filepath2=filepath2
        self.layout=layout
        self.vissim=None

    def init_vissim(self):
        filepath2=self.filepath2
        layout=self.layout
        self.vissim=com.Dispatch("vissim.vissim.430")
        self.vissim.LoadNet(filepath2)
        self.vissim.LoadLayout(layout)
        self.sim=self.vissim.Simulation
        self.vnet=self.vissim.Net
        self.eva=self.vissim.Evaluation
        self.scs=self.vnet.SignalControllers
        self.vhs=self.vnet.VehicleInputs 
        self.spds=self.vnet.DesiredSpeedDecisions
        
    def _check_vissim(self):
        if self.vissim==None:
            raise ValueError("please init vissim first")
        
    def _add_eva_config(self):
        self._check_vissim()
        eva=self.eva
        eva.SetAttValue('TRAVELTIME', True)
        eva.SetAttValue('DELAY', True)
        # eval.TravelTimeEvaluation.SetAttValue('FILE', True)
        # eval.DelayEvaluation.SetAttValue('FILE', True)
        eva.SetAttValue('NODE',True)
        # eval.NodeEvaluation.SetAttValue('FILE',True)
        eva.SetAttValue('NETPERFORMANCE',True)
        eva.SetAttValue('VEHICLERECORD',True)

    def set_offset(self,sc,offset):
        sc.SetAttValue("offset",int(offset))

    def set_cycle(self,sc,cycle):
        sc.SetAttValue("CYCLETIME",int(cycle))
    
    def _set_signalgroup(self,sg,ar,am,rEnd,gEnd):
        sg.SetAttValue("REDAMBER",int(ar))
        sg.SetAttValue("AMBER",int(am))
        sg.SetAttValue("REDEND",int(rEnd))
        sg.SetAttValue("GreenEnd",int(gEnd))
    def _set_input(self,vhi,ipt):
        vhi.SetAttValue("VOLUME",int(ipt))

    def Set_SignalGroups(self,sgs,g,fg=None):
        if len(g)!=sgs.Count:
            raise ValueError("g 与 sc 长度不一致.:g is {},sgs is {}".format(len(g),sgs.Count))
        gcum=g.cumsum()
        gcum=np.insert(gcum,0,0)
        if fg is not None:
            gcum=fg
            print(gcum)
        for i in range(sgs.Count):
            sg=sgs.GetSignalGroupByNumber(i+1)
            if g[i]==0:
                self._set_signalgroup(sg,0,0,0.1,0.1)
            else:
                self._set_signalgroup(sg,1,3,gcum[i]+4,gcum[i]+g[i])
    
    def set_randomseed(self,seed):
        self._check_vissim()
        sim=self.sim
        sim.SetAttValue("RandomSeed",seed)
    
    def set_dwelltime(self,dw):
        filepath=self.filepath
        filepath2=self.filepath2
        with open(filepath,"r",encoding="utf-8") as f1,open(filepath2,"w+",encoding="utf-8") as f2:
            while True:
                line1=f1.readline()
                line2=copy(line1)
                if not line1:
                    break
                fsch=re.search(r'TIMES +(\d+) *MEAN +(\d+.\d) +STANDARD_DEVIATION +(\d+.\d)',line1)
                if fsch:
                    id=int(fsch.group(1))
                    dt=dw[0]
                    if id >12:
                        id=id-12
                        dt=dw[1]
                    line2.replace(fsch.group(2), ("%2.1f"%(dt[id-1])).rjust(4," "))
                    line2.replace(fsch.group(3), "%2.1f"%(0))
                f2.write(line2)
                
    def set_vehInput(self,startuation):
        self._check_vissim()
        vhs=self.vhs
        for vh in  vhs:
            vh.SetAttValue("VOLUME",int(int(vh.AttValue("VOLUME"))*startuation))

    def set_desiredspeed(self,car_v,bus_v):
        self._check_vissim()
        spds=self.spds
        for spd in spds:
            id=spd.ID
            tp=id//1000
            if tp==9:
                crs=(id%1000)//10
                if crs<=12:
                    speed=bus_v[0,crs-1]*3.6//5
                    spd.SetAttValue("DESIREDSPEED",speed)
                else:
                    speed=bus_v[1,crs-12-1]*3.6//5
                    spd.SetAttValue("DESIREDSPEED",speed)
            else:
                crs=id//100
                if crs<=12:
                    speed=car_v[0,crs-1]*3.6//5
                    spd.SetAttValue("DESIREDSPEED",speed)
                else:
                    speed=car_v[1,crs-12-1]*3.6//5
                    spd.SetAttValue("DESIREDSPEED",speed)

    def start(self):
        sim=self.sim
        vissim=self.vissim
        vissim.SetWindow(0,0,100,100)
        sim.SetAttValue("ReSolution",1)
        self._add_eva_config()
        sim.RunContinuous()
    def set_period(self,period):
        sim=self.sim
        sim.SetAttValue("Period",period)