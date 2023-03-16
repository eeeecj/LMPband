# %%
import numpy as np

class Degree():
    def __init__(self,t,qsum,qmax,inlane,outlane) -> None:
        self.t=t
        self.qsum=qsum
        self.qmax=qmax
        self.inlane=inlane
        self.outlane=outlane
    def get(self):
        return 0.5/(1+self.t)*((self.inlane*self.qmax)/(self.qsum)-1)
# %%
qb1_max=np.array([564,564,632,265,611,924,508,644,733,877,506,658])
qb2_max=np.array([541,651,546,210,874,1291,934,430,595,326,406,699])
qb1=np.array([944,1220,937,691,924,924,854,1200,861,1039,658,658])
qb2=np.array([1052,1012,1082,565,1215,1291,1291,970,886,480,592,853])
d=np.array([800,520,500,490,370,254,585,1020,409,547,566,831])
spd1=np.array([32.058972,23.996858,15.724018,15.256098,29.385298,19.779803,18.120899,33.667991,33.479902,31.500023,34.247889,30.157781])
spd2=np.array([22.188947,19.430474,11.809555,20.727834,20.767901,18.893493,18.062931,40.049992,21.412021,22.836887,23.097506,34.374107])
a=Degree(d/spd1*3.6/60,qb1,qb1_max,3,4)
b=Degree(d/spd2*3.6/60,qb2,qb2_max,3,4)
c=np.max(np.array([a.get(),b.get()]),axis=0)
# %%
