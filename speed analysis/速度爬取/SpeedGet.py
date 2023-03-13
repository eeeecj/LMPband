import requests
from datetime import datetime
import pandas as pd
import time

url="http://api.map.baidu.com/directionlite/v1/driving?origin={},{}&destination={},{}&ak={}"
keys=["c1WGmn9966orj1B5pE4Y4YuX52Ef1TMK","bhbWkiGioWUXBLdWdW0sBpmaIagUViHG",
"WKldMPMgRAu85OHZlL9ehDXeuR6HTlTb","9L5E8LTh6uKol97OXvqfqb48mi5Wotz2","nGZcmj9YSZMwG9iux9Gcu7a3S629jUBN",
"yzhEcsWDe4j6aZh2YVoF4Qnm6ipK9oh4","IPby47Ev7Hnrp8g5kKQi4KYbB34POUHr","kuIKYGPyGP9fwfRmPC9iP03oLGx9QfxY",
"0nDyAGhtGdNaCoM1zctaR5CkpViLYgtG","htF1cpjvGIgLvVLOU8kFWOyAQxFcRZ8A","ul1Q1FUGgFx2kpGG09rguX2WTHvSgLWO",
"CO618GM3PkYtaGtff2igAi6irgcpWexV","j03k7e7csEZZ95dYqZ23j2tXmmKctYt8"]
origin=["113.371974,22.519177",
        "113.375752,22.519498",
        "113.383365,22.519686",
        "113.388121,22.519578",
        "113.390502,22.519369",
        "113.392887,22.519152",
        "113.39785,22.518781",
        "113.399844,22.518576",
        "113.403307,22.518313",
        "113.40982,22.518213",
        "113.418785,22.520625",
        "113.428487,22.523968",
        "113.433242,22.525827"]
destination=["113.374265,22.519394",
             "113.38199,22.519653",
             "113.386675,22.519707",
             "113.389869,22.519423",
             "113.392096,22.519219",
             "113.396592,22.518856",
             "113.399844,22.518576",
             "113.402728,22.518351",
             "113.408221,22.517863",
             "113.41795,22.520392",
             "113.426641,22.523204",
             "113.432335,22.525481",
             "113.438362,22.52924"]
origin2=[
    "113.438208,22.529375",
    "113.431399,22.525319",
    "113.426261,22.523266",
    "113.421657,22.521621",
    "113.417754,22.520549",
    "113.407648,22.518103",
    "113.402496,22.518537",
    "113.399653,22.518838",
    "113.396374,22.519067",
    "113.391748,22.519489",
    "113.389834,22.519656",
    "113.381862,22.519739",
    "113.37406,22.519455",
]
destination2=[
    "113.432886,22.525924",
    "113.428376,22.524079",
    "113.422398,22.52183",
    "113.418657,22.520795",
    "113.409651,22.518366",
    "113.403345,22.518529",
    "113.40039,22.51875",
    "113.397775,22.518976",
    "113.392929,22.519443",
    "113.390324,22.51961",
    "113.383326,22.51981",
    "113.375686,22.519593",
    "113.372273,22.519309",
]
# waypoints=["113.374296,22.519394"]
name=["中三二路与悦来南路交叉口",
      "中三二路与民生路交叉口",
      "中三二路与华柏路交叉口",
      "中三二路与华苑大街交叉口",
      "中三三路与兴中道交叉口",
      "中三四路与库充大街人行横道",
      "中三三路与东苑路交叉口",
      "中三四路与起湾道交叉口",
      "中三四路与东文路交叉口",
      "中三四路与后塘路交叉口",
      "中三五路与长江路",
      "中山六路-上陂头路（T型口）",
      "中三五路与濠东路人行横道"]
name=name[::-1]

idx=0

# for i in range(len(origin)):
def get_speed(origin,destination,idx,name,keys):
    org=origin.split(",")
    des=destination.split(",")
    success=False
    attempts=0
    while attempts<3 and not success:
        idx+=1
        try:
            tmp=dict()
            key=keys[idx%len(keys)]
            req=requests.get(url.format(org[1],org[0],des[1],des[0],key))
            if req.status_code:
                data=req.json()
                tmp["date"]=datetime.now().strftime("%Y-%m-%d")
                tmp["time"]=datetime.now().strftime("%H:%M:%S")
                tmp["name"]=name
                tmp["origin"]=origin
                tmp["destination"]=destination
                tmp["distance"]=data.get("result").get("routes")[0].get("distance")
                tmp["duration"]=data.get("result").get("routes")[0].get("duration")
                tmp["speed"]=tmp["distance"]/tmp["duration"]*3.6
                success=True
                return tmp,idx
            else:
                success=False
        except Exception as e:
            print(e)
            attempts+=1
            print("try origin:{},destination:{},attempt:{}".format(origin,destination,attempts))
            if attempts==3:
                break
    return None,idx

if __name__=="__main__":
    results=[]
    now=datetime.now()
    tx=5
    while (datetime.now()-now).seconds<=60*60*4:
        if len(results)>1000:
            print("solve data 1000........")
            d=pd.DataFrame(results)
            d.to_csv("speed_in_"+str(tx),encoding="utf_8_sig",index=True)
            tx+=1
            results=[]
        for i in range(len(origin)):
            res,idx=get_speed(origin2[i],destination2[i],idx,name[i],keys)
            print("get name:{},origin:{},destination:{}".format(name[i],origin2[i],destination2[i]))
            if res!=None:
                results.append(res)
        time.sleep(10)