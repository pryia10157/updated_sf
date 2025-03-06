import asyncio
from bilibili_api import video, Credential, user
from bilibili_api import comment, sync
import pandas as pd
import time


SESSDATA = ""
BILI_JCT = ""
BUVID3 = ""

def get_bvid(url):
    
    str_ar = url.split('/')
    bvid_url = str_ar[4]
    alt_id = bvid_url.split('?')
    bvid_id = alt_id[0]
    
    return bvid_id

def get_website(url):
    
    str_ar = url.split('/')
    website = str_ar[2]
    return website
    

def is_bilibili(str):
    
    if(str == "www.bilibili.com"):
         return True
    else:
        return False


async def get_User_ID(id,cred,i,df):
    await asyncio.sleep(1)
    data = video.Video(bvid=id, credential=cred)
    try:
        vid_info = await data.get_info()
        
    except:
        print("no data")
        user_id = 0
    else:
        owner = vid_info['owner']
        user_id = owner['mid']
        df.loc[i,['user_id']] = owner['mid']
    
    return user_id    

async def get_user_followers(user_id,cred,i,df):
    await asyncio.sleep(1)
    usr = user.User(uid=user_id,credential=cred)
    try: 
        user_data = await usr.get_relation_info()
    except:
        print("no data")
        user_followers = 0
    else:
        user_followers = user_data['follower']
        df.loc[i,['subscribers']] = user_data['follower']
    
    return user_followers

async def  get_username(user_id,cred,i,df):
    await asyncio.sleep(1)
    print(user_id)
    usr = user.User(uid=user_id,credential=cred)
    await asyncio.sleep(1)
    try:
        user_data = await usr.get_user_info()     
    except:
        print("no data")
        username = "not found"
    else:
        username = user_data['name']
        df.loc[i,['username']] = user_data['name']
   
    return username

async def main():
    
    
 df = pd.read_csv('singfake_bilibili_user.csv')
 cred = Credential(sessdata=SESSDATA, bili_jct=BILI_JCT, buvid3=BUVID3)
 

 df['username'] = df['username'].astype(str)
 
 for i in range(0, len(df)):
     url = df.loc[i,"url"]
     web = get_website(url)
          
     if(is_bilibili(web) == True):
            
         id = get_bvid(url)
         print(id)
         await asyncio.sleep(1)
         uid = await asyncio.gather(get_User_ID(id,cred,i,df))
         userid = uid[0]
         print(userid)  
         await asyncio.sleep(1)
         time.sleep(3)
         usrnme = await asyncio.gather(get_username(userid,cred,i,df))
         print(usrnme)
         await asyncio.sleep(1)
         subs = await asyncio.gather(get_user_followers(userid,cred,i,df))
         print(subs)
         #print(df.loc[i])

 df.to_csv('singfake_bilibili_user_stats.csv', index=False)       
 
 
 
asyncio.run(main())   