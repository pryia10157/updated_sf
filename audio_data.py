import wave
import pandas as pd

j = 1

df = pd.read_csv('singfake_bilibili_3.csv')

for i in range(0, len(df)): 
    file = f'{j}{".wav"}'
    audio = wave.open(file, "rb")

    params = audio.getparams()

    num_frames = audio.getnframes()

    frame_rate = audio.getframerate()
 
    audio_time = num_frames/frame_rate
    
    df.loc[i,['num_channels']] = params[0]
    df.loc[i,['sample_width']] = params[1]
    df.loc[i,['frame_rate']] = params[2]
    df.loc[i,['num_frames']] = params[3]
    df.loc[i,['audio_time']] = audio_time

    print(params)
    print("Audio Time: ", audio_time)
    j= j+1
  
df.to_csv('singfake_bilibili_audio.csv', index=False)    