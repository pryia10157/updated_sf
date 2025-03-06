import wave
import pandas as pd
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

j = 1

df = pd.read_csv('singfake_bilibili_audio_6.csv')


for i in range(0, len(df)): 
    
    y, sr= librosa.load(f'{j}{".wav"}')
    
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    spect_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    tempo,_= librosa.beat.beat_track(y=y, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max).mean()
    flat = librosa.feature.spectral_flatness(y=y).mean()
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    harmonic_ratio = librosa.effects.harmonic(y).mean()
    
    df.loc[i,['avg_zero_cross_rate']] = zcr
    df.loc[i,['avg_rms']] = rms
    df.loc[i,['avg_spect_cent']] = spect_cent
    df.loc[i,['avg_tempo']] = tempo
    df.loc[i,['avg_db_inten']] = mel_spectrogram_db
    df.loc[i,['avg_flatness']] = flat
    df.loc[i,['avg_contrast']] = contrast
    df.loc[i,['avg_harmonic_ratio']] = harmonic_ratio
    
    print(j)
    print("Zero Crossing Rate:",zcr)
    print("RMS: ", rms)
    print("Spectral Centroid: ", spect_cent)
    print("Tempo: ", tempo)
    print("Average DB intensity: " ,mel_spectrogram_db)
    print("Flatness: ", flat)
    print("Contrast: ", contrast)
    print("Harmonic Ratio: ", harmonic_ratio)
    
    j=j+1
   
  
df.to_csv('singfake_bilibili_audio_features.csv', index=False)    
print("Done.")