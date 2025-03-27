import numpy as np
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import sounddevice as sd
import IPython.display as ipd
import pandas as pd
from librosa import effects
import scipy.signal as sp
import statistics as st

def get_audio_split(y,sr):
    duration_ogn = len(y)/sr
    if(duration_ogn <= 30.0):

      data = 15 * sr
    elif(duration_ogn <= 15.0):
      data = 10 * sr 
    elif(duration_ogn <= 10.0):
      data = 5 * sr
    else:
      data = 30 * sr   

    return data  

def get_n_fft_val(y,sr):
  duration = len(y)/sr

  if(duration < 2.0):
    n_fft_val =512
  else:
    n_fft_val = 1024   

  return n_fft_val

def reduce_audio_noise(y,sr,n_fft_val):

  reduced_noise = nr.reduce_noise(y=y,sr=sr,stationary=False,prop_decrease=0.8,n_fft=n_fft_val,freq_mask_smooth_hz=300, time_mask_smooth_ms=50)  

  return reduced_noise

def get_compress(reduced_noise):
  
  y_comp = librosa.effects.preemphasis(reduced_noise, coef=0.97)

  return y_comp

def print_wave(y):
  
  librosa.display.waveshow(y)

def get_spect_db(y):
  audio_stft = librosa.stft(y)
  spect_db = librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max)

  return spect_db

def get_mel_spect_db(y,sr):
  mel_spect = librosa.feature.melspectrogram(y=y,sr=sr)
  mel_spect_db = librosa.power_to_db(mel_spect,ref=np.max)

  return mel_spect_db

def print_spect_db(spect_db):
  fig, ax = plt.subplots(figsize=(10, 5))
  stft_dis = librosa.display.specshow(spect_db,x_axis='time',y_axis='log',ax=ax)
  ax.set_title('Spectogram Example', fontsize=20)
  fig.colorbar(stft_dis, ax=ax, format=f'%0.2f')
  plt.show()

def print_mel_spect_db(mel_spect_db,sr):
  plt.figure(figsize=(10,4))
  librosa.display.specshow(mel_spect_db, x_axis='time', y_axis='mel',sr=sr,cmap='magma')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel Spectrogram')
  plt.show()  

def print_rms(rms,y,S):
  fig, ax = plt.subplots(nrows=2, sharex=True)
  times = librosa.times_like(rms)
  ax[0].semilogy(times, rms[0], label='RMS Energy')
  ax[0].set(xticks=[])
  ax[0].legend()
  ax[0].label_outer()
  librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])
  ax[1].set(title='log Power spectrogram')
  S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
  librosa.feature.rms(S=S)
  plt.show()

def get_avg_rms(rms):

  avg = rms.mean()
  return avg

def get_mode_rms(rms):
  for i in range(0,len(rms)):
    mode = st.mode(rms[i])
    
  return mode

def get_snip(y,sr):
  data = get_audio_split(y,sr)
  y = y[:data]
  return y



y, sr = librosa.load('CSA36386.ogg')

y = get_snip(y,sr) #only for CSA type files

print("BEFORE:")

print_wave(y)

spect_before = get_spect_db(y)
print_spect_db(spect_before)

mel_spect_before = get_mel_spect_db(y,sr)
print_mel_spect_db(mel_spect_before,sr)

S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)
print(rms.mean())

for i in range(0,len(rms)):
  mode = st.mode(rms[i])

print(mode)

print_rms(rms,y,S)

print("...CLEANING...")

n_fft = get_n_fft_val(y,sr)
reduced_noise = reduce_audio_noise(y,sr,n_fft)

y_reduced = reduced_noise

print("AFTER:")

print_wave(y_reduced)

spect_after = get_spect_db(y_reduced)
print_spect_db(spect_after)

mel_spect_after = get_mel_spect_db(y_reduced,sr)
print_mel_spect_db(mel_spect_after,sr)

S_reduced, phase = librosa.magphase(librosa.stft(y_reduced))
rms_reduced = librosa.feature.rms(S=S_reduced)
print(rms_reduced.mean())
for i in range(0,len(rms_reduced)):
  mode_r = st.mode(rms_reduced[i])

print(mode_r)

print_rms(rms_reduced,y_reduced,S_reduced)

print("...COMPRESSING...")

y_compressed = get_compress(reduced_noise)

print("AFTER COMPRESSION:")

print_wave(y_compressed)

spect_comp = get_spect_db(y_compressed)
print_spect_db(spect_comp)

mel_spect_comp = get_mel_spect_db(y_compressed,sr)
print_mel_spect_db(mel_spect_comp,sr)

S_compressed, phase = librosa.magphase(librosa.stft(y_compressed))
rms_compressed = librosa.feature.rms(S=S_compressed)
print(rms_compressed.mean())

for i in range(0,len(rms_compressed)):
  mode_c = st.mode(rms_compressed[i])

print(mode_c)

print_rms(rms_compressed,y_compressed,S_compressed)

print("Original Audio")
ipd.Audio(y,rate=sr,autoplay=False)

print("Noise Reduced Audio")
ipd.Audio(y_reduced,rate=sr,autoplay=False)

print("After Compression Audio")
ipd.Audio(y_compressed,rate=sr,autoplay=False)