import librosa
from pedalboard import *
import noisereduce as nr
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

y, sr= librosa.load('XC941297.wav')

mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Real')
plt.show()

librosa.display.waveshow(y)

reduced_noise = nr.reduce_noise(y=y,sr=sr)

new_y = reduced_noise

mel_spectrogram = librosa.feature.melspectrogram(y=new_y, sr=sr)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Real')
plt.show()

librosa.display.waveshow(new_y)

sd.play(new_y, sr)

S, phase = librosa.magphase(librosa.stft(new_y))
rms = librosa.feature.rms(S=S)
fig, ax = plt.subplots(nrows=2, sharex=True)
times = librosa.times_like(rms)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='log Power spectrogram')

S = librosa.magphase(librosa.stft(new_y, window=np.ones, center=False))[0]
librosa.feature.rms(S=S)
plt.show()
