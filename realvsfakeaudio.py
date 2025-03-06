import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile

audio = wave.open("1.wav", "rb")

params_r = audio.getparams()

num_frames_r = audio.getnframes()

frame_rate_r = audio.getframerate()
 
audio_time_r = num_frames_r/frame_rate_r

print("Real: ", params_r)
print("Audio Time Real: ", audio_time_r)


y_r, sr_r= librosa.load('1.wav')
mel_spectrogram_r = librosa.feature.melspectrogram(y=y_r, sr=sr_r)
mel_spectrogram_db_r = librosa.power_to_db(mel_spectrogram_r, ref=np.max)
print("Real Spectrogram: ", mel_spectrogram_r)
print("Real DB", mel_spectrogram_db_r)

librosa.display.waveshow(y_r)
#plt.savefig("wave1.png")

print(mel_spectrogram_db_r.dtype)

zcr_r = librosa.feature.zero_crossing_rate(y_r)
print(zcr_r)

cent_r = librosa.feature.spectral_centroid(y=y_r, sr=sr_r)
print(cent_r)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db_r, x_axis='time', y_axis='mel', sr=sr_r, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Real')
plt.show()





audio_f = wave.open("44.wav", "rb")

params_f = audio_f.getparams()

num_frames_f = audio_f.getnframes()

frame_rate_f = audio_f.getframerate()
 
audio_time_f = num_frames_f/frame_rate_f

print("Fake: ", params_f)
print("Audio Time Fake: ", audio_time_f)



y_f, sr_f= librosa.load('44.wav')
mel_spectrogram_f = librosa.feature.melspectrogram(y=y_f, sr=sr_f)
mel_spectrogram_db_f = librosa.power_to_db(mel_spectrogram_f, ref=np.max)
print("Fake Spectrogram: ", mel_spectrogram_f)
print("Fake DB: ", mel_spectrogram_db_f)

librosa.display.waveshow(y_f)

zcr_f = librosa.feature.zero_crossing_rate(y_f)
print(zcr_f)

cent_f = librosa.feature.spectral_centroid(y=y_f, sr=sr_f)
print(cent_f)


plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db_f, x_axis='time', y_axis='mel', sr=sr_f, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Fake')
plt.show()