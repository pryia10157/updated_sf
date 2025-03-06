import pandas as pd
import numpy as np
import wave
import librosa


y, sr= librosa.load('35.wav')

zcr = librosa.feature.zero_crossing_rate(y).mean()
print("Zero Crossing Rate:",zcr)

rms = librosa.feature.rms(y=y).mean()
print("RMS: ", rms)

spect_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
print("Spectral Centroid: ", spect_cent)

tempo,_= librosa.beat.beat_track(y=y, sr=sr)
print("Tempo: ", tempo)

mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max).mean()
print("Average DB intensity: " ,mel_spectrogram_db)

flat = librosa.feature.spectral_flatness(y=y).mean()
print("Flatness: ", flat)

contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
print("Contrast: ", contrast)

harmonic_ratio = librosa.effects.harmonic(y).mean()
print("Harmonic Ratio: ", harmonic_ratio)