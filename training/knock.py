import librosa
audio_data = 'wav_20200705_174137.wav'
x, sr = librosa.load(audio_data, sr=8000)


import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)