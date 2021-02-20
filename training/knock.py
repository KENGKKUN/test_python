import librosa.display
import matplotlib.pyplot as plt
import librosa


audio_data = 'samples/wav_20200705_174137.wav'
x, sr = librosa.load(audio_data, sr=8000)


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
