import numpy as np
import python_speech_features
from tflite_runtime.interpreter import Interpreter
import librosa
import noisereduce as nr


from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json


model_path = 'mfcc_16_16.tflite'


def predict(path):

    pred, conf = prediction(path)

    return json.dumbs({
        # Post results to html => IP:PORT/prediction
        "status": "success",
        "prediction": str(pred),
        "confidence": str(conf),
        "upload_time": datetime.now()
    })


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    print(y_mean)
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def check_sample(signal, fs):
    # mask, y_mean = envelope(signal, fs, threshold=0.02)
    # signal = signal[mask]
    delta_sample = int(8000)

    if signal.shape[0] < delta_sample:
        print('This is in loop if.')
        print(type(signal))
        sample = np.zeros(shape=(delta_sample, ), dtype=np.float32)
        sample[:signal.shape[0]] = signal
        return sample, fs

    else:
        trunc = signal.shape[0] % delta_sample
        print('This is in loop else.')
        for cnt, i in enumerate(np.arange(0, signal.shape[0] - trunc, delta_sample)):
            start = int(i)
            stop = int(i + delta_sample)
            sample = signal[start:stop]
            return sample, fs


def prediction(file_name):

    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    signal, fs = librosa.load(file_name, sr=8000, mono=True)


    # Noise reduction
    noisy_part = signal[:]
    reduced_noise = nr.reduce_noise(signal, noise_clip=noisy_part, verbose=False)

    envo, y_mean = envelope(signal, fs, threshold=0.1)

    y = []

    for i, sig in enumerate(envo):

        if (sig):
            y.append(0.6)
        else:
            y.append(0)

        # y.append(sig)


    print(sig)


    plt.figure(1)
    plt.plot(reduced_noise)
    plt.plot(y)

    
    # plt.figure(2)
    # plt.plot(reduced_noise)
    
    # plt.figure(3)
    # plt.plot(signal[envo])
    # plt.show()


    # noisy_part = signal[0:100]
    # reduced_noise = nr.reduce_noise(signal, noise_clip=noisy_part, verbose=False)

    trimmed, index = librosa.effects.trim(signal, top_db=20, frame_length=512, hop_length=64)


    # sameple, sr = check_sample(trimmed, fs)

    # print(sameple.shape)
    # plt.figure(1)
    # plt.plot(sameple)
    # plt.show()

    plt.figure(2)
    plt.plot(trimmed)
    plt.show()

    # # Comput features
    # mfccs = python_speech_features.mfcc(sameple,
    #                                     samplerate=8000,
    #                                     winlen=0.256,
    #                                     winstep=0.050,
    #                                     numcep=16,
    #                                     nfilt=26,
    #                                     nfft=2048,
    #                                     preemph=0.97,
    #                                     ceplifter=0,
    #                                     appendEnergy=False,
    #                                     winfunc=np.hanning)

    # mfccs = mfccs.transpose()

    # # print(mfccs.shape[0], mfccs.shape[1])

    # # Make prediction from model
    # in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0],
    #                                         mfccs.shape[1], 1))

    # # print(in_tensor)

    # interpreter.set_tensor(input_details[0]['index'], in_tensor)
    # interpreter.invoke()

    # labels = ['bed', 'bird', 'cat', 'dog',
    #             'down', 'eight', 'five', 'four',
    #             'go', 'happy', 'house', 'left',
    #             'marvin', 'nine', 'no', 'off',
    #             'on', 'one', 'right', 'seven',
    #             'sheila', 'six', 'stop', 'three',
    #             'tree', 'two', 'up', 'wow', 'yes', 'zero']  # เพิ่มมา

    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # val = output_data[0]

    # v = max(val)

    # for i, j in enumerate(val):
    #     if j == v:
    #         pred = labels[i]
    #         confidence = v

    # return pred, confidence


if __name__ == '__main__':

    print(prediction('wav_20200705_175208.wav'))
