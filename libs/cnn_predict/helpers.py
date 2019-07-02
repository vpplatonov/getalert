import time
import librosa
import logging

import numpy as np

from .utils import (
    remove_noise, file_to_array, add_padding, 
    normalize_npy, return_speech_array
)


logger = logging.getLogger("app.py")


def file_after_nr(audio_samples, conf):
    # read wav file
    # inp_audio = await file_to_array(wav_path, conf)
    # adding padding for audio
    inp_audio = add_padding(audio_samples, conf)
    # audio normalization
    inp_audio = normalize_npy(inp_audio, conf)
    # find chunks with speech:
    _, _, _, not_speech = return_speech_array(inp_audio, conf)
    # denoised sound
    if not_speech != []:
        t0 =time.time()
        out_audio = remove_noise(audio_clip=inp_audio.astype(np.float32),
                                 noise_clip=not_speech.astype(np.float32),
                                 conf=conf)
        logger.info("Time spent on removing noise - {} sec".format(time.time() - t0))
    else:
        out_audio = inp_audio
    # audio normalization
    out_audio = normalize_npy(out_audio, conf)
    return out_audio.astype(np.float32), conf['sampling_rate']


def file_after_vad(audio_samples, conf):
    # read wav file
    # inp_audio = await file_to_array(wav_path, conf)
    # adding padding for audio
    inp_audio = add_padding(audio_samples, conf)
    # audio normalization
    inp_audio = normalize_npy(inp_audio, conf)
    # find chunks with speech:
    full_audio, _, _, _ = return_speech_array(inp_audio, conf)
    return full_audio.astype(np.float32)


def read_audio(conf, audio_samples, padding=True):
    full_time_range = {}
    if 'filter_flag' not in conf or not conf['filter_flag']:
        # y, sr = librosa.load(audio_samples, sr=conf['sampling_rate'])
        y, sr = (audio_samples.astype(np.float32), conf['sampling_rate'])
        # full audio with zeros instead of not speech
        # y, sr = (await file_after_vad(audio_samples, conf), conf['sampling_rate'])
    else:
        y, sr = file_after_nr(audio_samples, conf)
    # trim silence
    # if 0 < len(y):  # workaround: 0 length causes error
    #     y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf['samples']:  # long enough
        if conf['audio_split'] == 'head':
            y = y[0:0 + conf['samples']]
    else:  # pad blank
        full_time_range['start'] = 0.0
        if padding:
            padding = conf['samples'] - len(y)  # add padding at both ends
            offset = padding // 2
            # Adding range only for chuck which is lower than config duration
            y = np.pad(y, (offset, conf['samples'] - len(y) - offset), 'constant')
            # full_time_range['stop'] = int(conf['samples'] / sr * 1000)
        # else:
        # return real time for sensitives
        full_time_range['stop'] = int(len(y) / sr * 1000.0)

    return full_time_range, y


def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf['sampling_rate'],
                                                 n_mels=conf['n_mels'],
                                                 hop_length=conf['hop_length'],
                                                 n_fft=conf['n_fft'],
                                                 fmin=conf['fmin'],
                                                 fmax=conf['fmax'])
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    # (48, 504)
    return spectrogram


def read_as_melspectrogram(conf, audio_samples, debug_display=False):
    time_range, x = read_audio(conf, audio_samples)
    mels = audio_to_melspectrogram(conf, x)

    if debug_display:
        # IPython.display.display(IPython.display.Audio(x, rate=conf['sampling_rate']))
        # show_melspectrogram(mels, conf)
        pass

    return time_range, mels


def value_to_millisec(conf, value):
    return int((value * conf['hop_length']) / conf['sampling_rate'] * 1000)


def split_long_data(conf, X, time_range, big_step=True):
    # Splits long mel-spectrogram data with small overlap
    L = X.shape[1]
    one_length = conf['dims'][1]
    loop_length = int(one_length * 0.9)
    min_length = int(one_length * 0.2)

    # different approach to predict
    if big_step:
        step_range = L // loop_length + 1
        step = loop_length
    else:  # small step
        step_range = (L - loop_length + min_length) // min_length
        step = min_length

    if time_range:
        for idx in range(step_range):
            cur = step * idx
            rest = L - cur
            if one_length <= rest:
                yield X[:, cur:cur+one_length], time_range

    else:
        # print(' sample length', L, 'to cut every', one_length)
        for idx in range(step_range):
            cur = step * idx
            rest = L - cur
            if one_length <= rest:
                time_range = {'start': value_to_millisec(conf, cur), 
                              'stop': value_to_millisec(conf, cur + one_length - 1)}
                yield X[:, cur:cur+one_length], time_range
            elif min_length <= rest:
                cur = L - one_length
                time_range = {'start': value_to_millisec(conf, cur), 
                              'stop': value_to_millisec(conf, cur + one_length - 1)}
                yield X[:, cur:cur+one_length], time_range
