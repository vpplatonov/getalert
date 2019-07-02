import os
import struct
import webrtcvad
import logging
import librosa
import scipy.signal
import grpc

import numpy as np
import tensorflow as tf

from math import log
from itertools import groupby
from pydub import AudioSegment

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


logger = logging.getLogger("app.py")


def millisec_to_value(value, conf):
    return int((value / 1000) * conf['sampling_rate'])


async def file_to_array(wav_path, conf):
    # read wav file
    inp_audio = AudioSegment.from_file(wav_path, "wav")
    # set the sample rate
    inp_audio = inp_audio.set_frame_rate(conf['sampling_rate'])
    # set the numbers of channels
    inp_audio = inp_audio.set_channels(conf['channels'])
    # convert wav to 16bit
    inp_audio = inp_audio.set_sample_width(conf['sample_width'])
    # convert to vector array and return
    return np.array(inp_audio.get_array_of_samples())


async def if_speech_exist(samples, conf):
    # exist issue when the VAD is outside the function
    # create voice activity detection class
    VAD = webrtcvad.Vad()
    # set aggressiveness from 0 to 3
    VAD.set_mode(conf['vad_mode'])

    sample_rate = conf['sampling_rate']

    raw_samples = struct.pack("%dh" % len(samples), *samples)
    # duration in seconds
    window_duration = conf['vad_window_duration']
    samples_per_window = int(window_duration * sample_rate + 0.5)
    bytes_per_sample = 2

    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
        try:
            is_speech = VAD.is_speech(raw_samples[start * bytes_per_sample:stop * bytes_per_sample], sample_rate=sample_rate)
            if is_speech:
                return True
        except:
            continue
    return False


async def send_on_predict(hots_port, model_name, version, data):
    """Send array to the Tensorflow Serving.

    Args:
        hots_port: The host:port string.
        model_name: The model name.
        version: The model version.
        data: The numpy array.

    Returns:
        predictions
    """
    try:
        # create the channel
        channel = grpc.insecure_channel(hots_port)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel) 
        # create the request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.version.value = version
        # convert the data
        send_data = tf.contrib.util.make_tensor_proto(data.astype(np.float32))
        request.inputs['input_array'].CopyFrom(send_data)
        # send on prediction
        result = stub.Predict(request, 10.0)  # 10 secs timeout
        return np.expand_dims(result.outputs["dense_2/Softmax:0"].float_val, axis=0)
    except Exception as err:
        logging.exception("Exception occurred")
        return np.array([])


async def get_feed_id(file_name):
    """Get FeedID from the file name"""
    if '\\' in file_name:
        feed_id = file_name.split('\\')[-1].split('_')[0]
    elif '/' in file_name:
        feed_id = file_name.split('/')[-1].split('_')[0]
    else:
        feed_id = ''
    return feed_id


async def predict_category(predictions, time_ranges, category='crying_baby', strategy='Panic', threshold=0.65):
    """ Is the category in predictions

    Strategy plan
    Confident - all category the same as in the max probability place
    50/50 - as min as half in the max probability place
    Panic(default) - even if selected category present in the second probability place

    :param
    predictions: list of Dict with {category: probabilities}
    time_ranges: list of Dict with {'start': <int>, 'stop': <int>}
    category: for seek in prediction
    strategy:
    :return: list of true ranges
    """
    pred_ranges = [time_ranges[ind] for ind, prediction in enumerate(predictions) for cat in [list(prediction.keys())[0]] if cat == category and prediction[category] > threshold]
    return {
        'Full': pred_ranges if len(pred_ranges) == len(predictions) else [],
        'Half': pred_ranges if len(pred_ranges) > len(predictions) / 2.0 else [],
        'Once': pred_ranges if len(pred_ranges) >= 1 else [],
        'Panic': [time_ranges[ind] for ind, prediction in enumerate(predictions) for cat in list(prediction.keys()) if cat == category],
        'Only_first': [time_ranges[ind] for ind, prediction in enumerate(predictions) if list(prediction.keys())[0] == category and list(prediction.values())[0] >= threshold]
    }[strategy]


async def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


async def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


async def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


async def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


async def remove_noise(audio_clip, noise_clip, conf, n_grad_freq=2, n_grad_time=4,
                 n_fft=2048, win_length=2048, hop_length=512, n_std_thresh=1.5,
                 prop_decrease=1.0):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)

    Returns:
        array: The recovered signal with noise subtracted

    """
    # STFT over noise
    noise_stft = await _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = await _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    # STFT over signal
    sig_stft = await _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = await _amp_to_db(np.abs(sig_stft))
    # Calculate value to mask dB to
    mask_gain_dB = np.min(await _amp_to_db(np.abs(sig_stft)))
    # print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (await _db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    # recover the signal
    recovered_signal = await _istft(sig_stft_amp, hop_length, win_length)
    return recovered_signal


async def add_padding(sample, conf):
    samples_per_window = int(conf['vad_window_duration'] * conf['sampling_rate'] + 0.5)
    module = len(sample) % samples_per_window
    if not module == 0:
        return np.concatenate([sample, np.zeros(samples_per_window - module, dtype=int)])
    return sample


async def db_to_float(db, using_amplitude=True):
    """
    Converts the input db to a float, which represents the equivalent
    ratio in power.
    """
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:  # using power
        return 10 ** (db / 10)


async def ratio_to_db(ratio, val2=None, using_amplitude=True):
    """
    Converts the input float to db, which represents the equivalent
    to the ratio in power represented by the multiplier passed in.
    """
    ratio = float(ratio)

    # accept 2 values and use the ratio of val1 to val2
    if val2 is not None:
        ratio = ratio / val2

    # special case for multiply-by-zero (convert to silence)
    if ratio == 0:
        return -float('inf')

    if using_amplitude:
        return 20 * log(ratio, 10)
    else:  # using power
        return 10 * log(ratio, 10)


async def normalize_npy(seg, conf):
    """
    headroom is how close to the maximum volume to boost the signal up to (specified in dB)
    """
    peak_sample = seg.max() + 1
    bowl_sample = seg.min() - 1
    if peak_sample >= conf['max_possible_amplitude'] or bowl_sample <= -conf['max_possible_amplitude']:
        target_peak = conf['max_possible_amplitude'] * await db_to_float(-conf['headroom'])
        needed_boost = await ratio_to_db(target_peak / conf['max_possible_amplitude'])
        seg = (seg * await db_to_float(float(needed_boost)))

    return seg.astype(np.int16)


async def return_speech_array(chunk, conf):
    # exist issue when the VAD is outside the function
    # create voice activity detection class
    VAD = webrtcvad.Vad()
    # set aggressiveness from 0 to 3
    VAD.set_mode(conf['vad_mode'])

    samples = chunk
    sample_rate = conf['sampling_rate']

    raw_samples = struct.pack("%dh" % len(samples), *samples)
    # duration in seconds
    window_duration = conf['vad_window_duration']
    samples_per_window = int(window_duration * sample_rate + 0.5)
    bytes_per_sample = 2
    segments = []
    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
        try:
            is_speech = VAD.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], sample_rate=sample_rate)
            segments.append(dict(start=start, stop=stop, is_speech=is_speech))
        except:
            continue

    # return chunks with speech
    full_with_zeros = []
    # speech values
    speech_groups = []
    # speech time ranges
    sp_time_ranges = []
    # not speech values
    not_speech_groups = []
    # not speech time ranges
    # nsp_time_ranges = []
    for key, group in groupby(segments, lambda x: x['is_speech']):
        l_group = list(group)
        values = np.concatenate([samples[sg['start']:sg['stop']] for sg in l_group])
        times = {'start': int(l_group[0].get('start')), 'stop': int(l_group[-1].get('stop'))}
        if key:
            full_with_zeros = np.concatenate([full_with_zeros, values])
            speech_groups.append(values)
            sp_time_ranges.append(times)
        else:
            full_with_zeros = np.concatenate([full_with_zeros, np.zeros(times['stop'] - times['start'])])
            not_speech_groups = np.concatenate([not_speech_groups, values])
            # nsp_time_ranges.append(times)

    return full_with_zeros, speech_groups, sp_time_ranges, not_speech_groups
