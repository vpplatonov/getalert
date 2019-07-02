import os
import copy

import numpy as np


def get_config():

    DATAROOT = './cry_detection/models'

    # FOLDER = 'X5D'  # See Readme.mdq
    # FOLDER = 'X5A'  # default for baby cry prediction with full baby cry(improved) dataset & scream & people_noise ( previous X5N ) & adult_cry
    # FOLDER = 'X5N'  # default for baby cry prediction with full baby cry dataset & scream & people_noise ( previous X5 )
    # FOLDER = 'X5'  # default for baby cry prediction with short baby cry dataset ( previous X5S )
    # FOLDER = 'X4S'  # default for baby cry prediction with full baby cry dataset
    # FOLDER = 'X2M16'  # the model was trained on 2sec with noise reduction samples
    FOLDER = 'X3M16'  # the model was trained on 3sec with noise reduction samples
    path_to_model = os.path.join(DATAROOT, FOLDER)

    path_to_conf = os.path.join(path_to_model, 'conf.npy')
    if not os.path.exists(path_to_conf):
        raise Exception('conf.npy file does not exist.')
    else:
        confX = np.load(path_to_conf).tolist()
#		# Fix old version of saved conf
#        if str(type(confX['folder'])).find('pathlib.WindowsPath') != -1:
#            _, confX = get_config_default()
    path_to_labels = os.path.join(path_to_model, 'labels.npy')
    if not os.path.exists(path_to_labels):
        raise Exception('labels.npy file does not exist.')
    else:
        labels = np.load(path_to_labels).tolist()

    confX['path_to_model'] = path_to_model
    confX['labels'] = labels
    confX['i2c'] = {i: l for i, l in enumerate(labels)}

    confX['normalize'] = 'featurewise'

    confX['samples'] = confX['sampling_rate'] * confX['duration']
    confX['dims'] = (confX['n_mels'], 1 + int(np.floor(confX['samples']/confX['hop_length'])), 1)

    confX['learning_rate'] = 0.0001
    confX['folder'] = FOLDER
    # enable NR filter
    confX['filter_flag'] = False
    # for denoise filter
    confX['channels'] = 1
    confX['sample_width'] = 2
    confX['vad_mode'] = 3
    # duration in seconds
    confX['vad_window_duration'] = 0.03
    # for normalization
    confX['headroom'] = 6
    confX['max_possible_amplitude'] = (2 ** (confX['sample_width'] * 8)) / 2
    return confX
