import sys
from predict import audio_prediction
import logging


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pred = audio_prediction()
    logging.debug(f'prediction: {pred}')
    sys.exit(int(pred))
