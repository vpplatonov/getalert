
Basicaly idea was got from https://github.com/giulbia/baby_cry_detection

Main kernel got from kaggle competition https://www.kaggle.com/anmour/svm-using-mfcc-features

code for Notebook https://www.kaggle.com/lesibius/data-augmentation-adding-signals


python 3.7 from conda package https://realpython.com/python-windows-machine-learning-setup/

### Clone this repo

### Clone ESC-50 repo https://github.com/karoldvl/ESC-50

```bash
git clone git@github.com:karoldvl/ESC-50.git
```
create folder structure
```
/output
    /model
    /dataset
    /prediction
```

from /libs folder
### Data set prepare & Feature extraction

```bash
python dataset.py
```

output
```bash
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1200/1200 [03:36<00:00,  5.26it/s]
done loading train mfcc
Feature names breathing, brushing_teeth, can_opening, cat, clapping, clock_alarm, clock_tick, coughing, cow, crow, crying_baby, dog, door_wood_creaks, 
door_wood_knock, drinking_sipping, footsteps, frog, glass_breaking, hen, insects, keyboard_typing, laughing, mouse_click, pig, rooster, sheep, sneezing, 
snoring, vacuum_cleaner, washing_machine
Class nums 30
```

### Train model & find best param & final train

```bash
python train.py
```

output 44.1 kHz
```bash
(960, 210)
(240, 210)
0.9152458416086409
0.625
0.6927083333333334
{'C': 4, 'gamma': 0.005}
SVC(C=4, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
0.7541666666666667
```

output 16 kHz
```bash
(960, 210)
(240, 210)
0.9142708246086968
0.578125
0.6588541666666666
{'C': 8, 'gamma': 0.001}
SVC(C=8, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
0.7666666666666667
```

8 kHz
```bash
(960, 210)
(240, 210)
0.9146139056018971
0.5572916666666666
0.6171875
{'C': 4, 'gamma': 0.005}
SVC(C=4, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
0.6833333333333333
```

### Prediction

see strategy.py
```
    Confident - all category the same as in the max probability place
    50/50 - as min as half in the max probability place
    Panic(default) - even if selected category present in the second probability place
```

```bash
python prediction.py --file_name V_2017-04-01+08_04_36=0_13.mp3
python prediction.py --file_name signal_9s.wav
```

see /output/prediction/prediction.txt

### Check recall on https://github.com/gveres/donateacry-corpus

output 44.1 kHz
```bash
python precision_recall.py
0.14450354609929078
```

by folders
```bash
donateacry-android-upload-bucket Audio: amr_nb (samr / 0x726D6173), 8000 Hz, mono, flt, 12 kb/s
0.09410112359550561
donateacry-ios-upload-bucket Audio: adpcm_ima_qt (ima4 / 0x34616D69), 16000 Hz, mono, s16p, 64 kb/s
0.23557692307692307
```

output 16 kHz
```bash
donateacry-android-upload-bucket Audio: amr_nb (samr / 0x726D6173), 8000 Hz, mono, flt, 12 kb/s
0.016853932584269662
donateacry-ios-upload-bucket Audio: adpcm_ima_qt (ima4 / 0x34616D69), 16000 Hz, mono, s16p, 64 kb/s
0.28125
```

8 kHz
```bash
donateacry-android-upload-bucket Audio: amr_nb (samr / 0x726D6173), 8000 Hz, mono, flt, 12 kb/s
0.23735955056179775
donateacry-ios-upload-bucket Audio: adpcm_ima_qt (ima4 / 0x34616D69), 16000 Hz, mono, s16p, 64 kb/s
0.4110576923076923
```

#### Check on https://github.com/giulbia/baby_cry_rpi

44.1 kHz
```bash
301 - Crying baby
0.7962962962962963
901 - Silence
0.0
902 - Noise
0.027777777777777776
903 - Baby laugh
0.1111111111111111
``` 

16 kHz
```bash
301 - Crying baby
0.6296296296296297
901 - Silence
0.0
902 - Noise
0.037037037037037035
903 - Baby laugh
0.12037037037037036
```

8 kHz
```bash
301 - Crying baby
0.8425925925925926
901 - Silence
0.0
902 - Noise
0.09259259259259259
903 - Baby laugh
0.06481481481481481
```


### On the image

first - original
second - after studio
third - w/o low frequency
