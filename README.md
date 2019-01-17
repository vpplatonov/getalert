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
Feature names breathing, brushing_teeth, can_opening, cat, clapping, clock_alarm, clock_tick, coughing, cow, crow, crying_baby, dog, door_wood_creaks, door_wood_knock, drinking_sipping, footsteps, frog, glass_breaking, hen, insects, keyboard_typing, laughing, mouse_click, pig, rooster, sheep, sneezing, snoring, vacuum_cleaner, washing_machine
Class nums 30
```

### Train model & fins best param & final train

```bash
python train.py
```

output
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