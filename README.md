# KDD-Cup-AutoML-5
KDD Cup AutoML Track 5th solution

## make submission file

```
$ mkdir tmp && cp -r ./optable_submission tmp/optable_submission
```

Disable Develop Mode(tmp/optable_submission/model.py)
```
DEVELOP_MODE = False
```

make zip
```
$ cd tmp/optable_submission && zip optable.zip optable_package/* -r && rm -rf optable_package
$ cd .. && zip mysubmission.zip optable_submission/*
```