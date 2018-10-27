# salt-identification

### TGS Salt Identification Challenge

The goal of this competition was to identify and segment salt deposits in
seismic images. The data includes (1) the seismic images, (2) masks showing the
portions of the training images containing salt, and (3) the depths at which
each of the images were taken. For an in-depth explanation of the contest, see the
official
[Kaggle page](https://www.kaggle.com/c/tgs-salt-identification-challenge).

### Final results

I placed 384<sup>th</sup> out of 3,234 teams, which is in the top
12<sup>th</sup> percentile. According to Kaggle, there were nearly 4,000
participants, making it one of their biggest competitions of the year. My Kaggle
profile can be found [here](https://www.kaggle.com/samwaterbury).

### What's in this repository?

The following files contain all of the code used to make my final predictions:

* `main.py` forms the backbone of the project. This file contains the code which
constructs the dataset, calls the models, and writes predictions to a file.

* `model.py` fully implements the deep residual network I built for this
competition in Tensorflow.

* `utilities.py` contains several functions used by the model. This includes the
specialized loss functions and a custom accuracy metric which matches the
competition's scoring mechanism.

The directories `data/` and `output/` are omitted from this repository. To
download the data and create these directories automatically,
run `setup.sh` from the command line.

Lastly, `papers/` contains several research papers I used as references for my
techniques and model architecture. All of these are freely available on the
internet.

### How do I run it?

This project is designed for Python 3.6+. The simplest
way to get the required packages is with an Anaconda distribution. In addition,
you must install the [Kaggle API](https://github.com/Kaggle/kaggle-api) to
obtain the data.

Then, clone this repository, run `setup.sh` to obtain the data, and run
`main.py` to generate predictions.
