#!/bin/sh
wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh
rm Anaconda3-4.0.0-Linux-x86_64.sh

conda install scikit-learn pandas jupyter ipython imblearn seaborn numpy \
matplotlib scipy
