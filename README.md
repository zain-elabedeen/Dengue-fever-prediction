# Dengue-fever-prediction

## Overview

This repo provides a solution to the DengAI problem from https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread.

The data provided by the website are part of this repo and can be found in the folder ```data/raw```.


## How to install 

We recommend using a virtual environment, e.g. using Conda, and installing a current Python version: 

```
conda create --name=<env_name> python
```

Kedro is used for running the processing pipeline and must be installed in the virtual environment: 

```
pip install kedro
```

All other requirements are listed in the requirements file and can be installed as such: 
```
pip install -r requirements.txt
```

## How to run the Kedro pipeline

You can run the Kedro project with:

```
kedro run
```

## Results:

The resulting predictions will be stored in CSV format in the file 

```
data/02_intermediate/submission_data.csv
```

(Sorry for that.)