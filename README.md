## STELLAR porphyry copper data mining scripts

This repository contains the Python scripts and notebooks required to extract the data, train the models, and produce the figures from "Spatio-temporal copper prospectivity in the American Cordillera predicted by positive-unlabelled machine learning".

Training data can be extracted from the plate model and other input datasets using the `00a-extract_training_data.ipynb` and `00b-extract_grid_data.ipynb` notebooks.
The first of these notebooks extracts data for the positive/negative mineral deposit observations in `data/deposit_data.csv`, to be used for training and testing.
The second notebook extracts data for a regular grid of points, to be used to create the time-dependent mineral prospectivity maps.

Alternatively, the above process can be skipped by using pre-prepared data downloaded from the Zenodo repository ([zenodo.org/record/8157691](https://zenodo.org/record/8157691)).
Running the notebooks in sequence, beginning with `01-create_pu_classifier.ipynb`, will automatically download this data to a directory named `prepared_data`.

### To run the notebooks:
1. Create a `conda` environment using the `environment.yml` file: `conda env create --file environment.yml`
1. Run the following notebooks to download and extract training data from Zenodo (optiona):
    - `00a-extract_training_data.ipynb`
    - `00b-extract_grid_data.ipynb`
1. Run these notebooks to train a PU classifier and create prospectivity maps:
    - `01-create_pu_classifier.ipynb`
    - `02-create_probability_maps.ipynb`
    - `03-create_probability_animation.ipynb`
1. Run these notebooks to train an SVM classifier and create prospectivity maps (optional):
    - `01a-create_svm_classifier_svm.ipynb`
    - `02a-create_probability_maps_svm.ipynb`
    - `03a-create_probability_animation_svm.ipynb`
1. Run the `01b-cross_validation.ipynb` notebook to perform cross validation and compare PU and SVM models

### To create the figures:
To create the figures used in the article, run the following notebooks:
- `Fig-01-02-probability_snapshots.ipynb`
- `Fig-03-time_dependent_present_day_probabilities.ipynb`
- `Fig-04-feature_importance.ipynb`
- `Fig-05-partial_dependence.ipynb`
- `Fig-06-performance.ipynb`
