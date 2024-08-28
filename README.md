# Classification of %MVC in sEMG Signals
## Abstract
This project implements a classification model to determine the maximum volunatry contraction percentage (%MVC) of surface electromyography (sEMG) signals. The dataset used was collected from the biceps brachii of 12 patients at 10%, 30%, and 50% of maximum voluntary contraction. User metadata is explored through training data valuation techniques to gain insights into how an individual’s characteristics impact model output. The results are used in discussing data-centric approachs to EMG collection.

## Methods
### Preprocessing

<p align="center">
<img src="https://github.com/gagetylee/mvc-classification/assets/48107607/0c1d7546-bb46-42c8-a686-8edcc0b00613" width=90%>
</p>

**Channel Selection**

The Power-Correlation Ratio (PCR) Maximization method is used for dimensionality reduction. This technique selects channels that not only exhibit maximal output but also minimal correlation with each other. This ensures the inclusion of a broader array of channels, while prioritizing those that provide the most significant information.

**Filtering**

A 4th order band-pass filter was applied with a 10 Hz low-pass and 500 Hz high-pass. Afterwards, the signals are rectified and smoothed using a 100ms moving average filter. The resulting linear envelope provides a clearer representation of muscle contraction strength. 

### Feature Extraction

The extracted features include:
- maximum value
- standard deviation
- mean
- mean frequency

In addition to the calcualted features, patient metadata is used. This includes the weight(kg), height(cm), and age of the subject. These features were used for each of the 10 selected channels of each recorded EMG signal.

### Results

The model was trained on 80% of the data using the Random Forest algorithm. This resulted in 91% accuracy for the test set, and the following confusion matrix was generated.
<p align="center">
<img src="https://github.com/gagetylee/mvc-classification/assets/48107607/da3aba46-cce4-4df4-8c13-9c5372a93376" width=30%>
</p>

## Setup & Installation
To clone the repository and install the required dependencies through Conda, please run:
```
git clone git@github.com:gagetylee/mvc-classification.git
cd mvc-classification
conda env create -n mvc-proj -f requirements.yml
conda activate mvc-proj
```
### Dataset
Download and extract the dataset into the project directory. The dataset can be found [here](https://www.kaggle.com/datasets/gagetylee/semg-data).

## Future Plans
- Implement a deep-learning approach with PyTorch
- Augment dataset with synthetic EMG signals
