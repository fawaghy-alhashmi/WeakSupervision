

# Automatic Labeling of Dermoscopy Images of Melanoma and Benign Keratosis-like Lesions Using Weak Supervision

State-of-the art supervised learning techniques require extensive collections of manually labeled training data. Such process of hand-labeling is notably time-consuming and costly especially when technical domain expertise is required. Which poses a significant hurdle in acquiring labeled datasets. The HAM1000 dataset, a part of the largest skin image challenge hosted by the ISIC (International Skin Imaging Collaboration) during the 2018 MICCAI conference boasts a vast repository of 10015 dermoscopic images motivating research in generating segmentations and classification submissions. Our work innovates within this domain, suggesting an approach to automatically label melanoma and benign keratosis-like lesions programmatically  through weak supervision, representing a novel exploration in skin imaging research that showcase potential advancements in automated labeling of medical imaging data.
![method ](https://github.com/fawaghy-alhashmi/WeakSupervision/assets/142748320/b2d9aec3-1db5-4462-982f-b237e91e3c41)

## Table of Contents

1. [Setup](#setup)
2. [Running the Notebooks](#running-the-notebooks)
3. [Files and Directories](#files-and-directories)


## Setup

### Google Colab Setup

1. **Install Required Packages:**
   - Open a code cell in your Colab notebook.
   - Run the following commands to install the required packages:

   ```python
   !pip install snorkel
   !pip install kaggle

## Running the Notebooks

### 1. main.ipynb

- Open and run the `main.ipynb` Colab Notebook.
- Ensure that the Kaggle API key (`kaggle.json`) is uploaded to the Colab environment, when running the fourth cell you should uploud your  Kaggle API key (`kaggle.json`) to uploud the dataset to the environment.

### 2. Data_Analytics_AI_project.ipynb

- Open and run the `Data_Analytics_AI_project.ipynb` Colab Notebook.
- The notebook focuses on the analysis of the metadata (`HAM10000_metadata.csv`).

## Files and Directories

- **`Dataset_AI_project.ipynb`:** Colab Notebook for dataset preparation.
- **`Data_Analytics_AI_project.ipynb`:** Colab Notebook for data analytics and visualization.
- **`kaggle.json`:** Kaggle API key file (required for downloading datasets).
- **`HAM10000_metadata.csv`:** Metadata file for the HAM10000 dataset.




