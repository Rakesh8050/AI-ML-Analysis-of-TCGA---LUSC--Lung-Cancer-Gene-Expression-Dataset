# README.md for TCGA Lung Cancer AI/ML Project

## Overview

This project focuses on the application of Artificial Intelligence (AI) and Machine Learning (ML) techniques to analyze lung cancer data from The Cancer Genome Atlas (TCGA). The goal is to build predictive models that can classify tumor samples based on gene expression data.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, ensure you have Python 3.x installed along with the necessary libraries. You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Data

The dataset used in this project is a CSV file containing gene expression levels for lung cancer samples. The data is structured as follows:

- **Columns**: Each column represents a gene, and the last column indicates the class label (normal or tumor).
- **Rows**: Each row corresponds to a different sample.

The dataset can be loaded using Pandas:

```python
import pandas as pd

data = pd.read_csv("LUSCexpfile.csv", sep=";")
```

## Preprocessing

The preprocessing steps applied to the dataset include:

1. **Handling Missing Values**: Checked for null values and confirmed none were present.
2. **Feature Scaling**: Standardized the features using `StandardScaler`.
3. **Dimensionality Reduction**: Applied Principal Component Analysis (PCA) to reduce the feature space while retaining 120 components.

Example code for preprocessing:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=120)
X_pca = pca.fit_transform(X_scaled)
```

## Models

Several machine learning models were implemented and evaluated:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **XGBoost**

Each model was trained on the training set and evaluated on the test set using metrics such as balanced accuracy, precision, and confusion matrix.

Example code for training a model:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

## Results

The performance of each model was compared based on balanced accuracy and precision. The KNN model achieved the highest balanced accuracy of 1.0000 during training and 0.9934 during testing.

### Model Comparison

| Model                  | Balanced Accuracy | Precision |
|------------------------|-------------------|-----------|
| Logistic Regression     | 0.9567            | 0.9933    |
| K-Nearest Neighbors     | 1.0000            | 0.9934    |
| Random Forest           | 0.9333            | 0.9667    |
| XGBoost                | 0.9667            | 0.9869    |

## Usage

To use this project, clone the repository and run the main script after ensuring all dependencies are installed:

```bash
git clone https://github.com/yourusername/tcga-lung-cancer-ai-ml.git
cd tcga-lung-cancer-ai-ml
python main.py
```

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

This README provides a comprehensive overview of the TCGA Lung Cancer AI/ML project, detailing installation instructions, data handling, preprocessing steps, model training, results, usage instructions, contribution guidelines, and licensing information.
