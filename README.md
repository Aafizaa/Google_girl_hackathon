# Predicting Negative Slack in RTL Designs using Machine Learning

## Overview
This project aims to predict negative slack in RTL designs using machine learning models. It utilizes `RandomForestClassifier` and `GradientBoostingClassifier` to classify whether negative slack is present based on given RTL parameters. The dataset is processed, balanced (if necessary), and evaluated using key performance metrics.

## Features
- **Automatic Data Upload**: Uses Google Colab's `files.upload()` to import the dataset.
- **Feature Engineering**: Computes "Negative Slack (ps)" based on propagation delay and setup time.
- **Class Balancing**: Uses resampling techniques to address class imbalance.
- **Data Preprocessing**: Standardizes features using `StandardScaler`.
- **Model Training**: Implements `RandomForestClassifier` and `GradientBoostingClassifier`.
- **Performance Evaluation**: Computes Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
- **Model Persistence**: Saves trained models as `.pkl` files.
- **Prediction Storage**: Saves predictions as CSV files.
- **Google Colab Compatibility**: Enables file downloads for models and predictions.

## Requirements
To run this project, you need the following Python libraries:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```
Alternatively, use **Google Colab**, where these libraries are pre-installed.

## Usage
1. **Upload Dataset**: Run the script in Google Colab and upload your dataset.
2. **Check Columns**: Ensure that column names match the expected ones (`Propagation delay (register to flip-flop) (ps)` and `Setup time (ps)`).
3. **Feature Processing**: The script automatically computes `Negative Slack (ps)` and balances the dataset if needed.
4. **Train Models**: The script trains both classifiers and evaluates their performance.
5. **Results Visualization**: Confusion matrices and performance metrics are displayed.
6. **Save & Download Models**: Models and predictions are stored as files for future use.

## Output Files
- `rfc_model.pkl`: Trained `RandomForestClassifier` model.
- `gbc_model.pkl`: Trained `GradientBoostingClassifier` model.
- `predictions_rfc.csv`: Predictions from `RandomForestClassifier`.
- `predictions_gbc.csv`: Predictions from `GradientBoostingClassifier`.

## Model Evaluation
Each model's performance is measured using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## Notes
- Ensure the dataset has the required columns for calculating `Negative Slack (ps)`.
- The model assumes that negative slack values below zero are classified as `1` (critical), while non-negative values are `0` (non-critical).
- If the dataset has an imbalance, the minority class is upsampled to match the majority.

## Authors
Aafizaa K
Departement of biomedical engineeering
Chennai Institute Of Technology.



