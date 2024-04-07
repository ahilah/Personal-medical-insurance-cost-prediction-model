# Personal Medical Insurance Cost Prediction

## [[Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)] description features
- age: age of primary beneficiary.
- sex: insurance contractor gender (female/male).
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.
- children: number of children covered by health insurance / Number of dependents.
- smoker: smoking (yes/no).
- region: the beneficiary's residential area in the US (northeast/southeast/southwest/northwest).
- charges: individual medical costs billed by health insurance.

## Request
This model aims to assist an insurance company [UnitedHealth Group](https://www.unitedhealthgroup.com/) in forecasting the medical expenses of its clients. The primary goal is to enhance insurance processes and ensure more accurate cost estimations for clients.
'charges' is a target to predict.


### Requirements

    python 3.7

    numpy==1.17.3
    pandas==1.1.5
    sklearn==1.0.0


### Running:

    To run the demo, execute:
        python predict.py 

    After running the script in the folder './data/' will be generated <prediction_results.csv> 
    The file has 'charges_pred' column with the predicted result value.

    The input is expected csv file in the folder './data/final/' with a name <new_data.csv>. The file should have all features columns. 

### Training a Model:

    Before you run the training script for the first time, you must create dataset. The file <train_data.csv> should contain all features columns and target for prediction Churn.
    After running the script the "param_dict.pickle" and "finalized_model.saw" will be created.
    
    Run the training script:
        python train.py

    The model achieves a score of 90%, with an error of approximately 10%.
    Note that there is currently no fraud check implemented.


#### Scripts
The project includes several Python scripts:
- predict.py. Implements the prediction model using machine learning techniques. It loads the model and predicts client's medical charges based on input features.
- model-training.py. Develops and trains the machine learning model using a GradientBoostingRegressor. Contains regression evaluation metrics and model training process using various algorithms such as Linear Regression, Ridge Regression, Lasso Regression, K-Nearest Neighbors, Random Forest, XGBoost, Decision Tree, Gradient Boosting, and Extra Tree.
- feature-engineering.py. Performs feature engineering tasks such as categorical encoding.
- split-data. Splits the data to Train/Test data.

#### Model Evaluation Metrics
The model performance is evaluated using the following metrics:
1. Score
2. Mean Squared Error (MSE)
3. Mean Absolute Error (MAE)
4. R2 Score
The results are saved in a CSV file named 'gradient-boosting-metrics.csv' in the folder './models/'.

#### Model Selection
The notebook model-training.ipynb iterates through various regression algorithms and selects the best-performing model based on the root mean squared error (RMSE) on the test data. The selected models are saved using pickle for future use.

#### Feature Importance
The notebook also analyzes the feature importance using the selected model (Gradient Boosting Regressor). The top five important features are visualized using a bar plot.