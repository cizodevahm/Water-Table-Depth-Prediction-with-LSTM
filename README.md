# Water-Table-Depth-Prediction-with-LSTM
This Jupyter Notebook is designed for predicting water table depth using an LSTM (Long Short-Term Memory) model. The notebook includes data preprocessing, model training, evaluation, and visualization of the results.

## Notebook Structure

The notebook is structured as follows:

1. **Import Libraries**: Import necessary libraries such as pandas, sklearn, matplotlib, numpy, and custom models.
2. **Data Preprocessing**: Load and preprocess the data, including standardization.
3. **Model Definition**: Define the LSTM model architecture.
4. **Model Training**: Train the LSTM model on the training data.
5. **Model Evaluation**: Evaluate the model using R-squared and Root Mean Squared Error (RMSE) metrics.
6. **Visualization**: Visualize the predicted vs actual water table depth.
7. **Model Checkpoints**: Save and load model checkpoints for future use.

## Detailed Description

### 1. Import Libraries

The following libraries are imported at the beginning of the notebook:
- pandas
- sklearn.preprocessing.StandardScaler
- sklearn.metrics.r2_score, mean_squared_error
- matplotlib.pyplot
- numpy
- Custom models from `models.py`

### 2. Data Preprocessing

The data is loaded from a CSV file (`data/demo.csv`). The features (inputs) and target variable (output) are extracted and standardized using `StandardScaler`.

### 3. Model Definition

An LSTM model with fully connected layers is defined using the `LSTM_FC_Model` class from the custom models.

### 4. Model Training

The LSTM model is trained on the training data for a specified number of iterations (`iters=20000`). The training process includes:
- Setting learning rate and dropout probability
- Iteratively fitting the model to the training data
- Saving the model parameters to checkpoints

### 5. Model Evaluation

The trained model is evaluated on the test data using the following metrics:
- R-squared
- Root Mean Squared Error (RMSE)

### 6. Visualization

The predicted water table depth is plotted against the actual measurements using matplotlib.

### 7. Model Checkpoints

The model parameters are saved to checkpoints during training. The notebook also includes code to load the model parameters from checkpoints and make predictions on new data.

## Results

The notebook provides the following results:
- R-squared value of water table depth prediction
- Root Mean Squared Error (RMSE) of water table depth prediction
- Visualization of predicted vs actual water table depth

## Usage

To use this notebook, follow these steps:
1. Ensure you have all necessary libraries installed.
2. Place your data in the specified CSV file (`data/demo.csv`).
3. Run each cell in the notebook sequentially.

## Conclusion

This notebook demonstrates how to build, train, evaluate, and visualize an LSTM model for predicting water table depth. The use of model checkpoints allows for saving and loading model parameters for future use.
