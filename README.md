# Startups_multiples
# Simplified Terminal-Based Prediction Script
This repository contains a simplified Python script that replaces the GUI components of the original code with a terminal-based interface. The script allows users to input data efficiently, particularly for categorical variables like industries represented by dummy variables.

## Table of Contents
Overview
Key Features
Prerequisites
Installation
Usage
Running the Script
Providing Inputs
Example Interaction
Code Explanation
Imports and Environment Setup
Data Loading and Preprocessing
Model Building and Training
Saving and Loading Models
Input Functions for Prediction
Main Execution Loop
Extending the Script
Adding More Categorical Variables
Adjusting for Column Naming Conventions
Error Handling
Conclusion
License
Overview

The original script included GUI components using tkinter to collect user inputs and display predictions. This updated script removes these GUI elements and introduces a terminal-based input system. Users can now provide inputs for prediction directly through the terminal, streamlining the process, especially for categorical variables represented by dummy variables.

## Key Features
Terminal-Based Input: Users can input data directly in the terminal.
Categorical Variable Handling: Efficient input system for categorical variables like 'Industry'.
Multiple Predictions: Allows users to make multiple predictions without restarting the script.
Error Handling: Includes validation and defaulting mechanisms for incorrect inputs.
Prerequisites
```python
Python 3.x
Required Python libraries:
numpy
pandas
scikit-learn
tensorflow
joblib
openpyxl (for reading Excel files)
```

## Installation
Clone the Repository:

bash
```python
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
Install Dependencies:
Make sure you have pip installed.

```python
pip install numpy pandas scikit-learn tensorflow joblib openpyxl
```
Usage
Running the Script
Update Dataset Path:
Replace the placeholder path in the script with the actual path to your dataset.

python
```python
df = pd.read_excel("/path/to/learning_dataset.xlsx")
```
Run the Script:

bash
Copia codice
python model_training.py

Providing Inputs
Categorical Variables:
The script will prompt you to select from a list.
Enter the number corresponding to your choice.
Numerical Features:
Enter the value when prompted.
Press Enter without typing anything to use the mean value.
Example Interaction
vbnet
Copia codice
Provide input for each feature (or press Enter to use mean value):

Select Industry from the list:
1. Technology
2. Healthcare
3. Finance
Enter the number corresponding to your choice: 1
Revenue: 5000000
Employees: 200
...

Predicted Normalized Valuation: 2.35

Do you want to make another prediction? (y/n): n
## Code Explanation
Imports and Environment Setup
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
Libraries Used:
numpy, pandas: For data manipulation.
scikit-learn: For preprocessing and model evaluation.
tensorflow.keras: For building the neural network.
joblib: For saving and loading the scaler and feature means.
Suppressing Warnings: Sets environment variables to reduce unnecessary warnings.
Data Loading and Preprocessing
```python
Load the dataset
df = pd.read_excel("/path/to/learning_dataset.xlsx")

# Separate features and target variable
X = df.drop('normalized_valuation', axis=1)
y = df['normalized_valuation']

# Identify categorical variables and their dummy columns
categorical_variables = {
    'Industry': [col for col in X.columns if col.startswith('Industry_')],
}

# Create mapping from feature to categorical variable
feature_to_category = {}
for category, dummy_columns in categorical_variables.items():
    for feature in dummy_columns:
        feature_to_category[feature] = category

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute mean values of each feature from X_train
feature_means = X_train.mean()
```
Loading the Dataset: Replace "/path/to/learning_dataset.xlsx" with the actual path to your dataset.
Separating Features and Target: X contains features, y contains the target variable.
Identifying Categorical Variables:
Automatically detects dummy variable columns for 'Industry'.
Creates a mapping (feature_to_category) for easy reference.
Data Splitting: Splits data into training and testing sets.
Scaling: Uses StandardScaler for feature scaling.
Feature Means: Calculates mean values for imputation.
Model Building and Training
python
Copia codice
def build_and_train_model(X_train_scaled, y_train):
    model = Sequential([
        Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=300,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    return model

# Train the model
model = build_and_train_model(X_train_scaled, y_train)
## Model Architecture: A sequential neural network with dropout layers to prevent overfitting.
Compilation: Uses Mean Squared Error as the loss function and Adam optimizer.
Early Stopping: Stops training when validation loss doesn't improve for 10 epochs.
Training: Fits the model to the scaled training data.
## Saving and Loading Models
```python
# Save the model
save_model(model, 'optimized_model.h5')

# Save the scaler and feature means using joblib
joblib.dump(scaler, 'scaler.save')
joblib.dump(feature_means, 'feature_means.save')

print("Model, scaler, and feature means have been saved.")

# Load the model, scaler, and feature means
model = load_model('optimized_model.h5')
scaler = joblib.load('scaler.save')
feature_means = joblib.load('feature_means.save')

# Load the column names (features)
feature_columns = feature_means.index.tolist()
Saving: Stores the trained model, scaler, and feature means for future use.
Loading: Retrieves the saved components when needed.
Feature Columns: Extracts the list of feature names.
Input Functions for Prediction
Preprocessing and Prediction Functions
python
Copia codice
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    return input_scaled

def predict(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction[0][0]
```
preprocess_input: Converts input data into a DataFrame and scales it using the loaded scaler.
predict: Uses the model to make a prediction on the preprocessed data.
User Input Function
```python
def input_for_prediction():
    input_data = {}
    print("\nProvide input for each feature (or press Enter to use mean value):")
    
    processed_categories = set()
    
    for feature in feature_columns:
        if feature in feature_to_category:
            category_name = feature_to_category[feature]
            if category_name not in processed_categories:
                # Present list of categories
                categories = categorical_variables[category_name]
                print(f"\nSelect {category_name} from the list:")
                for idx, cat_feature in enumerate(categories):
                    cat_name = cat_feature.replace(f"{category_name}_", "")
                    print(f"{idx+1}. {cat_name}")
                selection = input("Enter the number corresponding to your choice: ")
                try:
                    selection = int(selection)
                    if 1 <= selection <= len(categories):
                        selected_feature = categories[selection - 1]
                        # Set selected dummy variable to 1, others to 0
                        for cat_feature in categories:
                            input_data[cat_feature] = 1.0 if cat_feature == selected_feature else 0.0
                    else:
                        print("Invalid selection, using mean values.")
                        # Use mean values for all dummy variables
                        for cat_feature in categories:
                            input_data[cat_feature] = feature_means[cat_feature]
                except ValueError:
                    print("Invalid input, using mean values.")
                    # Use mean values for all dummy variables
                    for cat_feature in categories:
                        input_data[cat_feature] = feature_means[cat_feature]
                processed_categories.add(category_name)
            else:
                # Already processed this categorical variable
                continue
        else:
            user_input = input(f"{feature}: ")
            if user_input.strip() == "":
                input_data[feature] = feature_means[feature]
            else:
                input_data[feature] = float(user_input)
    return input_data
```
Categorical Variable Input:
Detects when a feature is part of a categorical variable.
Presents a list of options to the user.
Sets the selected dummy variable to 1.0 and others to 0.0.
Numerical Feature Input:
Prompts the user for each numerical feature.
Uses the mean value if the user presses Enter without input.
Error Handling:
Defaults to mean values for invalid inputs.
Main Execution Loop
```python
if __name__ == "__main__":
    while True:
        input_data = input_for_prediction()
        prediction = predict(input_data)
        print(f"\nPredicted Normalized Valuation: {prediction:.2f}\n")
        
        another = input("Do you want to make another prediction? (y/n): ")
        if another.lower() != 'y':
            break
```
Loop: Allows the user to make multiple predictions in one session.
Termination: Exits the loop when the user inputs anything other than 'y'.
Extending the Script
Adding More Categorical Variables
If your dataset includes other categorical variables represented by dummy variables, you can add them to the categorical_variables dictionary.

```python
categorical_variables = {
    'Industry': [col for col in X.columns if col.startswith('Industry_')],
    'Region': [col for col in X.columns if col.startswith('Region_')],
    # Add more categories as needed
}
```
Adjusting for Column Naming Conventions
Ensure that the prefixes used in startswith() match the actual column names in your dataset.

```python
# If your columns are like 'Sector_Technology', update accordingly
categorical_variables = {
    'Sector': [col for col in X.columns if col.startswith('Sector_')],
}
```
Error Handling
Invalid Categorical Selections:
If an invalid number is entered, the script defaults to using mean values for that category.
Non-numeric Inputs:
For numerical features, entering a non-numeric value will cause an error.
The script currently doesn't handle this exception; you can add error handling if needed.
Conclusion
By removing GUI components and implementing a terminal-based input system, the script becomes more accessible and easier to use, especially for users who prefer or require command-line interfaces. The streamlined process for entering categorical variables enhances user experience and reduces the likelihood of input errors.

License
This project is licensed under the MIT License.
