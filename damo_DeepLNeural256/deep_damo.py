import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib  # For saving the scaler and feature means

# Load the dataset
df = pd.read_excel("/Users/jacopobinati/Desktop/damodaran/learning_dataset.xlsx")

# Separate features and target variable
X = df.drop('normalized_valuation', axis=1)
y = df['normalized_valuation']

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

# Function to build and train the model
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

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    return model

# Train the model
model = build_and_train_model(X_train_scaled, y_train)

# Save the model
save_model(model, 'optimized_model.h5')

# Save the scaler and feature means using joblib
joblib.dump(scaler, 'scaler.save')
joblib.dump(feature_means, 'feature_means.save')

print("Model, scaler, and feature means have been saved.")