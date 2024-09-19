import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import joblib 
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('optimized_model.h5')

# Load the scaler and feature means
scaler = joblib.load('scaler.save')
feature_means = joblib.load('feature_means.save')

# Load the column names (features)
# Assuming the features are the same as those used during training
# If necessary, you can save and load the feature names as well
feature_columns = feature_means.index.tolist()

# Function to preprocess new input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Function to make predictions using the trained model
def predict(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction[0][0]

# Tkinter GUI Application
class PredictionApp:
    def __init__(self, root, feature_means):
        self.root = root
        self.root.title("Valuation Prediction")
        self.root.geometry("400x600")
        self.entries = {}
        self.feature_means = feature_means

        # Create input fields for each feature
        for idx, column in enumerate(feature_columns):
            label = tk.Label(root, text=column)
            label.grid(row=idx, column=0, padx=10, pady=5, sticky='e')
            entry = tk.Entry(root)
            entry.grid(row=idx, column=1, padx=10, pady=5)
            self.entries[column] = entry

        # Create Predict button
        self.predict_button = tk.Button(
            root, text="Predict", command=self.make_prediction
        )
        self.predict_button.grid(
            row=len(feature_columns), column=0, columnspan=2, pady=20
        )

    def make_prediction(self):
        input_data = {}
        try:
            for column, entry in self.entries.items():
                input_value = entry.get()
                if input_value == '':
                    # Use mean value for this feature if input is missing
                    mean_value = self.feature_means[column]
                    input_data[column] = mean_value
                else:
                    input_data[column] = float(input_value)

            prediction = predict(input_data)
            messagebox.showinfo(
                "Prediction", f"Predicted Normalized Valuation: {prediction:.2f}"
            )
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root, feature_means)
    root.mainloop()
