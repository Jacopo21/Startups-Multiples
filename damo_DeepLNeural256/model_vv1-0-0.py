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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
df = pd.read_excel("/Users/jacopobinati/Desktop/damodaran/learning_dataset.xlsx")

X = df.drop('normalized_valuation', axis=1)
y = df['normalized_valuation']

categorical_variables = {
    'Industry': [col for col in X.columns if col.startswith('Industry_')],
    'Region': [col for col in X.columns if col.startswith('Region_')]
}

#  mappin from feature to categorical variable
feature_to_category = {}
for category, dummy_columns in categorical_variables.items():
    for feature in dummy_columns:
        feature_to_category[feature] = category

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# get feature means, mins, and maxs
feature_means = X_train.mean()
feature_mins = X_train.min()
feature_maxs = X_train.max()

#  variables to be auto filled based on industry name
auto_fill_variables = [
    'Net Margin', 'Norm EV/Sales', 'Pre-tax Operating Margin', 'PBV', 'Norm ROE',
    'EV/ Invested Capital', 'ROIC', 'EV/EBITDAR&D', 'EV/EBITDA', 'EV/EBIT',
    'EV/EBIT (1-t)', 'EV/EBITDAR&D2', 'EV/EBITDA3', 'EV/EBIT4', 'EV/EBIT (1-t)5',
    'Norm % of Money Losing firms (Trailing)', 'Current PE', 'Trailing PE', 'Norm Forward PE',
    'Aggregate Mkt Cap/ Trailing Net Income (only money making firms)', 'Norm Expected growth 5 years'
]

# build and train the model
def build_and_train_model(X_train_scaled, y_train):
    model = Sequential([
        Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
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

# Train & save
model = build_and_train_model(X_train_scaled, y_train)
save_model(model, 'optimized_model.h5')
joblib.dump(scaler, 'scaler.save')
joblib.dump(feature_means, 'feature_means.save')

print("Model, scaler, and feature means have been saved.")

# Load the model, scaler, and feature means
model = load_model('optimized_model.h5')
scaler = joblib.load('scaler.save')
feature_means = joblib.load('feature_means.save')

# Load the column names (features)
feature_columns = feature_means.index.tolist()

# Reconstruct categorical variables and mappings
categorical_variables = {
    'Industry': [col for col in feature_columns if col.startswith('Industry_')],
    'Region': [col for col in feature_columns if col.startswith('Region_')],
}

# Create mapping from feature to categorical variable
feature_to_category = {}
for category, dummy_columns in categorical_variables.items():
    for feature in dummy_columns:
        feature_to_category[feature] = category

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

# Terminal input function for prediction
def input_for_prediction():
    input_data = {}
    print("\nProvide input for each feature (or press Enter to use mean value):")
    
    processed_categories = set()
    
    for feature in X.columns:
        if feature in feature_to_category:
            category_name = feature_to_category[feature]
            if category_name not in processed_categories:
                # Present list of categories
                categories = categorical_variables[category_name]
                print(f"\nSelect {category_name} from the list:")
                for idx, cat_feature in enumerate(categories):
                    cat_name = cat_feature.replace(f"{category_name}_", "")
                    print(f"{idx+1}. {cat_name} (1 for Yes, 0 for No)")
                selection = input("Enter the number corresponding to your choice: ")
                try:
                    selection = int(selection)
                    if 1 <= selection <= len(categories):
                        selected_feature = categories[selection - 1]
                        # Set selected dummy variable to 1, others to 0
                        for cat_feature in categories:
                            input_data[cat_feature] = 1.0 if cat_feature == selected_feature else 0.0
                        # Automatically fill values based on selected industry
                        industry_name = selected_feature.replace(f"{category_name}_", "")
                        industry_data = df[df[category_name] == industry_name]
                        for var in auto_fill_variables:
                            input_data[var] = industry_data[var].mean()
                    else:
                        print("Invalid selection, using mean values.")
                        for cat_feature in categories:
                            input_data[cat_feature] = feature_means[cat_feature]
                except ValueError:
                    print("Invalid input, using mean values.")
                    for cat_feature in categories:
                        input_data[cat_feature] = feature_means[cat_feature]
                processed_categories.add(category_name)
            else:
                continue
        else:
            min_val = feature_mins[feature]
            max_val = feature_maxs[feature]
            user_input = input(f"{feature} (min: {min_val}, max: {max_val}): ")
            if user_input.strip() == "":
                input_data[feature] = feature_means[feature]
            else:
                input_data[feature] = float(user_input)
    return input_data


# Main function to run in terminal
if __name__ == "__main__":
    while True:
        input_data = input_for_prediction()
        prediction = predict(input_data)
        print(f"\nPredicted Normalized Valuation: {prediction:.2f}\n")
        
        another = input("Do you want to make another prediction? (y/n): ")
        if another.lower() != 'y':
            break
