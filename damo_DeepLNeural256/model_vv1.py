import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
df = pd.read_excel("/Users/jacopobinati/Desktop/damo_DeepLNeural256/dataset/dataset_with_MACRO 2.xlsx")
df.drop(columns=[
    "EV/Sales",
    "ROE",
    "Expected growth - next 5 years",
    "Forward PE",
    "% of Money Losing firms (Trailing)"
], inplace=True)

X = df.drop('normalized_valuation', axis=1)

y = df['normalized_valuation']

categorical_variables = {
    'Industry': [col for col in X.columns if col.startswith('Industry_')],
    'Region': [col for col in X.columns if col.startswith('Region_')],
    'Last Funding Type': [col for col in X.columns if col.startswith('LastFundingType__')],
    'All Funding Type': [col for col in X.columns if col.startswith('FundingType__')],
    'size of the company': [col for col in X.columns if col.startswith('NumberEmployees__')]
}

#  mappin from feature to categorical variable - otherwise big problema
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

# get vars means, mins, and maxs
feature_means = X_train.mean()
feature_mins = X_train.min()
feature_maxs = X_train.max()

#  variables to be auto filled based on industry name - STILL YOU NEED TO WORK 
auto_fill_variables = [
    'Net Margin', 'Norm EV/Sales', 'Pre-tax Operating Margin', 'PBV', 'Norm ROE',
    'EV/ Invested Capital', 'ROIC', 'EV/EBITDAR&D', 'EV/EBITDA', 'EV/EBIT',
    'EV/EBIT (1-t)', 'EV/EBITDAR&D2', 'EV/EBITDA3', 'EV/EBIT4', 'EV/EBIT (1-t)5',
    'Norm % of Money Losing firms (Trailing)', 'Current PE', 'Trailing PE', 'Norm Forward PE',
    'Aggregate Mkt Cap/ Trailing Net Income (only money making firms)', 'Norm Expected growth 5 years'
]

# build and train the model - THIS IS ACTUALLY REALLY NICE! HAPPY
def build_and_train_enhanced_model(X_train, y_train, learning_rate=0.001, batch_size=32, epochs=500):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = Sequential([
        Dense(512, input_dim=X_train.shape[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history, scaler

# train & save
model, history, scaler = build_and_train_enhanced_model(X_train_scaled, y_train)

# update  saving code:
save_model(model, 'optimized_model.h5')
joblib.dump(scaler, 'scaler.save')
joblib.dump(feature_means, 'feature_means.save')
print("Model, scaler, and feature means have been saved.")

# load the model, scaler, and feature means
model = load_model('optimized_model.h5')
scaler = joblib.load('scaler.save')
feature_means = joblib.load('feature_means.save')

# Load the column names (features)
feature_columns = feature_means.index.tolist()

# Reconstruct categorical variables and mappings
categorical_variables = {
    'Industry': [col for col in feature_columns if col.startswith('Industry_')],
    'Region': [col for col in feature_columns if col.startswith('Region_')],
    'Last Funding Type': [col for col in feature_columns if col.startswith('LastFundingType__')],
    'All Funding Type': [col for col in feature_columns if col.startswith('FundingType__')],
    'size of the company': [col for col in feature_columns if col.startswith('NumberEmployees__')]
}

# here create mapping from feature to categorical variable
feature_to_category = {}
for category, dummy_columns in categorical_variables.items():
    for feature in dummy_columns:
        feature_to_category[feature] = category

# preprocess new input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    for col in X.columns:
        if col not in input_df.columns:
            if col in feature_means:
                input_df[col] = feature_means[col]
            else:
                input_df[col] = 0  
    
    input_df = input_df[X.columns] # reorder columns to match the training data structure
    input_scaled = scaler.transform(input_df)  # scale the input data using the pre-fitted scaler
    
    return input_scaled

# make predictions using the trained model
def predict(input_data):
    # handle missing Total Funding Amount
    if 'Total Funding Amount (in USD)' not in input_data:
        print("Warning: Feature 'Total Funding Amount (in USD)' is missing. Using mean value.")
        input_data['Total Funding Amount (in USD)'] = feature_means['Total Funding Amount (in USD)']
    
    # 
    preprocessed_data = preprocess_input(input_data)
    
    # 
    raw_prediction = model.predict(preprocessed_data)[0][0]
    
    # 
    if 'normalized_valuation' in feature_mins and 'normalized_valuation' in feature_maxs:
        min_val = feature_mins['normalized_valuation']
        max_val = feature_maxs['normalized_valuation']
        
        # 
        denormalized_prediction = raw_prediction * (max_val - min_val) + min_val
        
        print(f"Normalized Predicted Valuation: {raw_prediction:.6f}")
        print(f"Denormalized Predicted Valuation: ${denormalized_prediction:,.2f}")
    else:
        # Fallback behavior in case 'normalized_valuation' is missing
        print("Warning: 'normalized_valuation' not found in feature_mins or feature_maxs. Using raw prediction.")
        denormalized_prediction = raw_prediction
        
        print(f"Raw Predicted Valuation (no denormalization applied): {raw_prediction:.6f}")
    
    return denormalized_prediction


# Terminal input function for prediction
def input_for_prediction():
    input_data = {}
    print("\nProvide input for each feature (or press Enter to use mean value):")     
    
    processed_categories = set()
    selected_industries = []

    for feature in X.columns:
        if feature in auto_fill_variables or feature == 'Total Funding Amount (in USD)':
            continue
        
        if feature in feature_to_category:
            category_name = feature_to_category[feature]
            if category_name not in processed_categories:
                categories = categorical_variables[category_name]
                print(f"\nSelect {category_name}(s) from the list (you can select multiple):")
                for idx, cat_feature in enumerate(categories):
                    cat_name = cat_feature.replace(f"{category_name}_", "")
                    print(f"{idx+1}. {cat_name}")
                
                selections = input("Enter the numbers corresponding to your choices (comma-separated): ").split(',')
                selected_features = []
                
                try:
                    for selection in selections:
                        selection = int(selection.strip())
                        if 1 <= selection <= len(categories):
                            selected_features.append(categories[selection - 1])
                        else:
                            print(f"Invalid selection {selection}, ignoring.")
                    
                    for cat_feature in categories:
                        input_data[cat_feature] = 1.0 if cat_feature in selected_features else 0.0
                    
                    if category_name == 'Industry':
                        selected_industries = [feature.replace("Industry_", "") for feature in selected_features]
                    
                except ValueError:
                    print("Invalid input, using mean values.")
                    for cat_feature in categories:
                        input_data[cat_feature] = feature_means[cat_feature]
                
                processed_categories.add(category_name)
            else:
                continue    
        else:
            if feature == 'Norm Total Funding':
                min_val = feature_mins['Total Funding Amount (in USD)']
                max_val = feature_maxs['Total Funding Amount (in USD)']
                user_input = input(f"Total Funding Amount (in USD) (min: {min_val}, max: {max_val}): ")
                if user_input.strip() == "":
                    funding_amount = (
                        feature_means['Last Funding Amount (in USD)'] +
                        feature_means['Last Equity Funding Amount (in USD)'] +
                        feature_means['Total Equity Funding Amount (in USD)']
                    )
                    print(f"Total funding amount (in USD) left empty, using sum of funding amounts: {funding_amount}")
                else:
                    funding_amount = float(user_input)
                
                norm_funding = (funding_amount - min_val) / (max_val - min_val)
                input_data['Norm Total Funding'] = norm_funding
                print(f"Computed Norm Total Funding: {norm_funding:.6f}")
            else:
                min_val = feature_mins[feature]
                max_val = feature_maxs[feature]
                user_input = input(f"{feature} (min: {min_val}, max: {max_val}): ")
                if user_input.strip() == "":
                    input_data[feature] = feature_means[feature]
                else:
                    input_data[feature] = float(user_input)
    
    
    if 'Industry' in df.columns and selected_industries:
        industry_data = df[df['Industry'].isin(selected_industries)]
        
        if not industry_data.empty:
            for var in auto_fill_variables:
                if var in industry_data.columns:
                    if industry_data[var].notna().any():
                        input_data[var] = industry_data[var].mean()
                        print(f"Using mean value for '{var}' from selected industries.")
                    else:
                        input_data[var] = feature_means[var]
                        print(f"No data for '{var}' in selected industries. Using global mean.")
                else:
                    input_data[var] = feature_means[var]
                    print(f"Feature '{var}' not found in industry data. Using global mean.")
        else:
            print("Warning: No data available for the selected industries. Using global mean for all auto-fill variables.")
            for var in auto_fill_variables:
                input_data[var] = feature_means[var]
    else:
        print("No industries selected. Using global mean for all auto-fill variables.")
        for var in auto_fill_variables:
            input_data[var] = feature_means[var]

    # fetching Price/Sales based on the selected industry!! THIS IS PROBLEMATIC AAAAAAAAA
    # I FOUND YOU and I WILL FIX YOU
    if selected_industries:
        industry_column = f'Industry_{selected_industries[0]}'
        price_sales_value = df[df[industry_column] == 1]['Price/Sales'].mean()
        input_data['Price/Sales'] = price_sales_value
        print(f"Using Price/Sales value for {selected_industries[0]}: {price_sales_value}")

    print(f"Number of features in input_data: {len(input_data)}")
    print("Features in input_data:", list(input_data.keys()))

    for feature in X.columns:
        if feature not in input_data:
            print(f"Warning: Feature '{feature}' is missing. Using mean value.")
            input_data[feature] = feature_means.get(feature, 0)

    # is feature mismatch
    if len(input_data) != len(X.columns):
        print(f"Warning: Number of features mismatch. Expected {len(X.columns)}, got {len(input_data)}.")
        print("Missing features:", set(X.columns) - set(input_data.keys()))

    input_data['Norm Total Funding'] = joblib.load('scaler_total_funding.pkl').transform([[input_data['Total Funding Amount (in USD)']]])[0][0]
    input_data['Norm EV/Sales'] = joblib.load('scaler_ev_sales.pkl').transform([[input_data['EV/Sales']]])[0][0]
    input_data['Norm ROE'] = joblib.load('scaler_roe.pkl').transform([[input_data['ROE']]])[0][0]
    input_data['Norm Expected growth 5 years'] = joblib.load('scaler_expected_growth.pkl').transform([[input_data['Expected growth - next 5 years']]])[0][0]
    input_data['Norm Forward PE'] = joblib.load('scaler_forward_pe.pkl').transform([[input_data['Forward PE']]])[0][0]
    input_data['Norm % of Money Losing firms (Trailing)'] = joblib.load('scaler_money_losing.pkl').transform([[input_data['% of Money Losing firms (Trailing)']]])[0][0]

    weights = {
        "Norm EV/Sales": 0.2,
        "Norm ROE": 0.2,
        "Norm Expected growth 5 years": 0.25,
        "Norm Forward PE": 0.1,
        "Norm % of Money Losing firms (Trailing)": 0.1,
        "Norm Total Funding": 0.15
    }
    normalized_valuation = sum(input_data[col] * weight for col, weight in weights.items())

    normalized_valuation = joblib.load('scaler_normalized_valuation.pkl').transform([[normalized_valuation]])[0][0]

    return input_data, normalized_valuation

#  run in terminal pls - better Warp. let's be serious right?
if __name__ == "__main__":
    while True:
        input_data, normalized_valuation = input_for_prediction()
        print(f"\nPredicted Normalized Valuation: {normalized_valuation:.2f}\n")
        
        another = input("Do you want to make another prediction? (y/n): ")
        if another.lower() != 'y':
            break







