using CSV
using DataFrames
using Flux
using Flux.Optimise: ADAM
using Statistics
using JLD2
using Printf

# Load the dataset
df = CSV.read("/Users/jacopobinati/Desktop/damo_DeepLNeural256/learning_dataset.xlsx", DataFrame)

# Drop specified columns
drop!(df, [:EV_Sales, :ROE, :Expected_growth_next_5_years, :Forward_PE, :percent_of_Money_Losing_firms_Trailing])

# Split the data into features and target
X = select(df, Not(:normalized_valuation))
y = df.normalized_valuation

# Define categorical variables
categorical_variables = Dict(
    "Industry" => names(X, r"^Industry_"),
    "Region" => names(X, r"^Region_"),
    "Last Funding Type" => names(X, r"^LastFundingType__"),
    "All Funding Type" => names(X, r"^FundingType__"),
    "size of the company" => names(X, r"^NumberEmployees__")
)

# Mapping from feature to categorical variable
feature_to_category = Dict()
for (category, dummy_columns) in categorical_variables
    for feature in dummy_columns
        feature_to_category[feature] = category
    end
end

# Split the data into training and test sets
train_indices = sample(1:nrow(df), round(Int, 0.8 * nrow(df)), replace=false)
test_indices = setdiff(1:nrow(df), train_indices)

X_train = X[train_indices, :]
X_test = X[test_indices, :]
y_train = y[train_indices]
y_test = y[test_indices]

# Standardize the data
scaler = Flux.Data.DataLoader(X_train, batchsize=32, shuffle=true)
X_train_scaled = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
X_test_scaled = (X_test .- mean(X_train, dims=1)) ./ std(X_train, dims=1)

# Get feature means, mins, and maxs
feature_means = mean(X_train, dims=1)
feature_mins = minimum(X_train, dims=1)
feature_maxs = maximum(X_train, dims=1)

# Variables to be auto-filled based on industry name
auto_fill_variables = [
    "Net Margin", "Norm EV/Sales", "Pre-tax Operating Margin", "PBV", "Norm ROE",
    "EV/ Invested Capital", "ROIC", "EV/EBITDAR&D", "EV/EBITDA", "EV/EBIT",
    "EV/EBIT (1-t)", "EV/EBITDAR&D2", "EV/EBITDA3", "EV/EBIT4", "EV/EBIT (1-t)5",
    "Norm % of Money Losing firms (Trailing)", "Current PE", "Trailing PE", "Norm Forward PE",
    "Aggregate Mkt Cap/ Trailing Net Income (only money making firms)", "Norm Expected growth 5 years"
]

# Define and train the model
function build_and_train_enhanced_model(X_train, y_train; learning_rate=0.001, batch_size=32, epochs=500)
    model = Chain(
        Dense(size(X_train, 2), 512, relu),
        BatchNorm(512),
        Dropout(0.4),
        Dense(512, 256, relu),
        BatchNorm(256),
        Dropout(0.4),
        Dense(256, 128, relu),
        BatchNorm(128),
        Dropout(0.3),
        Dense(128, 64, relu),
        BatchNorm(64),
        Dropout(0.2),
        Dense(64, 32, relu),
        Dense(32, 1)
    )

    opt = ADAM(learning_rate)
    loss(x, y) = Flux.mse(model(x), y)
    ps = Flux.params(model)

    for epoch in 1:epochs
        for (x, y) in Flux.Data.DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
            gs = gradient(() -> loss(x, y), ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        println("Epoch $epoch: Loss = $(loss(X_train, y_train))")
    end

    return model
end

# Train the model
model = build_and_train_enhanced_model(X_train_scaled, y_train)

# Save the model, scaler, and feature means
@save "optimized_model.jld2" model
@save "scaler.jld2" feature_means feature_mins feature_maxs

# Load the model, scaler, and feature means
@load "optimized_model.jld2" model
@load "scaler.jld2" feature_means feature_mins feature_maxs

# Preprocess new input data
function preprocess_input(input_data)
    input_df = DataFrame(input_data)
    for col in names(X)
        if !(col in names(input_df))
            input_df[!, col] = 0
        end
    end
    input_df = input_df[:, names(X)]
    input_scaled = (input_df .- feature_means) ./ (feature_maxs .- feature_means)
    return input_scaled
end

# Make predictions using the trained model
function predict(input_data)
    preprocessed_data = preprocess_input(input_data)
    raw_prediction = model(preprocessed_data)[1]
    min_val = feature_mins[:normalized_valuation]
    max_val = feature_maxs[:normalized_valuation]
    denormalized_prediction = raw_prediction * (max_val - min_val) + min_val
    println("Normalized Predicted Valuation: $(raw_prediction)")
    println("Denormalized Predicted Valuation: \$(denormalized_prediction)")
    return denormalized_prediction
end

# Terminal input function for prediction
function input_for_prediction()
    input_data = Dict{String, Any}()
    println("\nProvide input for each feature (or press Enter to use mean value):")
    
    processed_categories = Set()
    selected_industries = []

    for feature in names(X)
        if feature in auto_fill_variables || feature == "Total Funding Amount (in USD)"
            continue
        end
        
        if feature in keys(feature_to_category)
            category_name = feature_to_category[feature]
            if !(category_name in processed_categories)
                categories = categorical_variables[category_name]
                println("\nSelect $category_name(s) from the list (you can select multiple):")
                for (idx, cat_feature) in enumerate(categories)
                    cat_name = replace(cat_feature, "$category_name_" => "")
                    println("$idx. $cat_name")
                end
                
                selections = split(chomp(readline()), ",")
                selected_features = []
                
                try
                    for selection in selections
                        selection = parse(Int, strip(selection))
                        if 1 <= selection <= length(categories)
                            push!(selected_features, categories[selection])
                        else
                            println("Invalid selection $selection, ignoring.")
                        end
                    end
                    
                    for cat_feature in categories
                        input_data[cat_feature] = cat_feature in selected_features ? 1.0 : 0.0
                    end
                    
                    if category_name == "Industry"
                        selected_industries = [replace(feature, "Industry_" => "") for feature in selected_features]
                    end
                catch e
                    println("Invalid input, using mean values.")
                    for cat_feature in categories
                        input_data[cat_feature] = feature_means[cat_feature]
                    end
                end
                
                push!(processed_categories, category_name)
            end
        else
            if feature == "Norm Total Funding"
                min_val = feature_mins["Total Funding Amount (in USD)"]
                max_val = feature_maxs["Total Funding Amount (in USD)"]
                user_input = readline()
                if strip(user_input) == ""
                    funding_amount = feature_means["Last Funding Amount (in USD)"] +
                                     feature_means["Last Equity Funding Amount (in USD)"] +
                                     feature_means["Total Equity Funding Amount (in USD)"]
                    println("Total funding amount (in USD) left empty, using sum of funding amounts: $funding_amount")
                else
                    funding_amount = parse(Float64, user_input)
                end
                
                norm_funding = (funding_amount - min_val) / (max_val - min_val)
                input_data["Norm Total Funding"] = norm_funding
                println("Computed Norm Total Funding: $norm_funding")
            else
                min_val = feature_mins[feature]
                max_val = feature_maxs[feature]
                user_input = readline()
                if strip(user_input) == ""
                    input_data[feature] = feature_means[feature]
                else
                    input_data[feature] = parse(Float64, user_input)
                end
            end
        end
    end
    
    input_data["Status__Private"] = 1
    input_data["CompanyType__For Profit"] = 1
    input_data["CompanyType__Non-profit"] = 0
    
    if "Industry" in names(df) && !isempty(selected_industries)
        industry_data = filter(row -> row.Industry in selected_industries, df)
        
        if !isempty(industry_data)
            for var in auto_fill_variables
                if var in names(industry_data)
                    if any(!ismissing.(industry_data[!, var]))
                        input_data[var] = mean(skipmissing(industry_data[!, var]))
                        println("Using mean value for '$var' from selected industries.")
                    else
                        input_data[var] = feature_means[var]
                        println("No data for '$var' in selected industries. Using global mean.")
                    end
                else
                    input_data[var] = feature_means[var]
                    println("Feature '$var' not found in industry data. Using global mean.")
                end
            end
        else
            println("Warning: No data available for the selected industries. Using global mean for all auto-fill variables.")
            for var in auto_fill_variables
                input_data[var] = feature_means[var]
            end
        end
    else
        println("No industries selected. Using global mean for all auto-fill variables.")
        for var in auto_fill_variables
            input_data[var] = feature_means[var]
        end
    end
    
    println("Number of features in input_data: $(length(input_data))")
    println("Features in input_data: $(keys(input_data))")

    for feature in names(X)
        if !(feature in keys(input_data))
            println("Warning: Feature '$feature' is missing. Using mean value.")
            input_data[feature] = feature_means[feature]
        end
    end

    if length(input_data) != length(names(X))
        println("Warning: Number of features mismatch. Expected $(length(names(X))), got $(length(input_data)).")
        println("Missing features: $(setdiff(names(X), keys(input_data)))")
    end

    return input_data
end

# Main function to run in terminal
function main()
    while true
        input_data = input_for_prediction()
        prediction = predict(input_data)
        println("\nPredicted Normalized Valuation: $prediction\n")
        
        println("Do you want to make another prediction? (y/n): ")
        another = readline()
        if lowercase(strip(another)) != "y"
            break
        end
    end
end

main()