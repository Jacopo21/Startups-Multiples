import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("/Users/jacopobinati/Desktop/damo_DeepLNeural256/raw files/multiples/multiples_combined.csv")
dataset = dataset[dataset["Operating Status"] != "Closed"]
dataset["Last Equity Funding Amount (in USD)"] = dataset["Last Equity Funding Amount (in USD)"].fillna(0)
dataset["Last Funding Amount (in USD)"] = dataset["Last Funding Amount (in USD)"].fillna(0)
dataset["Total Funding Amount (in USD)"] = dataset["Total Funding Amount (in USD)"].fillna(0)
dataset["Total Equity Funding Amount (in USD)"] = dataset["Total Equity Funding Amount (in USD)"].fillna(0)
dataset["Company Type"] = dataset["Company Type"].fillna("For Profit")

columns_todrop = [
    "Growth Confidence",
    "Actively Hiring",
    "Investment Stage",
    "Funding Status",
    "Most Recent Valuation Range",
    "Number of Investments",
    "Last Funding Amount Currency",
    "Investor Type",
    "Acquisition Status",
    "Growth Category",
    "Organization Name URL"
]
dataset = dataset.drop(columns=columns_todrop)
dataset = dataset.dropna(subset=["Number of Founders"])
dataset = dataset.dropna(subset=["Top 5 Investors"])
dataset = dataset.dropna(subset=["Number of Investors"])
dataset = dataset.dropna(subset=["Headquarters Regions"])
dataset = dataset.dropna(subset=["Number of Employees"])
dataset = dataset.dropna(subset=["Industry Groups"])
dataset = dataset.dropna(subset=["Founded Date"])
dataset = dataset.dropna(subset=["Last Funding Date"])
dataset = dataset.dropna(subset=["Last Funding Type"])

dataset["Number of Investors"].astype(int)
dataset["Number of Founders"].astype(int)
dataset["Last Funding Year"] = dataset["Last Funding Date"].str[:4]
dataset["Founded Year"] = dataset["Founded Date"].str[:4]

dataset = pd.DataFrame(dataset)
dataset['Industries'] = dataset['Industries'].astype(str)
def extract_main_industry(industries):
    return industries.split(',')[0].strip()
    
dataset['Main Industry'] = dataset['Industries'].apply(extract_main_industry)

industry_mapping_reversed = {
    "Aerospace/Defense": ["Aerospace", "Classifieds", "Government", "GovTech"],
    "Air Transport": ["Air Transportation", "Drones"],
    "Apparel": ["Apparel", "Fashion"],
    "Auto & Truck": ["Automotive", "Electric Vehicle"],
    "Auto Parts": ["Battery"],
    "Bank (Money Center)": ["Banking", "Bitcoin"],
    "Banks (Regional)": ["Commercial Lending", "Consumer Lending", "Lending"],
    "Beverage (Alcoholic)": [],
    "Beverage (Soft)": ["Coffee"],
    "Broadcasting": [],
    "Brokerage & Investment Banking": ["Angel Investment", "Crowdfunding", "Crowdsourcing"],
    "Building Materials": ["Building Material", "Construction", "Architecture"],
    "Business & Consumer Services": ["Consulting", "Professional Services", "Leasing", "B2B", "Business Development", "Enterprise Resource Planning (ERP)", "B2C", "Business Travel",
                                     "CRM", "Event Management", "Business Process Automation (BPA)", "Bookkeeping and Payroll", "Employment", "Billing", "Enterprise Software",
                                     "Technical Support", "Enterprise Applications", "Lead Generation", "Productivity Tools", "Sales Automation", "Accounting", "Commercial",
                                     "Service Industry"],
    "Cable TV": [],
    "Chemical (Basic)": ["Chemical"],
    "Chemical (Diversified)": ["Chemical Engineering"],
    "Chemical (Specialty)": ["Advanced Materials"],
    "Coal & Related Energy": ["Nuclear"],
    "Computer Services": ["Information Technology", "Cloud Computing", "Cloud Data Services", "Cloud Management", "Cloud Infrastructure", "Data Center", "SaaS", "Graphic Design"],
    "Computers/Peripherals": ["Computer", "Hardware", "Robotics"],
    "Construction Supplies": [],
    "Diversified": ["Consumer Goods", "Consumer Applications", "Consumer Software", "Consumer"],
    "Drugs (Biotechnology)": ["Biotechnology", "Biopharma", "Alternative Medicine"],
    "Drugs (Pharmaceutical)": ["Health Care", "Emergency Medicine", "Addiction Treatment", "Dental", "Beauty"],
    "Education": ["EdTech", "Corporate Training", "E-Learning", "Education"],
    "Electrical Equipment": ["Electrical Distribution"],
    "Electronics (Consumer & Office)": ["Consumer Electronics"],
    "Electronics (General)": ["Electronics", "Embedded Systems", "Embedded Software"],
    "Engineering/Construction": ["Industrial Engineering", "Civil Engineering", "Industrial Automation"],
    "Entertainment": ["Gaming", "Digital Entertainment", "Content", "Fantasy Sports", "eSports", "Communities"],
    "Environmental & Waste Services": ["CleanTech", "Recycling", "Environmental Consulting"],
    "Farming/Agriculture": ["Agriculture", "AgTech", "Dairy", "Animal Feed", "Pet"],
    "Financial Svcs. (Non-bank & Insurance)": ["FinTech", "Insurance", "Auto Insurance", "Health Insurance", "Credit Bureau", "Fraud Detection", "Compliance", "Advice", 
                                               "Decentralized Finance (DeFi)", "Cryptocurrency", "Finance"],
    "Food Processing": ["Alternative Protein", "Dietary Supplements", "Cooking", "Bakery"],
    "Food Wholesalers": ["Food and Beverage"],
    "Furn/Home Furnishings": [],
    "Green & Renewable Energy": ["GreenTech", "Clean Energy", "Energy Efficiency", "Carbon Capture"],
    "Healthcare Products": ["Health Care", "Alternative Medicine"],
    "Healthcare Support Services": ["Assisted Living", "Clinical Trials", "Baby", "Child Care", "Fitness", "Sports"],
    "Heathcare Information and Technology": ["Biometrics", "Human Resources", "Employee Benefits"],
    "Homebuilding": ["Property Development", "Real Estate (Development)"],
    "Hospitals/Healthcare Facilities": ["Hospitals/Healthcare Facilities", "Dental"],
    "Hotel/Gaming": ["Hospitality", "Adventure Travel", "Events"],
    "Household Products": ["Home Services"],
    "Information Services": ["Business Information Systems", "Information Services", "Analytics", "Big Data", "Market Research", "Business Intelligence", "Database"],
    "Insurance (General)": ["Insurance", "Credit", "Legal"],
    "Insurance (Life)": [],
    "Insurance (Prop/Cas.)": ["Insurance", "Auto Insurance"],
    "Investments & Asset Management": ["Asset Management", "Investments & Asset Management", "Credit Cards", "Financial Services", "PropTech"],
    "Machinery": ["Manufacturing", "3D Printing", "Industrial Manufacturing", "3D Technology"],
    "Metals & Mining": [],
    "Office Equipment & Services": [],
    "Oil/Gas (Integrated)": [],
    "Oil/Gas (Production and Exploration)": [],
    "Oil/Gas Distribution": [],
    "Oilfield Svcs/Equip.": [],
    "Packaging & Container": [],
    "Paper/Forest Products": [],
    "Power": ["Energy"],
    "Precious Metals": ["Advanced Materials"],
    "Publishing & Newspapers": ["Blogging Platforms", "Copywriting"],
    "R.E.I.T.": ["Real Estate (REITs)"],
    "Real Estate (Development)": ["Real Estate (Development)", "Property Development", ],
    "Real Estate (General/Diversified)": ["Commercial Real Estate"],
    "Real Estate (Operations & Services)": ["Property Management", "Real Estate (Operations & Services)"],
    "Recreation": ["Recreation", "Adventure Travel"],
    "Reinsurance": [],
    "Restaurant/Dining": ["Restaurants", "Food Delivery"],
    "Retail (Automotive)": ["Retail"],
    "Retail (Building Supply)": [],
    "Retail (Distributors)": ["Retail (General)", "Retail"],
    "Retail (General)": ["Retail", "E-Commerce Platforms", "E-Commerce", "Subscription Service"],
    "Retail (Grocery and Food)": ["Food and Beverage"],
    "Retail (REITs)": [],
    "Retail (Special Lines)": ["Fashion"],
    "Rubber & Tires": [],
    "Semiconductor": ["Semiconductor"],
    "Semiconductor Equip": ["Semiconductor Equip"],
    "Shipbuilding & Marine": [],
    "Shoe": [],
    "Software (Entertainment)": ["Ad Server", "Augmented Reality"],
    "Software (Internet)": ["Cloud Security", "Developer APIs", "Cloud Computing", "Software", "SaaS", "Mobile Apps", "PropTech", "Cloud Security"],
    "Software (System & Application)": ["Generative AI", "Artificial Intelligence (AI)", "Developer Platform", "Blockchain", "Machine Learning", "Apps", "CMS", "Data Storage",
                                        "Cloud Storage", "Internet of Things", "Internet of Things (IoT)"],
    "Steel": [],
    "Telecom (Wireless)": [],
    "Telecom. Equipment": ["Cyber Security", "Network Security", "Telecommunications", "Fleet Management", "GPS"],
    "Telecom. Services": ["Messaging", "Internet", "Telecommunications", "Satellite Communication", "Advertising", "Collaboration", "Content Delivery Network", 
                          "Communications Infrastructure", "Digital Marketing", "Ad Exchange", "Digital Media"],
    "Tobacco": [],
    "Transportation": ["Freight Service", "Delivery", "Logistics", "Delivery Service"],
    "Transportation (Railroads)": ["Infrastructure"],
    "Trucking": [],
    "Utility (General)": [],
    "Utility (Water)": [],
}

def allocate_sector(main_sector, industry_mapping_reversed):
    if main_sector == 'Aerospace':
        return 'Aerospace/Defense'
    
    for sector, industries in industry_mapping_reversed.items():
        if main_sector in industries:
            return sector
    return None

dataset['Industry Name'] = dataset['Main Industry'].apply(lambda x: allocate_sector(x, industry_mapping_reversed))
dataset = dataset.dropna(subset=['Industry Name'])
dataset['Industry Name'].isna().sum()

dataset["IPO Status"].unique()
ipo_dummies = pd.get_dummies(dataset["IPO Status"], prefix="Status_")
dataset = pd.concat([dataset, ipo_dummies], axis=1)

status_columns = [col for col in dataset.columns if col.startswith("Status_")]
#growth_columns = [col for col in dataset.columns if col.startswith("GrowthCategory_")]
company = status_columns #+ growth_columns

company_type_dummies = pd.get_dummies(dataset["Company Type"], prefix="CompanyType_")
dataset = pd.concat([dataset, company_type_dummies], axis=1)
dataset.head()

dataset["Number of Employees"].hist()
N_employees_dummies = pd.get_dummies(dataset["Number of Employees"], prefix="NumberEmployees_")
dataset = pd.concat([dataset, N_employees_dummies], axis=1)
dataset.head()

dataset['Last Funding Year'] = dataset['Last Funding Date'].str[:4].astype(int)
dataset = dataset[dataset["Last Funding Year"] > 1950]

dataset['Last Funding Type'].unique(), dataset['Last Funding Date'].describe()
Funding_type_dummies = pd.get_dummies(dataset["Last Funding Type"], prefix="FundingType_")
dataset = pd.concat([dataset, Funding_type_dummies], axis=1)

industry_dummies = dataset["Industry Name"].str.get_dummies().add_prefix('Industry_')
dataset = pd.concat([dataset, industry_dummies], axis=1)
industries = dataset.columns[dataset.columns.str.startswith('Industry_')]

unique_headquarters = dataset["Headquarters Regions"].unique()
unique_headquarters_df = pd.DataFrame(unique_headquarters, columns=["Headquarters Location"])

data = {
    'Headquarters Location': [
        'San Francisco Bay Area, Silicon Valley, West Coast',
        'Greater Boston Area, East Coast, New England',
        'San Francisco Bay Area, West Coast, Western US',
        'Greater New York Area, East Coast, Northeastern US',
        'Asia-Pacific (APAC), Southeast Asia',
        'Greater Seattle Area, West Coast, Western US',
        'Europe, Middle East, and Africa (EMEA)',
        'Greater Denver Area, Western US',
        'Asia-Pacific (APAC)',
        'Greater Los Angeles Area, West Coast, Western US',
        'Southern US',
        'Greater Atlanta Area, East Coast, Southern US',
        'Greater Chicago Area, Great Lakes, Midwestern US',
        'Europe, Middle East, and Africa (EMEA), Middle East, MENA',
        'European Union (EU), Europe, Middle East, and Africa (EMEA)',
        'Research Triangle, East Coast, Southern US',
        'Greater Phoenix Area, Western US',
        'Washington DC Metro Area, East Coast, Southern US',
        'Great Lakes',
        'Greater Baltimore-Maryland Area, East Coast, Southern US',
        'Asia-Pacific (APAC), Western US',
        'Great Lakes, Northeastern US',
        'Europe, Middle East, and Africa (EMEA), GCC, Middle East',
        'European Union (EU), Nordic Countries, Scandinavia',
        'Tampa Bay Area, East Coast, Southern US',
        'Great Lakes, Midwestern US',
        'East Coast, Southern US',
        'Washington DC Metro Area, Southern US',
        'Latin America',
        'Western US',
        'Greater San Diego Area, West Coast, Western US',
        'Greater Detroit Area, Great Lakes, Midwestern US',
        'Greater Philadelphia Area, East Coast, Southern US',
        'West Coast, Western US',
        'Great Lakes, East Coast, Northeastern US',
        'Greater Miami Area, East Coast, Southern US',
        'Greater Philadelphia Area, Great Lakes, Northeastern US',
        'Dallas/Fort Worth Metroplex, Southern US',
        'Midwestern US',
        'Europe, Middle East, and Africa (EMEA), European Union (EU), Middle East',
        'East Coast, New England, Northeastern US',
        'Greater Houston Area, Southern US',
        'Asia-Pacific (APAC), Australasia',
        'Nordic Countries, Scandinavia, Europe, Middle East, and Africa (EMEA)',
        'Asia-Pacific (APAC), Middle East and North Africa (MENA)',
        'East Coast, Northeastern US',
        'Greater Minneapolis-Saint Paul Area, Great Lakes, Midwestern US',
        'New England, Northeastern US',
        'Middle East and North Africa (MENA)',
        'Central America, Latin America',
        'Greater Los Angeles Area, Inland Empire, West Coast',
        'Greater Philadelphia Area, East Coast, Northeastern US',
        'European Union (EU), Middle East and North Africa (MENA), EMEA',
        'Middle East and North Africa (MENA), North Africa, EMEA'
    ]
}

dataset1 = pd.DataFrame(data)

def map_to_macro_region(region):
    # North America
    if any(x in region for x in ['West Coast', 'Western US', 'East Coast', 'Great Lakes', 'Midwestern US', 'Southern US', 'Northeastern US']):
        return 'North America'
    
    # Asia-Pacific (APAC)
    elif any(x in region for x in ['Asia-Pacific', 'APAC', 'Southeast Asia', 'Australasia']):
        return 'Asia-Pacific'
    
    # Europe, Middle East, and Africa (EMEA)
    elif any(x in region for x in ['Europe', 'Middle East', 'Africa', 'EMEA', 'European Union', 'EU', 'Scandinavia', 'Nordic', 'GCC', 'MENA', 'North Africa']):
        return 'EMEA'
    
    # Latin America
    elif any(x in region for x in ['Latin America', 'Central America']):
        return 'Latin America'
    
    # Default to 'Other' if no matches found
    else:
        return 'Other'

# Apply the mapping function
dataset['Macro Region'] = dataset['Headquarters Regions'].apply(map_to_macro_region)
dataset["Macro Region"] = dataset["Macro Region"].replace("Latin America", "Emerging Markets")
macro_region_dummies = pd.get_dummies(dataset["Macro Region"], prefix="Region")
dataset = pd.concat([dataset, macro_region_dummies], axis=1)
unique_headquarters = dataset["Macro Region"].unique()
unique_headquarters_df = pd.DataFrame(unique_headquarters, columns=["Headquarters Regions"])
unique_headquarters_df

# FINANCIAL VARIABLES
dataset["Last Funding Type"].unique()
last_equity_funding_type_dummies = pd.get_dummies(dataset["Last Funding Type"], prefix="LastFundingType_")
dataset = pd.concat([dataset, last_equity_funding_type_dummies], axis=1)

# MERGING WITH DAMODARAN DATA
damodaran_MACRO = pd.read_excel("/Users/jacopobinati/Desktop/damo_DeepLNeural256/dataset/AGGREGATE_DAMO.xlsx")
analysis2 = pd.merge(dataset, damodaran_MACRO, on=["Industry Name", "Macro Region"], how="left")

# NORMALIZED VARIABLES AND NORMALIZED VALUATION
columns_to_normalize = [
    "EV/Sales",
    "ROE",
    "Expected growth - next 5 years",
    "Forward PE",
    "% of Money Losing firms (Trailing)"
]

scaler = MinMaxScaler()

analysis2["Norm Total Funding"] = scaler.fit_transform(analysis2[["Total Funding Amount (in USD)"]])
analysis2["Norm EV/Sales"] = scaler.fit_transform(analysis2[["EV/Sales"]])
analysis2["Norm ROE"] = scaler.fit_transform(analysis2[["ROE"]])
analysis2["Norm Expected growth 5 years"] = scaler.fit_transform(analysis2[["Expected growth - next 5 years"]])
analysis2["Norm Forward PE"] = scaler.fit_transform(analysis2[["Forward PE"]])
analysis2["Norm % of Money Losing firms (Trailing)"] = scaler.fit_transform(analysis2[["% of Money Losing firms (Trailing)"]])

    # Define weights for the normalized columns
weights = {
    "Norm EV/Sales": 0.2,
    "Norm ROE": 0.2,
    "Norm Expected growth 5 years": 0.25,
    "Norm Forward PE": 0.1,
    "Norm % of Money Losing firms (Trailing)": 0.1,
    "Norm Total Funding": 0.15
}

# Calculate the weighted sum for normalized_valuation
analysis2["normalized_valuation"] = sum(analysis2[col] * weight for col, weight in weights.items())

analysis2["normalized_valuation"] = scaler.fit_transform(analysis2[["normalized_valuation"]])

# EXPORTING THE DATASET
analysis2.to_csv("/Users/jacopobinati/Desktop/damo_DeepLNeural256/dataset/dataset_with_MACRO 3.csv", index=False)