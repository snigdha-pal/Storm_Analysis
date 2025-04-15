import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm 

# Reload the data
try:
    df = pd.read_csv('storm_event_data.csv')
except FileNotFoundError:
    print("Error: 'storm_event_data.csv' not found.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None

if df is not None:
    def convert_damage(value):
       try:
         if isinstance(value, str):
            if value.endswith('K'):
                return float(value[:-1]) * 1000
            elif value.endswith('M'):
                return float(value[:-1]) * 1000000
            elif value.endswith('B'):
                return float(value[:-1]) * 1000000000
            elif value == '0':
                return 0
            #If value is not numeric and endswith K,M or B return 0
            else:
                return np.nan # Return NaN for non-numeric values

        #If value is not a string and is numeric return value
         return float(value) if pd.notna(value) else 0
       except:
          return np.nan  #Return NaN for any other errors

    # 1. Handle Missing Values
    print("Converting 'DAMAGE_PROPERTY' and 'DAMAGE_CROPS' to numeric...")
    #df['DAMAGE_PROPERTY'] = pd.to_numeric(df['DAMAGE_PROPERTY'], errors='coerce')
    #df['DAMAGE_CROPS'] = pd.to_numeric(df['DAMAGE_CROPS'], errors='coerce')
    df['DAMAGE_PROPERTY'] = df['DAMAGE_PROPERTY'].apply(convert_damage)
    df['DAMAGE_PROPERTY'].fillna(df['DAMAGE_PROPERTY'].mean(), inplace=True)
    df['LOG_DAMAGE_PROPERTY'] = np.log1p(df['DAMAGE_PROPERTY'])

    df['DAMAGE_CROPS'] = df['DAMAGE_CROPS'].apply(convert_damage)
    df['DAMAGE_CROPS'].fillna(df['DAMAGE_CROPS'].mean(), inplace=True)
    df['LOG_DAMAGE_CROPS'] = np.log1p(df['DAMAGE_CROPS'])

    # 2. Outlier Removal (using IQR for DAMAGE_PROPERTY and DAMAGE_CROPS)
    def remove_outliers_iqr(df, column, factor=1.0):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Apply IQR only on non-zero values
    non_zero_df = df[df['LOG_DAMAGE_PROPERTY'] > 0].copy()
    filtered_df = remove_outliers_iqr(non_zero_df, 'LOG_DAMAGE_PROPERTY', factor=2.5)
    filtered_df = remove_outliers_iqr(non_zero_df, 'LOG_DAMAGE_CROPS', factor=2.5)
    filtered_df = pd.concat([df[df['LOG_DAMAGE_PROPERTY'] == 0], filtered_df])

    df = filtered_df

    # 3. Data Type Conversion (already handled above for damage columns)
    # Further data type conversion would be done here if needed

    # 4. Remove Duplicate Rows
    print("Removing duplicate rows...")
    df.drop_duplicates(inplace=True)

    # Display basic info after cleaning
    print("Displaying the first few rows and dataframe info...")
    display(df.head())
    display(df.info())

    print(f"Number of rows remaining: {len(df)}")

    #Check if 'EVENT_TYPE' exists
    if 'EVENT_TYPE' not in df.columns:
        print("'EVENT_TYPE' column not found after data cleaning.")

    # Identify infrequent event types
event_counts = df['EVENT_TYPE'].value_counts()
infrequent_events = event_counts[event_counts < 2].index

# Remove rows with infrequent event types
df_filtered = df[~df['EVENT_TYPE'].isin(infrequent_events)]

#Check if the column exists
if 'total_damage' not in df_filtered.columns:
    df_filtered['total_damage'] = df_filtered['LOG_DAMAGE_PROPERTY'] + df_filtered['LOG_DAMAGE_CROPS']

df_filtered = df_filtered.drop(columns=['LOG_DAMAGE_PROPERTY', 'LOG_DAMAGE_CROPS', 'DAMAGE_PROPERTY','DAMAGE_CROPS'])

predictor_stats = df_filtered[['total_damage','BEGIN_LAT','BEGIN_LON','INJURIES_DIRECT', \
                              'INJURIES_INDIRECT','DEATHS_DIRECT', \
                             'DEATHS_INDIRECT','MAGNITUDE']].describe().T 
display(predictor_stats)

model = sm.ols('total_damage ~ BEGIN_LAT + BEGIN_LON + INJURIES_DIRECT + INJURIES_INDIRECT + DEATHS_DIRECT + DEATHS_INDIRECT + MAGNITUDE', data=df_filtered).fit()
print(anova_lm(model, typ=2))

# Create bins for total_damage (adjust bins as needed)
damage_bins = [0, 5, 10, 15, 20, float('inf')]  
damage_labels = ['0-5', '5-10', '10-15', '15-20', '20+'] 

# Create a new column with damage categories
df_filtered['damage_category'] = pd.cut(df_filtered['total_damage'], bins=damage_bins, labels=damage_labels)

# Calculate the frequency table
freq_table = pd.crosstab(index=[df_filtered['EVENT_TYPE'], df_filtered['STATE']], 
                         columns=df_filtered['damage_category'])

# Display the frequency table
print(freq_table)


# Define features (X) and target (y)
X = df_filtered.drop('total_damage', axis=1)
y = df_filtered['total_damage']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123, stratify=df_filtered['EVENT_TYPE'])

# Select only numerical columns for VIF and scaling
numerical_cols = X_train.select_dtypes(include=np.number).columns
X_train_num = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
X_test_num = X_test[numerical_cols].fillna(X_test[numerical_cols].median())

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_num.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_num.values, i) for i in range(len(X_train_num.columns))]
display(vif_data)

# Remove features with high VIF
high_vif_cols = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
print(f"Features with high VIF (>10): {high_vif_cols}")

# Columns to exclude based on previous VIF calculations and NaN values
cols_to_exclude = ['BEGIN_YEARMONTH', 'END_YEARMONTH', 'YEAR', 'EVENT_ID', 'CATEGORY', 'TOR_OTHER_CZ_FIPS', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']
high_vif_cols = ['TOR_LENGTH', 'TOR_WIDTH', 'BEGIN_RANGE', 'END_RANGE',]

# --- Encode EVENT_TYPE and STATE using get_dummies in one go ---
X_train_encoded = pd.get_dummies(X_train[['EVENT_TYPE', 'STATE']], \
                                 prefix=['EVENT_TYPE', 'STATE'], drop_first=True).astype(int)
X_test_encoded = pd.get_dummies(X_test[['EVENT_TYPE', 'STATE']], \
                                prefix=['EVENT_TYPE', 'STATE'], drop_first=True).astype(int)
# Align columns in case any categories are missing in test set
# X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left',\
#                                                        axis=1, fill_value=0)

# --- Encode MONTH_NAME as integer (ordinal) ---
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
X_train['MONTH_NUM'] = X_train['MONTH_NAME'].map(month_mapping).astype(int)
X_test['MONTH_NUM'] = X_test['MONTH_NAME'].map(month_mapping).astype(int)

# Create a 'season' feature, specifying the datetime format
X_train['BEGIN_DATE_TIME'] = pd.to_datetime(X_train['BEGIN_DATE_TIME'], \
                                            format='%d%b%y:%H:%M:%S', errors='coerce')
X_train['SEASON'] = (X_train['BEGIN_DATE_TIME'].dt.month % 12 + 3) // 3
X_test['BEGIN_DATE_TIME'] = pd.to_datetime(X_test['BEGIN_DATE_TIME'], \
                                            format='%d%b%y:%H:%M:%S', errors='coerce')
X_test['SEASON'] = (X_test['BEGIN_DATE_TIME'].dt.month % 12 + 3) // 3

# --- Final merge: drop original and concat encoded ---
X_train_final = pd.concat([
    X_train.drop(columns=['EVENT_TYPE', 'STATE', 'MONTH_NAME']),
    X_train_encoded
], axis=1).copy()

X_test_final = pd.concat([
    X_test.drop(columns=['EVENT_TYPE', 'STATE', 'MONTH_NAME']),
    X_test_encoded
], axis=1).copy()

# Select numerical columns, excluding problematic ones
numerical_cols = [col for col in X_train_final.select_dtypes(include=np.number).columns if col not in cols_to_exclude]
X_train_num = X_train_final[numerical_cols].fillna(X_train_final[numerical_cols].median())
X_test_num = X_test_final[numerical_cols].fillna(X_test_final[numerical_cols].median())

# Drop columns with high VIF
X_train_reduced = X_train_num.drop(columns=high_vif_cols, errors='ignore')
X_test_reduced = X_test_num.drop(columns=high_vif_cols, errors='ignore')

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Instantiate a LinearRegression object
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Analyze residuals
residuals = y_test - y_pred

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

print("First few predicted values:", y_pred[:5])
print("First few actual values:", y_test[:5].values)

# Hyperparameter Tuning (L1 and L2 Regularization)
lasso = Lasso(random_state=123, max_iter=10000) # Increase max_iter
ridge = Ridge(random_state=123)

param_grid = {'alpha': np.logspace(-2, 2, 10)}

lasso_grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train_scaled, y_train)

ridge_grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train_scaled, y_train)

print(f"Best Lasso alpha: {lasso_grid_search.best_params_['alpha']}")
print(f"Best Ridge alpha: {ridge_grid_search.best_params_['alpha']}")

# Feature Selection (Based on Regularization Coefficients)
lasso_best = lasso_grid_search.best_estimator_
lasso_coef = lasso_best.coef_
selected_features = X_train_reduced.columns[np.abs(lasso_coef) > 0]

print("Number of selected features:", len(selected_features))
print("Selected features:", selected_features)

if len(selected_features) == 0:
  print("No features selected by Lasso. Using all features.")
  selected_features = X_train_reduced.columns

# Retrain Model with Selected Features
model_selected_features = LinearRegression()

# Use only selected features for training
X_train_selected = X_train_scaled[:, np.isin(X_train_reduced.columns,selected_features)]
X_test_selected = X_test_scaled[:, np.isin(X_test_reduced.columns, selected_features)]

model_selected_features.fit(X_train_selected, y_train)
y_pred_selected = model_selected_features.predict(X_test_selected)


# Evaluate Models
y_pred_lasso = lasso_best.predict(X_test_scaled)
y_pred_ridge = ridge_grid_search.best_estimator_.predict(X_test_scaled)


print("Original Model:")
print(f"  R-squared: {r2_score(y_test,y_pred)}")
print(f"  Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

print("\nLasso Model:")
print(f"  R-squared: {r2_score(y_test, y_pred_lasso)}")
print(f"  Mean Squared Error: {mean_squared_error(y_test, y_pred_lasso)}")

print("\nRidge Model:")
print(f"  R-squared: {r2_score(y_test, y_pred_ridge)}")
print(f"  Mean Squared Error: {mean_squared_error(y_test, y_pred_ridge)}")

print("\nModel with Selected Features:")
print(f"  R-squared: {r2_score(y_test, y_pred_selected)}")
print(f"  Mean Squared Error: {mean_squared_error(y_test, y_pred_selected)}")