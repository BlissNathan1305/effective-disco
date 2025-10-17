import pandas as pd
import numpy as np

# ========================================
# PART 1: GETTING STARTED WITH PANDAS
# ========================================

# 1. CREATING DATAFRAMES
# ----------------------

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000],
    'department': ['HR', 'IT', 'IT', 'Finance']
}
df = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df)
print("\n")

# From list of lists
data_list = [
    ['Alice', 25, 50000],
    ['Bob', 30, 60000],
    ['Charlie', 35, 75000]
]
df2 = pd.DataFrame(data_list, columns=['name', 'age', 'salary'])

# 2. READING DATA FROM FILES
# ---------------------------

# Read CSV
# df = pd.read_csv('data.csv')

# Read Excel
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read JSON
# df = pd.read_json('data.json')

# 3. BASIC DATAFRAME INSPECTION
# ------------------------------

print("First 3 rows:")
print(df.head(3))
print("\n")

print("Last 2 rows:")
print(df.tail(2))
print("\n")

print("DataFrame info:")
print(df.info())
print("\n")

print("Statistical summary:")
print(df.describe())
print("\n")

print("Column names:")
print(df.columns.tolist())
print("\n")

print("Shape (rows, columns):")
print(df.shape)
print("\n")

# 4. ACCESSING DATA
# -----------------

# Select single column (returns Series)
print("Ages:")
print(df['age'])
print("\n")

# Select multiple columns
print("Name and salary:")
print(df[['name', 'salary']])
print("\n")

# Select rows by index
print("First row:")
print(df.iloc[0])
print("\n")

# Select rows by condition
print("People over 28:")
print(df[df['age'] > 28])
print("\n")

# Select with loc (by label)
print("Specific row and columns:")
print(df.loc[0:2, ['name', 'age']])
print("\n")

# ========================================
# PART 2: DATA CLEANING
# ========================================

# Create a messy dataset for cleaning examples
messy_data = {
    'Name': ['Alice', 'bob', 'CHARLIE', 'David', 'Eve', 'Frank', None, 'Grace'],
    'Age': [25, 30, np.nan, 28, 150, 22, 27, 24],
    'Email': ['alice@email.com', 'bob@email', 'charlie@email.com', 'david@email.com', 
              'eve@email.com', 'frank@email.com', 'grace@email.com', 'grace@email.com'],
    'Salary': [50000, 60000, 75000, np.nan, 90000, -5000, 48000, 52000],
    'Join_Date': ['2020-01-15', '2019-03-20', '2018-06-10', '2021-02-28', 
                  '2020-08-05', 'invalid_date', '2021-11-12', '2022-01-30']
}
messy_df = pd.DataFrame(messy_data)

print("=" * 50)
print("MESSY DATASET:")
print(messy_df)
print("\n")

# 1. HANDLING MISSING VALUES
# ---------------------------

print("Missing values count:")
print(messy_df.isnull().sum())
print("\n")

# Drop rows with any missing values
df_dropped = messy_df.dropna()
print("After dropping rows with missing values:")
print(df_dropped)
print("\n")

# Fill missing values
df_filled = messy_df.copy()
df_filled['Name'].fillna('Unknown', inplace=True)
df_filled['Age'].fillna(df_filled['Age'].median(), inplace=True)
df_filled['Salary'].fillna(df_filled['Salary'].mean(), inplace=True)
print("After filling missing values:")
print(df_filled)
print("\n")

# 2. REMOVING DUPLICATES
# ----------------------

print("Duplicate emails:")
print(messy_df[messy_df.duplicated(subset=['Email'], keep=False)])
print("\n")

# Remove duplicates
df_unique = messy_df.drop_duplicates(subset=['Email'], keep='first')
print("After removing duplicate emails:")
print(df_unique)
print("\n")

# 3. DATA TYPE CONVERSION
# -----------------------

print("Original data types:")
print(messy_df.dtypes)
print("\n")

# Convert to appropriate types
df_converted = messy_df.copy()
df_converted['Age'] = pd.to_numeric(df_converted['Age'], errors='coerce')
df_converted['Salary'] = pd.to_numeric(df_converted['Salary'], errors='coerce')
df_converted['Join_Date'] = pd.to_datetime(df_converted['Join_Date'], errors='coerce')

print("After type conversion:")
print(df_converted.dtypes)
print("\n")

# 4. STRING CLEANING
# ------------------

# Standardize text case
df_clean = df_converted.copy()
df_clean['Name'] = df_clean['Name'].str.title()  # Title case
print("After standardizing names:")
print(df_clean['Name'])
print("\n")

# Remove whitespace
df_clean['Name'] = df_clean['Name'].str.strip()

# 5. HANDLING OUTLIERS
# --------------------

# Identify outliers using IQR method
Q1 = df_clean['Age'].quantile(0.25)
Q3 = df_clean['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Age outlier bounds: {lower_bound} to {upper_bound}")
outliers = df_clean[(df_clean['Age'] < lower_bound) | (df_clean['Age'] > upper_bound)]
print("Age outliers:")
print(outliers)
print("\n")

# Remove outliers
df_no_outliers = df_clean[(df_clean['Age'] >= lower_bound) & (df_clean['Age'] <= upper_bound)]

# 6. REPLACING VALUES
# -------------------

# Replace negative salaries
df_clean.loc[df_clean['Salary'] < 0, 'Salary'] = np.nan

# Replace specific values
df_clean['Department'] = df_clean.get('Department', pd.Series()).replace('IT', 'Technology')

# 7. CREATING NEW COLUMNS
# -----------------------

# Add calculated column
df_clean['Years_Employed'] = (pd.Timestamp.now() - df_clean['Join_Date']).dt.days / 365.25
print("With calculated column:")
print(df_clean[['Name', 'Join_Date', 'Years_Employed']])
print("\n")

# 8. RENAMING COLUMNS
# -------------------

df_clean = df_clean.rename(columns={'Name': 'employee_name', 'Age': 'employee_age'})
print("After renaming columns:")
print(df_clean.columns.tolist())
print("\n")

# 9. FILTERING AND VALIDATION
# ---------------------------

# Keep only valid data
df_final = df_clean[
    (df_clean['employee_age'].notna()) &
    (df_clean['employee_age'] > 0) &
    (df_clean['employee_age'] < 120) &
    (df_clean['Salary'].notna()) &
    (df_clean['Salary'] > 0)
].copy()

print("FINAL CLEANED DATASET:")
print(df_final)
print("\n")

# 10. EXPORTING CLEANED DATA
# --------------------------

# Save to CSV
# df_final.to_csv('cleaned_data.csv', index=False)

# Save to Excel
# df_final.to_excel('cleaned_data.xlsx', index=False)

print("Data cleaning complete!")
