import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the Titanic dataset
df = sns.load_dataset('titanic')
# Axes tells the Rangeindex, start =0, stop = total count, step =1 default values.
print(df.axes)
# List of first 5 rows, total count and shape is the dimension (rows, columns)
print(df.head(), df.count(), df.shape)
print(df.style.background_gradient(cmap='viridis'))

# Display Data types
print(df.dtypes)
print(df.describe())

# Selecting selective datatype objects
cat_vars = df.select_dtypes(include=['int']).columns
cat_vars.tolist()
# select_dtypes function returns subset of dataframe.
print('cat_vars', cat_vars)

# Boxplot for visualization
# Boxplot gives outliers
# Refer diagram 1.0 in datatypes notes file
plt.figure(figsize=(8,5))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age")
plt.show()

# Skewness in data
# Histogram to visualize skewness
# Refer diagram 2.0 in datatypes notes file
plt.figure(figsize=(8,5))
sns.histplot(df['age'].dropna(), kde=True, bins=30)
plt.title("Distribution of Age")
plt.show()

# Covariance Matrix, Identify the direction of relationship between 2 variables.
cov_matrix = df[['age', 'fare']].cov()
print("Covariance Matrix:\n", cov_matrix)

# Get covariance between two specific columns
cov_ab = df['age'].cov(df['fare'])
# Value 73.84 indicates that when age increases fare also increases.
print("Covariance between age and fare:", cov_ab)

# Correlation Matrix
"""
correlation matrix is a table showing the correlation coefficients between variables.
Each cell in the table shows the correlation (usually Pearson’s correlation) between two variables.
Correlation values range from -1 to 1:
1 → perfect positive correlation (when one increases, the other always increases).
-1 → perfect negative correlation (when one increases, the other always decreases).
0 → no correlation (no linear relationship).
"""
corr_matrix = df[['age','fare']].corr()
print("CorrelationMatrix:\n", corr_matrix)

# Heatmap of Correlation
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Label Encoding
le = LabelEncoder()
df['sex_encoded'] = le.fit_transform(df['sex'])
print(df[['sex', 'sex_encoded']].head())

# One-Hot Encoding, like embark_cherbourg, embark_Southampton...
df_encoded = pd.get_dummies(df, columns=['embark_town'], drop_first=True)
print(df_encoded.head())

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Min-Max Scaling, Values brought between 0 and 1.
scaler = MinMaxScaler()
df[['fare_scaled']] = scaler.fit_transform(df[['fare']])
print(df[['fare', 'fare_scaled']].head())

# Standardization, Centers data with a mean of 0 and a standard deviation of 1
std_scaler = StandardScaler()
df[['age_standardized']] = std_scaler.fit_transform(df[['age']])
print(df[['age', 'age_standardized']].head())
