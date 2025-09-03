import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Titanic dataset
df = sns.load_dataset('titanic')
# Axes tells the Rangeindex, start =0, stop = total count, step =1 default values.
print(df.axes)
print('DF params.....',df.head(), df.count(), df.shape)

# Display Data types
print(df.dtypes)
print(df.describe())

# Selecting selective datatype objects
cat_vars = df.select_dtypes(include=['int']).columns
cat_vars.tolist()
# select_dtypes fn returns subset of dataframe.
print('cat_vars', cat_vars)

# Boxplot for visualization
# Boxplot gives outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age")
plt.show()