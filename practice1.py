

import pandas as pd
import re

df = pd.read_csv("/Users/akshitkashyap/Desktop/RASHMI/Job_projects/projects/project_3/Luxury_Beauty_5_part0.csv")

# Data loading
print(" Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())



# Data Understanding / Exploration
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())




#  Handling Missing and Duplicate Data
df = df.dropna(subset=['reviewText'])
df = df.drop_duplicates(subset=['reviewText'])
print("\n Removed null and duplicate reviews.")
print("New Shape:", df.shape)



#  Convert text to lowercase
df['reviewText'] = df['reviewText'].str.lower()



