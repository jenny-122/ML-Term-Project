import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# initialize file names
f1 = 'dataset1-pre.csv' # electrons
f2 = 'dataset2.csv' #
f3 = 'dataset3.csv' #

# initialize dataframes: all empty
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

files_list = [f1, f2, f3]

#file_path = '/content/drive/MyDrive/Colab-Notebooks/ML-Data/' + f1
#df1 = pd.read_csv(file_path, names=['userId', 'productId','Rating','timestamp'])

def load_df(fname, df, name):
  #file_path = '/content/drive/MyDrive/Colab-Notebooks/ML-Data/' + fname
  file_path = fname

  # Read the CSV file
  new_data = pd.read_csv(file_path)

  # If df is still empty, you can directly assign new_data to df
  if df.empty:
    df = new_data
  else:
    # Concatenate the new data with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

  print(name + ' has loaded in.')
  #print(df.shape)
  #print(df.head())
  #print('\n')
  return df

df1 = load_df(f1, df1, 'df1')
df2 = load_df(f2, df2, 'df2')
df3 = load_df(f3, df3, 'df3')

df_list = [df1, df2, df3]

sample_size = 5000
sampled_df1 = df1.sample(sample_size, random_state=42)

# Define the filename
filename = 'dataset1.csv'

dir = os.getcwd()

# Write DataFrame to CSV file in the specified directory
file_path = os.path.join(dir, filename)
sampled_df1.to_csv(file_path, index=False)

# Print the path where the CSV file was saved
print("CSV file saved at:", file_path)