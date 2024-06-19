#Load the CSV File
import pandas as pd

#Load the CSV file using pandas
#url = 'https://github.com/HanSquared/cmu-95829-Azure-TDSP/blob/main/Sample_Data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
dataLoc ="/Users/hanhan/Desktop/CMU/Software_Design_for_Data_Scientists/GitHub/cmu-95829-Azure-TDSP/Sample_Data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(dataLoc,sep = ',')
# Display the first few rows of the dataframe to ensure it's loaded correctly
print("Data preview:")
print(df.head())
print("\nData info:")
print(df.info)
print("\nData description:")
print(df.describe())


