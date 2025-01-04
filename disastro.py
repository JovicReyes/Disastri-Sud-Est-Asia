import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display
from sklearn.linear_model import LinearRegression


df = pd.read_excel(r"/Users/jovic/Desktop/sea/disaster_sea.xlsx")

#celle
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#colonne
df_info = pd.read_csv("/Users/jovic/Desktop/sea/attributes.csv")
#df.columns

#caratteristiche peculiari
df["Disaster Type"].unique()
len(df["Disaster Type"].unique())

df["Country"].unique()
len(df["Country"].unique())

df.groupby("Country")["ISO"].unique()

#valori nulli
df.isnull().sum()

#eliminazione colonne
# df = df.drop(columns=[col for col in ['Seq', 'Glide', 'Disaster Group', 'Disaster Subgroup', 
#                                       'Disaster Subtype', 'Disaster Subsubtype', 'Region', 
#                                       'Continent', 'Origin', 'Associated Dis', 'Associated Dis2', 
#                                       'Appeal', 'Declaration', 'Aid Contribution', 'Latitude', 
#                                       'Longitude', 'Local Time', 'River Basin', 'Start Year', 
#                                       'No Affected', 'Reconstruction Costs (\'000 US$)', 
#                                       'Reconstruction Costs, Adjusted (\'000 US$)', 
#                                       'Insured Damages (\'000 US$)', 'Insured Damages, Adjusted (\'000 US$)', 
#                                       'CPI', 'Adm Level', 'Admin1 Code', 'Admin2 Code', 'Geo Locations']
#                    if col in df.columns], axis=1)


df1 = df.sort_values("Start Year", ascending=True).reset_index(drop=True) #indice resettato
pd.reset_option('display.max_columns')
pd.reset_option('display.width', 1000)
#df1

df1.loc[(df["Disaster Type"]=="Earthquake") | (df["Disaster Type"] == "Flood") | (df["Disaster Type"] == "Storm")]

result = df1.groupby("Country").agg(Total_Disasters=("DisNo.", "count"), Total_Damages = ('Total Damage (\'000 US$)', "sum")).reset_index()

result_sorted = result.sort_values(by="Total_Disasters", ascending=False)
#print(result_sorted)


# print(df1.loc[(df1["ISO"] == "PHL")])
# print(df1.loc[(df1["ISO"] == "PHL") & (df1["Disaster Type"] == "Storm") & (df1["Start Year"] >= 2020)])

#scala livelli tempeste
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df_scale = pd.read_csv("/Users/jovic/Desktop/sea/level_storm_scale.csv")
#print(df_scale)

#LR
philippines_data = df[df["Country"] == "Philippines"]

disaster_counts = philippines_data.groupby(["Start Year", "Disaster Type"]).size().unstack(fill_value=0)

storm_counts = disaster_counts["Storm"]

X = storm_counts.index.values.reshape(-1, 1)
y = storm_counts.values

model = LinearRegression()
model.fit(X, y)

future_years = np.array([[2025], [2026], [2027], [2028]])
predictions = model.predict(future_years)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color="blue", label="Historical Storm Counts")
plt.plot(X, model.predict(X), color="green", label="Linear Regression Line")
plt.scatter(future_years, predictions, color="red", label="Predicted Storm Counts", marker="x")
plt.title("Storm Counts and Linear Regression Model")
plt.xlabel("Year")
plt.ylabel("Number of Storms")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for year, prediction in zip(future_years.flatten(), predictions):
    print(f"Predicted number of storms in {year}: {int(prediction)}")

