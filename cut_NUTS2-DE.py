import pandas as pd

df = pd.read_csv("data/H_ERA5_ECMW_T639_GHI_0000m_Euro_NUT0.csv", header=52)
print(df.head(20))