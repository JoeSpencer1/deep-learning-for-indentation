import pandas as pd

df = pd.read_csv("../data/Ti33_750a.csv", encoding='latin1')
df["hm (um)"] = df["hmax(nm)"] / 1000
new_column = df["hm (um)"]
df["dP/dh (N/m)"] = df["Pmax(uN)"].diff() / df["hm (um)"].diff()
new_column = df["dP/dh (N/m)"]
df.to_csv("../data/Ti33_750a.csv", index=False)