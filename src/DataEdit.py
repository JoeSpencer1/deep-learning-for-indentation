import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ["Depth (nm)", "Load (uN)", "Time (s)", "Depth (V)", "Load (V)"]
with open("../data/Ti33/25C/Method 1_00000 LC.txt", encoding='latin1') as file:
    lines = file.readlines()
blankLines = []
for i, line in enumerate(lines):
    if line.strip() == "":
        blankLines.append(i)
exp = pd.read_csv("../data/Ti33/25C/Method 1_00000 LC.txt", encoding='latin1', names=columns, skiprows=3, delimiter='\t')
exp = exp.dropna(subset=["Depth (nm)", "Load (uN)"])
exp["Depth (nm)"] = pd.to_numeric(exp["Depth (nm)"], errors='coerce')
exp["Load (uN)"] = pd.to_numeric(exp["Load (uN)"], errors='coerce')
#print(exp["Load (uN)"])
#plt.scatter(exp["Depth (nm)"], exp["Load (uN)"])
#plt.show()
#print("Blank lines found at line numbers:", blankLines)
Wt = 0
Wp = 0
We = 0
for i,  _ in enumerate(exp["Load (uN)"]):
    if i >= blankLines[3] and i < blankLines[5]:
        #print(exp.loc[i, "Depth (nm)"], ' Here1 ', exp.loc[i, "Load (uN)"])
        Wt += (exp.loc[i, "Depth (nm)"] - exp.loc[i - 1, "Depth (nm)"]) * (exp.loc[i, "Load (uN)"] + exp.loc[i - 1, "Load (uN)"]) / 2
    if i >= blankLines[5]:
        #print(exp.loc[i, "Depth (nm)"], ' Here2 ', exp.loc[i, "Load (uN)"])
        Wp += (exp.loc[i - 1, "Depth (nm)"] - exp.loc[i, "Depth (nm)"]) * (exp.loc[i, "Load (uN)"] + exp.loc[i - 1, "Load (uN)"]) / 2
We = Wt - Wp
#print(We, ' ', Wp, ' ', Wt)
outf = pd.DataFrame()