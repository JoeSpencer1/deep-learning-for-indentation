import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temperature = '500'
method = '2'
n = 22

WpWt = np.zeros(n)
dPdh = np.zeros(n)
C = np.zeros(n)

for i in range(n):
    title = '../data/Ti33/' + temperature + 'C/Method ' + method + '_000'
    if i < 10:
        title = title + '0'
    title += str(i)
    title += ' LC.txt'
    print(title)
    columns = ['Depth (nm)', 'Load (uN)', 'Time (s)', 'Depth (V)', 'Load (V)', 'hmax(nm)', 'Pmax(uN)']
    with open(title, encoding='latin1') as file:
        lines = file.readlines()
    blankLines = []
    for j, line in enumerate(lines):
        if line.strip() == '':
            blankLines.append(j)
    exp = pd.read_csv(title, encoding='latin1', names=columns, skiprows=3, delimiter='\t')
    exp = exp.dropna(subset=['Depth (nm)', 'Load (uN)'])
    exp['Depth (nm)'] = pd.to_numeric(exp['Depth (nm)'], errors='coerce')
    exp['Load (uN)'] = pd.to_numeric(exp['Load (uN)'], errors='coerce')

    #print('Blank lines found at line numbers:', blankLines)
    Wt = 0
    Wp = 0
    We = 0
    #print('blankLines = ', blankLines)
    for j,  _ in enumerate(exp['Load (uN)']):
        if j >= blankLines[3] and j < blankLines[5]:
            #print(exp.loc[i, 'Depth (nm)'], ' Here1 ', exp.loc[i, 'Load (uN)'])
            Wt += (exp.loc[j, 'Depth (nm)'] - exp.loc[j - 1, 'Depth (nm)']) * (exp.loc[j, 'Load (uN)'] + exp.loc[j - 1, 'Load (uN)']) / 2
        if j >= blankLines[5]:
            #print(exp.loc[j, 'Depth (nm)'], ' Here2 ', exp.loc[j, 'Load (uN)'])
            We += (exp.loc[j - 1, 'Depth (nm)'] - exp.loc[j, 'Depth (nm)']) * (exp.loc[j, 'Load (uN)'] + exp.loc[j - 1, 'Load (uN)']) / 2
                
    Wp = Wt - We
    #print(We, ' ', Wp, ' ', Wt)
    WpWt[i] = Wp / Wt
    print('WpWt = ', WpWt[i])
    nsl = 200

    dPdh[i] = 1000 * (exp.loc[blankLines[5] + nsl, 'Load (uN)'] - exp.loc[blankLines[5], 'Load (uN)']) / (exp.loc[blankLines[5] + nsl, 'Depth (nm)'] - exp.loc[blankLines[5], 'Depth (nm)'])
    print('dP/dh = ', dPdh[i])

    title = '../data/Ti33_' + temperature + 'a.csv'
    res = pd.read_csv(title)
    res['hmax(nm)'] = pd.to_numeric(res['hmax(nm)'], errors='coerce')
    res['Pmax(uN)'] = pd.to_numeric(res['Pmax(uN)'], errors='coerce')
    #print(res.loc[0, 'hmax(nm)'])
    C[i] = 1000 * res.loc[i, 'Pmax(uN)'] / res.loc[i, 'hmax(nm)'] ** 2
    print('C = ', C[i])
    
    res['C (GPa)'] = C
    res['dP/dh'] = dPdh
    res['Wp/Wt'] = WpWt
    res.to_csv(title, index=False)