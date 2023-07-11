from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

'''
I think 3N was not used in the papar.
# Scale dP/dh from 3N to hm = 0.2um
    df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / 3) ** 0.5 * 10 ** (-1.5)

# Scale dP/dh from Pm to hm = 0.2um
    df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / df["Pm (N)"]) ** 0.5 * 10 ** (-1.5)
# Scale dP/dh from hm to hm = 0.2um
    df["dP/dh (N/m)"] *= 0.2 / df["hm (um)"]
# Scale c* from Berkovich to Conical
    df["dP/dh (N/m)"] *= 1.128 / 1.167
'''

class FEMData(object):
    def __init__(self, yname, angles):
        '''
        __init__ takes in a name and a quantity of angles. The number in [] is \
            passed to self.angles, which is the angle of the indentation. This \
            indentation angle is then used to find the correct file to read \
            to obtain the data. \n
        The class FEMData has member functions init, read_1angle, read_2angles, \
            and read_4angles. The half-included tip angles used for the read_angle \
            functions were 70.3˚, 60˚, 50˚, and 80˚. 70.3˚ was used in all \
            three and 60˚ was used in the last two. The same accuracy could be \
            achieved with a smaller training data set size for more indentors, \
            but only one indenter was used to train the single-fidelity NN.
        '''
        self.yname = yname
        self.angles = angles

        self.X = None
        self.y = None

        if len(angles) == 1:
            self.read_1angle()
        elif len(angles) == 2:
            self.read_2angles()
        elif len(angles) == 4:
            self.read_4angles()

    def read_1angle(self):
        df = pd.read_csv("../data/FEM_{}deg.csv".format(self.angles[0]))
        df["E* (GPa)"] = Etoestar(df["E (GPa)"])
        df["sy/E*"] = df["sy (GPa)"] / df["E* (GPa)"]
        df = df.loc[~((df["n"] > 0.3) & (df["sy/E*"] >= 0.03))]
        #
        # df = df.loc[df["n"] <= 0.3]
        # Scale c* from Conical to Berkovich
        # df["dP/dh (N/m)"] *= 1.167 / 1.128
        # Add noise
        # sigma = 0.2
        # df["E* (GPa)"] *= 1 + sigma * np.random.randn(len(df))
        # df["sy (GPa)"] *= 1 + sigma * np.random.randn(len(df))
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt"]].values
        if self.yname == "Estar":
            self.y = df["E* (GPa)"].values[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = float(self.yname[6:])
            self.y = (
                df["sy (GPa)"]
                * (1 + e_plastic * df["E (GPa)"] / df["sy (GPa)"]) ** df["n"]
            ).values[:, None]

    def read_2angles(self):
        df1 = pd.read_csv("../data/FEM_70deg.csv")
        df2 = pd.read_csv("../data/FEM_60deg.csv")
        df = df1.set_index("Case").join(
            df2.set_index("Case"), how="inner", rsuffix="_60"
        )
        # df = df.loc[:100]
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt", "C (GPa)_60"]].values
        # self.X = df[["C (GPa)", "dP/dh (N/m)", "C (GPa)_60", "dP/dh (N/m)_60"]].values
        if self.yname == "Estar":
            self.y = Etoestar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]

    def read_4angles(self):
        df1 = pd.read_csv("../data/FEM_50deg.csv")
        df2 = pd.read_csv("../data/FEM_60deg.csv")
        df3 = pd.read_csv("../data/FEM_70deg.csv")
        df4 = pd.read_csv("../data/FEM_80deg.csv")
        df = (
            df3.set_index("Case")
            .join(df1.set_index("Case"), how="inner", rsuffix="_50")
            .join(df2.set_index("Case"), how="inner", rsuffix="_60")
            .join(df4.set_index("Case"), how="inner", rsuffix="_80")
        )
        print(df.describe())

        self.X = df[
            [
                "C (GPa)",
                "dP/dh (N/m)",
                "Wp/Wt",
                "C (GPa)_50",
                "C (GPa)_60",
                "C (GPa)_80",
            ]
        ].values
        if self.yname == "Estar":
            self.y = Etoestar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]


class ModelData(object):
    def __init__(self, yname, n, model):
        self.yname = yname
        self.n = n
        self.model = model

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv("../data/model_{}.csv".format(self.model))
        self.X = df[["C (GPa)", "dP/dh (N/m)", "WpWt"]].values
        if self.yname == "Estar":
            self.y = Etoestar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        idx = np.random.choice(np.arange(len(self.X)), self.n, replace=False)
        self.X = self.X[idx]
        self.y = self.y[idx]


class ExpData(object):
    def __init__(self, filename, yname):
        '''
        ExpData reads in data from an experimental data file. It intakes values \
            for C, E*, sy, and s for varying plastic strains. The filename it \
            receives as an argument is the experimental data file that will be \
            read.
        '''
        self.filename = filename
        self.yname = yname

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv(self.filename)

        #
        # Scale nm to um for Ti33 files
        # I'M PRETTY SURE THESE MULTI-LINE COMMENTED CAN BE DELETED.
        '''
        df["hm (um)"] = df["hmax(nm)"] / 1000
        df["C (GPA)"] = df["H(GPa)"] * df["hm (um)"] ** 2
        df["H(GPa)"]
        df["hc(nm)"]
        df["hf(nm)"]
        self.X = df[["hmax(nm)", "H(GPa)", "hc(nm)"]].values
        '''
        # Scale dP/dh from 3N to hm = 0.2um

# This is for Al alloys
        #df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / 3) ** 0.5 * 10 ** (-1.5)

        
        # Scale dP/dh from Pm to hm = 0.2um
        # df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / df["Pm (N)"]) ** 0.5 * 10 ** (-1.5)
        # Scale dP/dh from hm to hm = 0.2um 

# This is for Ti alloys
#        df["dP/dh (N/m)"] *= 0.2 / df["hm (um)"]
# This is for the Yanbo's Ti alloys
        df["dP/dh (N/m)"] *= 0.2 * 1000 / df["hmax(nm)"]

        # Scale c* from Berkovich to Conical
#        df["dP/dh (N/m)"] *= 1.128 / 1.167
        #

        print(df.describe())

# I just commented this line for my own work.
        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt"]].values
        if self.yname == "Estar":
            self.y = df["E* (GPa)"].values[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = self.yname[6:]
            self.y = df["s" + e_plastic + " (GPa)"].values[:, None]


class BerkovichData(object):
    def __init__(self, yname, scale_c=False):
        '''
        The class BerkovichData reads a file from a Berkovich indentation test. \
            It has member functions init and read. init sets the scale and the \
            name of the dependent variables. read reads the csv of the given name \
            and stores its C, E*, sy, and n. It can also store dP/dh if scale is \
            listed as being true. \n
        The Berkovich indenter has a half angle of 65.3˚ from the tip to the pyramid \
            surface.
        '''
        self.yname = yname
        self.scale_c = scale_c

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv("../data/Berkovich.csv")
        # Scale c* from Berkovich to Conical
        if self.scale_c:
            df["dP/dh (N/m)"] *= 1.128 / 1.167
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt"]].values
        if self.yname == "Estar":
            self.y = Etoestar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname == "n":
            self.y = df["n"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = float(self.yname[6:])
            self.y = (
                df["sy (GPa)"]
                * (1 + e_plastic * df["E (GPa)"] / df["sy (GPa)"]) ** df["n"]
            ).values[:, None]


class FEMDataT(object):
    def __init__(self, yname, angles):
        '''
        __init__ takes in a name and a quantity of angles. The number in [] is \
            passed to self.angles, which is the angle of the indentation. This \
            indentation angle is then used to find the correct file to read \
            to obtain the data. \n
        The class FEMData has member functions init, read_1angle, read_2angles, \
            and read_4angles. The half-included tip angles used for the read_angle \
            functions were 70.3˚, 60˚, 50˚, and 80˚. 70.3˚ was used in all \
            three and 60˚ was used in the last two. The same accuracy could be \
            achieved with a smaller training data set size for more indentors, \
            but only one indenter was used to train the single-fidelity NN.
        '''
        self.yname = yname
        self.angles = angles

        self.X = None
        self.y = None

        if len(angles) == 1:
            self.read_1angle()
        elif len(angles) == 2:
            self.read_2angles()
        elif len(angles) == 4:
            self.read_4angles()

    def read_1angle(self):
        df = pd.read_csv("../data/FEM_{}deg.csv".format(self.angles[0]))
        df["E* (GPa)"] = Etoestar(df["E (GPa)"])
        df["sy/E*"] = df["sy (GPa)"] / df["E* (GPa)"]
        df = df.loc[~((df["n"] > 0.3) & (df["sy/E*"] >= 0.03))]
        #
        # df = df.loc[df["n"] <= 0.3]
        # Scale c* from Conical to Berkovich
        # df["dP/dh (N/m)"] *= 1.167 / 1.128
        # Add noise
        # sigma = 0.2
        # df["E* (GPa)"] *= 1 + sigma * np.random.randn(len(df))
        # df["sy (GPa)"] *= 1 + sigma * np.random.randn(len(df))
        print(df.describe())
        df["T (C)"] = df['Wp/Wt']
        df["T (C)"] = 25
        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt", "T (C)"]].values
        if self.yname == "Estar":
            self.y = df["E* (GPa)"].values[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = float(self.yname[6:])
            self.y = (
                df["sy (GPa)"]
                * (1 + e_plastic * df["E (GPa)"] / df["sy (GPa)"]) ** df["n"]
            ).values[:, None]


class BerkovichDataT(object):
    def __init__(self, yname, scale_c=False):
        '''
        The class BerkovichData reads a file from a Berkovich indentation test. \
            It has member functions init and read. init sets the scale and the \
            name of the dependent variables. read reads the csv of the given name \
            and stores its C, E*, sy, and n. It can also store dP/dh if scale is \
            listed as being true. \n
        The Berkovich indenter has a half angle of 65.3˚ from the tip to the pyramid \
            surface.
        '''
        self.yname = yname
        self.scale_c = scale_c

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv("../data/Berkovich.csv")
        # Scale c* from Berkovich to Conical
        if self.scale_c:
            df["dP/dh (N/m)"] *= 1.128 / 1.167
        print(df.describe())

        df["T (C)"] = df['Wp/Wt']
        df["T (C)"] = 25
        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt", "T (C)"]].values
        if self.yname == "Estar":
            self.y = Etoestar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname == "n":
            self.y = df["n"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = float(self.yname[6:])
            self.y = (
                df["sy (GPa)"]
                * (1 + e_plastic * df["E (GPa)"] / df["sy (GPa)"]) ** df["n"]
            ).values[:, None]


class ExpDataT(object):
    def __init__(self, filenames, yname):
        '''
        ExpData reads in data from an experimental data file. It intakes values \
            for C, E*, sy, and s for varying plastic strains. The filename it \
            receives as an argument is the experimental data file that will be \
            read.
        '''
        self.filenames = filenames
        self.yname = yname

        self.X = None
        self.y = None

        self.read()

    def read(self):
        for filename in self.filenames:
            df = pd.read_csv(filename)
            # Scale dP/dh from 3N to hm = 0.2um
    # This is for Al alloys
            #df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / 3) ** 0.5 * 10 ** (-1.5)

            # Scale dP/dh from Pm to hm = 0.2um
            # df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / df["Pm (N)"]) ** 0.5 * 10 ** (-1.5)
            # Scale dP/dh from hm to hm = 0.2um 

    # This is for Ti alloys
    #        df["dP/dh (N/m)"] *= 0.2 / df["hm (um)"]
    # This is for the Yanbo's Ti alloys
            df["dP/dh (N/m)"] *= 0.2 * 1000 / df["hmax(nm)"]

            # Scale c* from Berkovich to Conical
    #        df["dP/dh (N/m)"] *= 1.128 / 1.167
            #

            print(df.describe())

            if self.X is None:
                self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt", "T (C)"]].values
                if self.yname == "Estar":
                    self.y = df["E* (GPa)"].values[:, None]
                elif self.yname == "sigma_y":
                    self.y = df["sy (GPa)"].values[:, None]
                elif self.yname.startswith("sigma_"):
                    e_plastic = self.yname[6:]
                    self.y = df["s" + e_plastic + " (GPa)"].values[:, None]
            else:
                self.X = np.vstack((self.X, df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt", "T (C)"]].values))
                if self.yname == "Estar":
                    self.y = np.vstack((self.y, df["E* (GPa)"].values[:, None]))
                elif self.yname == "sigma_y":
                    self.y = np.vstack((self.y, df["sy (GPa)"].values[:, None]))
                elif self.yname.startswith("sigma_"):
                    e_plastic = self.yname[6:]
                    self.y = np.vstack((self.y, df["s" + e_plastic + " (GPa)"].values[:, None]))


def Etoestar(E):
    nu = 0.3
    nu_i, E_i = 0.07, 1100
    return 1 / ((1 - nu ** 2) / E + (1 - nu_i ** 2) / E_i)
