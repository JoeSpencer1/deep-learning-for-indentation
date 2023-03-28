from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, ShuffleSplit

import deepxde as dde
from data import BerkovichData, ExpData, FEMData, ModelData

'''
General summary:
Function                        Purpose
svm                             Used to check validation_model
mfgp                            In a commented line of validation_mf
nn                              Creates a regular nn, used in validation_model and validation_scaling.
validation_model                Creates a model from 2D FEM data only
validation_FEM                  A NN is trained with 2D FEM data only. 3D data is commented.
mfnn                            This function trains a MFNN with a single dataset.
validation_mf                   This makes sure the MFNN is within bounds set by 2D and Berkovich data.
validation_scaling              This makes sure the MFNN is within scaling functions
validation_exp                  Makes sure the MFNN is within exponential functions
validation_exp_cross            Trains the MFNN
validation_exp_cross2           Further trains MFNN
validation_exp_crosse           Trains the MFNN more
validation_exp_cross_transfer   Finishes training MFNN. Significant parts of this function are commented.
main                            Main function.
'''

def svm(data):
    '''
    This function is used once, in a commented line of validation_model. It can test \
        the output of the SVR function and returns the error. SVR stands for Support \
        Vector Regression.
    '''
    clf = SVR(kernel="rbf")
    clf.fit(data.train_x, data.train_y[:, 0])
    y_pred = clf.predict(data.test_x)[:, None]
    return dde.metrics.get("MAPE")(data.test_y, y_pred)


def mfgp(data):
    '''
    mfgp stands for Multi Fidelity Gaussian Process. It is used once in a commented \
        line of validation_mf. It returns the error between the predicted and \
        actual values from the high-fidelity data. 
    '''
    from mfgp import LinearMFGP

    model = LinearMFGP(noise=0, n_optimization_restarts=5)
    model.train(data.X_lo_train, data.y_lo_train, data.X_hi_train, data.y_hi_train)
    _, _, y_pred, _ = model.predict(data.X_hi_test)
    return dde.metrics.get("MAPE")(data.y_hi_test, y_pred)


def nn(data):
    '''
    nn creates and trains a neural network to model the data. This function \
        prints the mean and standard deviation of the values found by the nn. \n \
    dde.maps.FNN defines a forward neural network with activation ("selu") as the \
        function name, layer_size (data.train_x.shape[1] + [32] * 2 + [1]) as the \
        number of neurons in each layer, initializer (LeCun normal) as the weight \
        initiation, and regularization (["l2", 0.01]). L2 regularization is a \
        standard regularization also known as Ridge regularization that prevents \
        the neural network from being skewed too much by data from one set. \n \
    dde.Model creates a neural network architecture from data (obtained without nn) \
        and net (the neural netweork just created). This network is then compiled \
        with a training model.
    '''
    layer_size = [data.train_x.shape[1]] + [32] * 2 + [1]
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]

    loss = "MAPE"
    optimizer = "adam"
    if data.train_x.shape[1] == 3:
        lr = 0.0001
    else:
        lr = 0.001
    epochs = 30000
    '''
    dde.maps.FNN defines a forward neural network with activation ("selu") as the \
        function name, layer_size (data.train_x.shape[1] + [32] * 2 + [1]) as the \
        number of neurons in each layer, initializer (LeCun normal) as the weight \
        initiation, and regularization (["l2", 0.01]). L2 regularization is a \
        standard regularization also known as Ridge regularization that prevents \
        the neural network from being skewed too much by data from one set.
    '''
    net = dde.maps.FNN(
        layer_size, activation, initializer, regularization=regularization
    )
    '''
    dde.Model creates a neural network architecture from data (obtained without nn) \
        and net (the neural netweork just created). This network is then compiled \
        with a training model.
    '''
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss=loss, metrics=["MAPE"])
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return train_state.best_metrics[0]


def validation_model(yname, train_size):
    '''
    This function uses data from fitting functions of 2D FEM simulations to train \
        the NN (method 1). It is commented in the original code.
    '''
    datafem = FEMData(yname, [70])

    mape = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))

        datamodel = ModelData(yname, train_size, "forward")
        X_train, X_test = datamodel.X, datafem.X
        y_train, y_test = datamodel.y, datafem.y

        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        # mape.append(svm(data))
        mape.append(nn(data))

    print(yname, train_size)
    print(np.mean(mape), np.std(mape))


def validation_FEM(yname, angles, train_size):
    '''
    This program uses data from 2D FEM simulations to train the NN (method 2?). \
        It is commented in the original code.
    '''
    datafem = FEMData(yname, angles)
    # datafem = BerkovichData(yname)

    if train_size == 80:
        kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)
    elif train_size == 90:
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
    else:
        kf = ShuffleSplit(
            n_splits=10, test_size=len(datafem.X) - train_size, random_state=0
        )

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datafem.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter))

        X_train, X_test = datafem.X[train_index], datafem.X[test_index]
        y_train, y_test = datafem.y[train_index], datafem.y[test_index]

        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        mape.append(dde.utils.apply(nn, (data,)))

    print(mape)
    print(yname, train_size, np.mean(mape), np.std(mape))


def mfnn(data):
    '''
    This function at first did not appear to be called anywhere in the code, \
        which left me a little concerned... It actually is called at various \
        points. It is compared to the outputs of the different validation \
        functions and the experimental data during training.
    '''
    x_dim, y_dim = 3, 1
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]
    net = dde.maps.MfNN(
        [x_dim] + [128] * 2 + [y_dim],
        [8] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
        residue=True,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, loss="MAPE", metrics=["MAPE", "APE SD"])
    losshistory, train_state = model.train(epochs=30000)
    # checker = dde.callbacks.ModelCheckpoint(
    #     "model/model.ckpt", verbose=1, save_better_only=True, period=1000
    # )
    # losshistory, train_state = model.train(epochs=30000, callbacks=[checker])
    # losshistory, train_state = model.train(epochs=5000, model_restore_path="model/model.ckpt-28000")

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return (
        train_state.best_metrics[1],
        train_state.best_metrics[3],
        train_state.best_y[1],
    )

def validation_mf(yname, train_size):
    '''
    mf in this function stands for multi-fidelity. This can either use the \
        mathematical model data combined with 2D FEM data or the 2D FEM data \
        combined with 3D (Berkovich) data. All references to it are commenyed \
        at first.
    '''
    datalow = FEMData(yname, [70])
    # datalow = ModelData(yname, 10000, "forward_n")
    datahigh = BerkovichData(yname)
    # datahigh = FEMData(yname, [70])

    kf = ShuffleSplit(
        n_splits=10, test_size=len(datahigh.X) - train_size, random_state=0
    )
    # kf = LeaveOneOut()

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datahigh.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter), flush=True)

        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=datahigh.X[train_index],
            y_lo_train=datalow.y,
            y_hi_train=datahigh.y[train_index],
            X_hi_test=datahigh.X[test_index],
            y_hi_test=datahigh.y[test_index],
            standardize=True
        )
        mape.append(dde.utils.apply(mfnn, (data,))[0])
        # mape.append(dde.utils.apply(mfgp, (data,)))

    print(mape)
    print(yname, train_size, np.mean(mape), np.std(mape))


def validation_scaling(yname):
    '''
    This function outputs the value found by scaling functions. The neural \
        network never comes into play when scaling functions are used.
    '''
    datafem = FEMData(yname, [70])
    # dataexp = ExpData(yname)
    dataexp = BerkovichData(yname, scale_c=True)

    mape = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))
        data = dde.data.DataSet(
            X_train=datafem.X, y_train=datafem.y, X_test=dataexp.X, y_test=dataexp.y
        )
        mape.append(nn(data))

    print(yname)
    print(np.mean(mape), np.std(mape))


def validation_exp(yname):
    '''
    This function uses data from FEM simulations and experiment. It forms a \
        multi-fidelity data set for this.
    '''
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3067.csv", yname)

    ape = []
    y = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=dataBerkovich.X,
            y_lo_train=datalow.y,
            y_hi_train=dataBerkovich.y,
            X_hi_test=dataexp.X,
            y_hi_test=dataexp.y,
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3067.csv", yname)
    train_size = 10

    ape = []
    y = []

    # cases = range(6)
    # for train_index in itertools.combinations(cases, 3):
    #     train_index = list(train_index)
    #     test_index = list(set(cases) - set(train_index))

    kf = ShuffleSplit(
        n_splits=10, test_size=len(dataexp.X) - train_size, random_state=0
    )
    for train_index, test_index in kf.split(dataexp.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index, "==>", test_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp.X[train_index])),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp.y[train_index])),
            X_hi_test=dataexp.X[test_index],
            y_hi_test=dataexp.y[test_index],
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross2(yname, train_size):
    '''
    This function uses a data from both FEM tests and Berkovich (3D indentation) \
        tests and then trains them against data from experiments (method 4).
    '''
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp1 = ExpData("../data/B3067.csv", yname)
    dataexp2 = ExpData("../data/B3090.csv", yname)

    ape = []
    y = []

    '''
    Shufflesplit trains the neural network. train_size is the proportion of the \
        data (0-1) used to train the neural netweork. n_splits is the number of \
        iterations of the training.
    '''
    kf = ShuffleSplit(n_splits=10, train_size=train_size, random_state=0)
    '''
    This function cycles through the training data output from ShuffleSplit. It \
        displays the training index and records the y values in a .dat file. The \
        mean and standard deviation  
    '''
    for train_index, _ in kf.split(dataexp1.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp1.X[train_index])),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp1.y[train_index])),
            X_hi_test=dataexp2.X,
            y_hi_test=dataexp2.y,
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname, train_size)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross3(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp1 = ExpData("../data/Al6061.csv", yname)
    dataexp2 = ExpData("../data/Al7075.csv", yname)

    ape = []
    y = []
    for _ in range(10):
        print("\nIteration: {}".format(len(ape)))
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp1.X)),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp1.y)),
            X_hi_test=dataexp2.X,
            y_hi_test=dataexp2.y,
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt("y.dat", np.hstack(y))


def validation_exp_cross_transfer(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3090.csv", yname)
    train_size = 5

    data = dde.data.MfDataSet(
        X_lo_train=datalow.X,
        X_hi_train=dataBerkovich.X,
        y_lo_train=datalow.y,
        y_hi_train=dataBerkovich.y,
        X_hi_test=dataexp.X,
        y_hi_test=dataexp.y,
        standardize=True
    )
    res = dde.utils.apply(mfnn, (data,))
    '''
    Not sure what's going on here. The function returns but there is still more code.
    '''
    return

    ape = []
    y = []

    # cases = range(6)
    # for train_index in itertools.combinations(cases, 3):
    #     train_index = list(train_index)
    #     test_index = list(set(cases) - set(train_index))

    kf = ShuffleSplit(
        n_splits=10, test_size=len(dataexp.X) - train_size, random_state=0
    )
    for train_index, test_index in kf.split(dataexp.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index, "==>", test_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=dataexp.X[train_index],
            y_lo_train=datalow.y,
            y_hi_train=dataexp.y[train_index],
            X_hi_test=dataexp.X[test_index],
            y_hi_test=dataexp.y[test_index],
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def main():
    '''
    The main function selects which approach will be used and then performs it. \n
    validation_FEM 
    '''
    #''
    validation_FEM("Estar", [50, 60, 70, 80], 70)
    validation_mf("Estar", 9)
    validation_scaling("Estar")
    validation_exp("Estar")
    validation_exp_cross("Estar")
    validation_exp_cross2("Estar", 10)
    validation_exp_cross3("Estar")
    validation_exp_cross_transfer("Estar")
    return
    #''

    for train_size in range(1, 10):
        '''
        Varying the number of train_sizes allows you to test the advantages of \
            using a larger or smaller training dataset.
        '''
        # validation_model("Estar", train_size)
        # validation_FEM("sigma_y", [50, 60, 70, 80], train_size)
        # validation_mf("Estar", train_size)
        validation_exp_cross2("Estar", train_size)
        print("=======================================================")
        print("=======================================================")


if __name__ == "__main__":
    main()
