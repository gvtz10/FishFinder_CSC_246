import pandas as pd
import sklearn
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def linearSVM (c):
   svm = sklearn.svm.LinearSVC(C=c, dual = True)
   return svm

def kernalSVM(c, gamma):
    if gamma is None:
        svm = sklearn.svm.SVC(C=c)
    else:
        svm = sklearn.svm.SVC(C=c, gamma = gamma)
    return svm

def MLP(hidden_units, learning_rate,):
    mlp = keras.Sequential()
    input_layer = keras.layers.Dense(56,input_dim=(56),
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
    hidden_layer = keras.layers.Dense(hidden_units, activation = 'relu',
                                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
    output_layer = keras.layers.Dense(1, activation = 'sigmoid',
                                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
    mlp.add(input_layer)
    mlp.add(hidden_layer)
    mlp.add(output_layer)
    mlp.compile(
        optimizer= keras.optimizers.Adam( learning_rate = learning_rate),
        loss = keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    mlp.summary()
    return mlp

def kfoldSVM(svm, x_data, y_data, folds):
    kf = sklearn.model_selection.KFold(n_splits = folds, random_state=None, shuffle=False)
    scores = sklearn.model_selection.cross_val_score(svm,X = x_data, y = y_data , cv = kf, scoring='f1_macro')
    return np.average(scores)

def kfoldMLP(mlp, x_data, y_data, folds, num_epochs):
    f1_per_fold = []
    kFold = sklearn.model_selection.KFold(n_splits= folds)
    for train, test in kFold.split(x_data, y_data):
        mlp.fit(x_data.iloc[train], y_data.iloc[train], epochs=num_epochs)
        y_pred = mlp.predict(x_data.iloc[test])
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        scores = sklearn.metrics.f1_score(y_pred, y_data.iloc[test])
        f1_per_fold.append(scores)
    return np.average(f1_per_fold)

def get_kfold_scores_MLP(x_data, y_data, folds, num_epochs, Range, learning_rate):
    kfold_scores = []
    for i in Range:
        mlp = MLP(i, learning_rate)
        kfold_scores.append(kfoldMLP(mlp, x_data, y_data, folds, num_epochs))
    return kfold_scores

def get_kfold_scores_keranl_SVM( x_data, y_data, folds, Range, C_or_gamma): # C = True , Gamma = False
    kfold_scores = []
    for i in Range:
        if C_or_gamma:
            svm = kernalSVM(c=i, gamma='scale')
        else:
            svm = kernalSVM(c=1.0, gamma=i)
        kfold_scores.append(kfoldSVM(svm, x_data, y_data, folds))
    return kfold_scores

def get_kfold_scores_linear_SVM( x_data, y_data, folds, Range):
    kfold_scores = []
    for i in Range:
        svm = linearSVM(i)
        kfold_scores.append(kfoldSVM(svm, x_data, y_data, folds))
    return kfold_scores

def get_training_score_MLP (x_data, y_data, num_epochs, Range,learning_rate):
    training_score = []
    for i in Range:
        mlp = MLP(hidden_units=i,learning_rate =learning_rate)
        mlp.fit(x_data, y_data, epochs= num_epochs)
        y_pred = mlp.predict(x_data)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        training_score.append(sklearn.metrics.f1_score(y_pred, y_data))
    return training_score

def get_training_score_Linear_SVM(x_data, y_data,Range):
    training_score = []
    for i in Range:
        svm = linearSVM(i)
        svm.fit(X=x_data, y=y_data)
        y_pred = svm.predict(x_data)
        training_score.append(sklearn.metrics.f1_score(y_pred, y_data))
    return training_score

def get_training_score_kernal_SVM(x_data, y_data, Range, C_or_gamma): # C = True , Gamma = False
    training_score = []
    for i in Range:
        if(C_or_gamma):
            svm = kernalSVM(c=i, gamma='scale')
        else:
            svm = kernalSVM(c=1.0, gamma=i)
        svm.fit(X=x_data, y=y_data)
        y_pred = svm.predict(x_data)
        training_score.append(sklearn.metrics.f1_score(y_pred, y_data))
    return training_score

def get_Range(min, max, rate):
    Range = []
    while(min< max):
        min = min + rate
        Range.append(float(min))
    return Range

def plot_Model(model_type, Cross_Val_Score, Training_Score, Range, X_axis):
    plt.plot(Range, Cross_Val_Score, '-o', label="Cross-Validation Score")
    plt.plot(Range, Training_Score, '-o', label="Training Score")
    plt.xlabel(X_axis)
    if model_type == "Kernel SVM" or model_type == "Linear SVM":
        plt.xscale('log')
    plt.ylabel("F1 Score")
    plt.title(f"{model_type} Preformace Based on {X_axis}")
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    plt.show()

def Plot_linear_SVM():
    xvals, yvals = load_newts(do_min_max=True)
    range = [.0001, .001, .01, 1, 10, 100, 1000]
    training_score = get_training_score_Linear_SVM(xvals,yvals,range)
    kfold_score = get_kfold_scores_linear_SVM(xvals, yvals,5, range)
    plot_Model("Linear SVM", kfold_score,training_score, range, "C (log scale)")

def Plot_keranl_SVM_C():
    xvals, yvals = load_newts(do_min_max=True)
    range = [.0001, .001, .01, 1, 10, 100, 1000]
    training_score = get_training_score_kernal_SVM(xvals, yvals, range, True)
    kfold_score = get_kfold_scores_keranl_SVM(xvals, yvals,5, range, True)
    plot_Model("Kernel SVM", kfold_score, training_score, range, "C (log scale)")
    print("KFOLD F1 Score")
    print(kfold_score)

def Plot_keranl_SVM_gamma():
    xvals, yvals = load_newts(do_min_max=True)
    range = [.0001, .001, .01, 1, 10, 100]
    training_score = get_training_score_kernal_SVM(xvals, yvals, range, False)
    kfold_score = get_kfold_scores_keranl_SVM(xvals, yvals, 5, range, False)
    plot_Model("Kernel SVM", kfold_score, training_score, range, "gamma (log scale)")
    print("KFOLD F1 Score")
    print(kfold_score)
def Plot_MLP():
    num_epochs = 100
    folds = 5
    learning_rate = .001
    xvals, yvals = load_newts(do_min_max=True)
    range = get_Range(1, 100, 10)
    training_score = get_training_score_MLP(xvals, yvals, num_epochs, range,learning_rate)
    kfold_score = get_kfold_scores_MLP(xvals, yvals, folds, num_epochs, range, learning_rate)
    plot_Model("MLP", kfold_score, training_score, range, "Hidden Units")
    print("KFOLD F1 Score")
    print(kfold_score)

def load_newts(do_min_max=False):
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00528/dataset.csv', delimiter=';', skiprows=1)
    xvals_raw = data.drop(['ID', 'Green frogs', 'Brown frogs', 'Common toad', 'Tree frog', 'Common newt', 'Great crested newt', 'Fire-bellied toad'], axis=1)
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR'])
    yvals = data['Fire-bellied toad']

    if (do_min_max):
        for col in ['SR', 'NR', 'TR', 'VR', 'OR', 'RR', 'BR']:
            xvals_raw[col] = (xvals_raw[col] - xvals_raw[col].min())/(xvals_raw[col].max() - xvals_raw[col].min())
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR'])
    return xvals, yvals
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   model = input('would you like to test a linear svm(0), a kernel svm(1), or a MLP(2):')
   if model == '0':
       Plot_linear_SVM()
   elif model == '1':
       c_g = input('sweeping C(0) of sweeping gamma(1)')
       if c_g == '0':
           Plot_keranl_SVM_C()
       elif c_g == '1':
           Plot_keranl_SVM_gamma()
   elif model == '2':
       Plot_MLP()
   else:
       print('please enter a valid input')


