import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import biosppy.signals.ecg as ecg
from sklearn.model_selection import train_test_split

input_train = torch.ones((20, 5), dtype=torch.float32)
target = torch.empty(20, dtype=torch.long).random_(5)
print(input_train.shape)
print(target.shape)
print(F.cross_entropy(input_train, target))
exit(-1)


def readts(file_name):
    lines = []
    with open(file_name) as file:
        for line in file:
            line = line.strip()
            lines.append(line)
    lines.remove(lines[0])
    for i in range(len(lines)):
        lines[i] = re.split(',', lines[i])
        lines[i] = list(map(int, lines[i]))
    return lines


x_train = readts('data/X_train.csv')
y_train = pd.read_csv('data/Y_train.csv')
x_test = readts('data/X_test.csv')

out = ecg.ecg(signal=x_test[2490], sampling_rate=300, show=True)
out[6]
ts = out[0]
filtered = out[1]
rpeaks = out[2]
templates_ts = out[3]
templates = out[4]
heart_rate_ts = out[5]
heart_rate = np.mean(out[6])
hr_var = np.std(out[6])
dataset = pd.DataFrame({'hart': x_train[0]})

measures = {}


def calc_RR():
    peaklist = out[0][out[2]]
    RR_interval = np.diff(peaklist) * 1000
    measures['ibi'] = np.mean(RR_interval)
    measures['sdnn'] = np.std(RR_interval)
    interval_diff = abs(np.diff(RR_interval))
    nn20 = [x for x in interval_diff if (x > 20)]
    nn50 = [x for x in interval_diff if (x > 50)]
    measures['pnn20'] = float(len(nn20)) / float(len(interval_diff))
    measures['pnn50'] = float(len(nn50)) / float(len(interval_diff))
    measures['rmssd'] = np.sqrt(np.mean(np.power(interval_diff, 2)))
    measures['sdsd'] = np.std(interval_diff)


def calc_ts_measures():
    RR_list = measures['RR_list']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']

    # NN20 = [x for x in RR_diff if (x>20)]
    # NN50 = [x for x in RR_diff if (x>50)]
    # measures['nn20'] = NN20
    # measures['nn50'] = NN50
    # measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    # measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))


def process(dataset,
            fs):  # Remember; hrw was the one-sided window size (we used 0.75) and fs was the sample rate (file is recorded at 100Hz)
    calc_RR(dataset, fs)
    # calc_ts_measures()


meanhr = []
minhr = []
maxhr = []
stdhr = []
IBI = []
SDNN = []
SDSD = []
RMSSD = []
pNN20 = []
pNN50 = []
for i in range(len(x_train)):
    out = ecg.ecg(signal=x_train[i][10:], sampling_rate=300, show=False)
    meanhr.append(np.mean(out[6]))
    minhr.append(min(out[6]))
    maxhr.append(max(out[6]))
    stdhr.append(np.std(out[6]))
    calc_RR()
    IBI.append(measures['ibi'])
    pNN20.append(measures['pnn20'])
    pNN50.append(measures['pnn50'])
    SDNN.append(measures['sdnn'])
    SDSD.append(measures['sdsd'])
    RMSSD.append(measures['rmssd'])
    print(str(i))
train_features = pd.DataFrame()
train_features['heart_rate'] = pd.Series(meanhr)
train_features['minhr'] = pd.Series(minhr)
train_features['maxhr'] = pd.Series(maxhr)
train_features['stdhr'] = pd.Series(stdhr)
train_features['ibi'] = pd.Series(IBI)
# train_features['pnn20'] = pd.Series(pNN20)
train_features['pnn50'] = pd.Series(pNN50)
train_features['sdnn'] = pd.Series(SDNN)
train_features['sdsd'] = pd.Series(SDSD)
train_features['rmssd'] = pd.Series(RMSSD)

meanhr = []
minhr = []
maxhr = []
stdhr = []
IBI = []
SDNN = []
SDSD = []
RMSSD = []
pNN20 = []
pNN50 = []
for i in range(len(x_test)):
    out = ecg.ecg(signal=x_test[i][10:], sampling_rate=300, show=False)
    meanhr.append(np.mean(out[6]))
    minhr.append(min(out[6]))
    maxhr.append(max(out[6]))
    stdhr.append(np.std(out[6]))
    calc_RR()
    IBI.append(measures['ibi'])
    # pNN20.append(measures['pnn20'])
    pNN50.append(measures['pnn50'])
    SDNN.append(measures['sdnn'])
    SDSD.append(measures['sdsd'])
    RMSSD.append(measures['rmssd'])
    print(str(i))
test_features = pd.DataFrame()
test_features['heart_rate'] = pd.Series(meanhr)
test_features['minhr'] = pd.Series(minhr)
test_features['maxhr'] = pd.Series(maxhr)
test_features['stdhr'] = pd.Series(stdhr)
test_features['ibi'] = pd.Series(IBI)
# test_features['pnn20'] = pd.Series(pNN20)
test_features['pnn50'] = pd.Series(pNN50)
test_features['sdnn'] = pd.Series(SDNN)
test_features['sdsd'] = pd.Series(SDSD)
test_features['rmssd'] = pd.Series(RMSSD)

xtrain, xtest, labels, y_test = train_test_split(train_features.iloc[:, 1:], y_train['y'], test_size=0.1,
                                                 random_state=0)

from sklearn.svm import SVC

clf = SVC(decision_function_shape='ovo', gamma='scale', kernel='rbf', class_weight='balanced')
clf.fit(xtrain, labels)
clf.fit(train_features, y_train)
prediction = clf.predict(test_features)
id = pd.DataFrame({'id': range(len(test_features))})
id['y'] = prediction
id.to_csv('prediction3.csv', index=False)
f1_score(y_test, prediction, average='micro')

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
xtrain, xtest, labels, y_test = train_test_split(train_features, y_train, test_size=0.1, random_state=0)
clf.fit(xtrain, labels)
prediction = clf.predict(xtest)
f1_score(y_test, prediction, average='micro')

np.any(np.isnan(xtrain))
np.all(np.isfinite(xtrain))
xtrain.isnull().any()
xtrain[xtrain.isnull().values == True]

out = ecg.ecg(signal=x_train[0][10:], sampling_rate=300, show=True)
out[2]
np.mean(out[6])
plt.plot(x_train[4467][10:1000])

test_features.isnull().any()
train_features[train_features.isnull().values == True]
train_features = train_features.drop([2719, 3178, 4299, 4467])
y_train = y_train['y'].drop([2719, 3178, 4299, 4467])

indexes = [2719, 3178, 4299, 4467]
for index in sorted(indexes, reverse=True):
    del x_train[index]
for index in sorted(indexes, reverse=True):
    del y_train[index]

del x_test[31]
del x_test[2490]

out = ecg.ecg(signal=x_test[2490][10:], sampling_rate=300, show=True)
del x_train[2719]
del x_train[3177]
del x_train[4297]
del x_train[4464]