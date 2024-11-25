import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, balanced_accuracy_score

from source.preprocessing.preprocessing_helper import get_egoexo4d_takes

# Get takes
with open('./configs/preprocessing/config_preprocessing_egoexo4d.yml', 'r') as yamlfile:
    configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
takes = get_egoexo4d_takes(configs['exclusion_list'])

# Get labels
label_path = '/data/bjbraun/Projects/TimeSformer/egoexo4d/ego/splits/'
train_split = pd.read_csv(label_path + 'train.csv', sep=' ', header=None)
val_split = pd.read_csv(label_path + 'val.csv', sep=' ', header=None)
train_split = pd.concat([train_split, val_split])
test_split = pd.read_csv(label_path + 'test.csv', sep=' ', header=None)
train_split[0] = train_split[0].apply(lambda x: x.split('/')[-5])
test_split[0] = test_split[0].apply(lambda x: x.split('/')[-5])

# Merge train and test split and add a third row that either says train or test
train_split['split'] = 'train'
test_split['split'] = 'test'
splits = pd.concat([train_split, test_split])
splits = splits.reset_index(drop=True)
splits.columns = ['take_name', 'label', 'split']
print(splits['label'].value_counts())


# Define X and y
X_all, y_all = [], []
X_train, y_train, X_test, y_test = [], [], [], []
X_train_cont, y_train_cont, X_test_cont, y_test_cont = [], [], [], []
for index, row in splits.iterrows():
    take_name = row['take_name']
    label = row['label']
    split = row['split']
    hrs_temp = np.load(f'/data/bjbraun/Projects/egoPPG/predicted_hrs/egoexo4d/{take_name}_hrs.npy')
    indexes = np.unique(hrs_temp, return_index=True)[1]
    hrs_unique = [hrs_temp[index] for index in sorted(indexes)]
    X_all.append(hrs_temp.shape[0])
    # if hrs_temp.shape[0] < 2700:
    #     continue
    if index == 52:
        print('AH')
    if split == 'train':
        X_train.append([np.nanmean(hrs_temp), np.nanstd(hrs_temp), np.nanmin(hrs_temp), np.nanmax(hrs_temp), np.nanmean(np.diff(hrs_temp))])
        y_train.append(label)
        # X_train_cont.append(hrs_unique)
        # y_train_cont.append([label] * len(hrs_unique))
    else:
        X_test.append([np.nanmean(hrs_temp), np.nanstd(hrs_temp), np.nanmin(hrs_temp), np.nanmax(hrs_temp), np.nanmean(np.diff(hrs_temp))])
        y_test.append(label)
        # X_test_cont.append(hrs_unique)
        # y_test_cont.append([label] * len(hrs_unique))
X_train = np.array(X_train).reshape(-1, 5)
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(-1, 5)
y_test = np.array(y_test)
print(np.argwhere(np.isnan(X_train)))
print(X_train.shape)
# X_train_cont = np.array(X_train_cont)
# y_train_cont = np.array(y_train_cont)
# X_test_cont = np.array(X_test_cont)
# y_test_cont = np.array(y_test_cont)

# Train classifier
classifier = RandomForestClassifier(random_state=0)
# classifier = LogisticRegression(random_state=0)
# classifier = SVC(random_state=0)
# classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate results
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
print("Accuracy:", accuracy)
print("Balanced accuracy:", balanced_accuracy)
print("Precision:", precision)
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()


print('AH')
