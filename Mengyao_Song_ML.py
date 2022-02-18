# standard packages used to handle files
import sys
import os 
import glob
import time

# commonly used library for data manipulation
import pandas as pd

# numerical
import numpy as np

# machine learning library
import sklearn
import sklearn.preprocessing

# used to serialize python objects to disk and load them back to memory
import pickle

# plotting
import matplotlib.pyplot as plt

# helper functions
import helpers

# specific helper functions for feature extraction

import features
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
# tell matplotlib that we plot in a notebook
# %matplotlib notebook
# %matplotlib inline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier 
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

dataset_path = 'E:/py/images/'
output_path = './'

dataset_path_train = os.path.join(dataset_path, 'train')
dataset_path_test = os.path.join(dataset_path, 'test')

features_path = os.path.join(output_path, 'features')
features_path_train = os.path.join(features_path, 'train')
features_path_test = os.path.join(features_path, 'test')

prediction_path = os.path.join(output_path, 'predictions')

# filepatterns to write out features
filepattern_descriptor_train = os.path.join(features_path_train, 'train_features_{}.pkl')
filepattern_descriptor_test = os.path.join(features_path_test, 'test_features_{}.pkl')

# create paths in case they don't exist:
helpers.createPath(features_path)
helpers.createPath(features_path_train)
helpers.createPath(features_path_test)
helpers.createPath(prediction_path)

folder_paths = glob.glob(os.path.join(dataset_path_train,'*'))
label_strings = np.sort(np.array([os.path.basename(path) for path in folder_paths]))
num_classes = label_strings.shape[0]

train_paths = dict((label_string, helpers.getImgPaths(os.path.join(dataset_path_train, label_string))) for label_string in label_strings)

test_paths = helpers.getImgPaths(dataset_path_test)

descriptor_dict = {
    'daisy': features.extractDAISYCallback, # SIFT replacement, very fast, can be computed dense if necessary
    'orb': features.extractORBCallback, # another fast SIFT replacement, oriented BRIEF w. FAST keypoints  
    'freak': features.extractFREAKCallback, # biologically motivated descriptor
    'lucid': features.extractLUCIDCallback,  
    'vgg': features.extractVGGCallback, # Trained as proposed by VGG lab, don't confuse it with VGG-Net features
    'boost_desc': features.extractBoostDescCallback, # Image descriptor learned with boosting
}
if features.checkForSIFT():
    descriptor_dict['sift'] = features.extractSIFTCallback # One descriptor to rule them all
if features.checkForSURF():
    descriptor_dict['surf'] = features.extractSURFCallback # Another very good descriptor

def tunBookSize(desired):
    descriptor_desired = desired
    with open(filepattern_descriptor_train.format(descriptor_desired), 'rb') as pkl_file_train:
        train_features_from_pkl = pickle.load(pkl_file_train)
    with open(filepattern_descriptor_test.format(descriptor_desired), 'rb') as pkl_file_test:
        test_features_from_pkl = pickle.load(pkl_file_test)       
    size_range = range(2,12)
    num_folds=10
    kf = KFold(n_splits=num_folds, shuffle=True)
    cv_scores = []
    for n in size_range:
        codebook_size = n*100
        clustered_codebook = helpers.createCodebook(train_features_from_pkl, codebook_size=codebook_size)
    # encode all train images 
        train_data = []
        train_labels = []
    
        for image_features in train_features_from_pkl:
            bow_feature_vector = helpers.encodeImage(image_features.data, clustered_codebook)
            train_data.append(bow_feature_vector)
            train_labels.append(image_features.label)
        
        label_encoder = sklearn.preprocessing.LabelEncoder()
        sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        scores = cross_val_score(sgd_clf,train_data,train_labels,cv=kf,scoring='accuracy')
        cv_scores.append(scores.mean()) 

    plt.plot(size_range,cv_scores)
    plt.xlabel('code book size')
    plt.ylabel('Accuracy')		#select the best hyperparameter
    plt.show()
    return train_features_from_pkl,test_features_from_pkl

tunBookSize('daisy')
tunBookSize('vgg')
tunBookSize('sift')

def getFeatures(desired):
    descriptor_desired = 'daisy'   
    with open(filepattern_descriptor_train.format(descriptor_desired), 'rb') as pkl_file_train:
        train_features_from_pkl = pickle.load(pkl_file_train)
    print('Number of encoded train images: {}'.format(len(train_features_from_pkl)))
    with open(filepattern_descriptor_test.format(descriptor_desired), 'rb') as pkl_file_test:
        test_features_from_pkl = pickle.load(pkl_file_test)       
    print('Number of encoded test images: {}'.format(len(test_features_from_pkl)))
    return train_features_from_pkl,test_features_from_pkl

def learnBook(desired,bookSize):
    codebookSize = bookSize
    train_features_from_pkl, test_features_from_pkl = getFeatures(desired)

    train_data = []
    train_labels = []

    clustered_codebook = helpers.createCodebook(train_features_from_pkl, codebook_size=codebookSize)
    
    for image_features in train_features_from_pkl:
        bow_feature_vector = helpers.encodeImage(image_features.data, clustered_codebook)
        train_data.append(bow_feature_vector)
        train_labels.append(image_features.label)
        
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(label_strings)
    train_labels = label_encoder.transform(train_labels)   
    train_D = np.stack(train_data,axis=0)
    train_L = np.stack(train_labels, axis=0)
    return clustered_codebook,train_features_from_pkl,test_features_from_pkl,train_D,train_L

#Split into train and validation sets
def splitData(train_D,train_L):
    X_train, X_val, y_train, y_val = train_test_split(train_D, 
                                                    train_L, 
                                                    train_size = 0.7, 
                                                    random_state=42)
    return X_train,X_val,y_train,y_val

# Tun params for sgd classifier
def tunParamSelection(X_train,y_train):
    pipeline = Pipeline([('scaler',StandardScaler()),('pca', PCA()), ('sgd', SGDClassifier(tol=1e-4))])
    tuned_parameters = {
        'pca__n_components': [2],
        'sgd__loss': ['log'],
        'sgd__penalty': ['elasticnet'],
        'sgd__alpha': [10 ** x for x in range(-6, 1)],
        'sgd__l1_ratio': [ 0.05, 0.1,  0.5, 1],
        'sgd__max_iter': [3000,5000]
    }
    
    kf = KFold(n_splits=10, shuffle=True)
    model = GridSearchCV(pipeline, tuned_parameters, scoring = 'accuracy', cv=kf)
    model.fit(X_train, y_train)
    # scores = ['accuracy','precision', 'recall']
    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
    return model.best_params_

# Fit sgd classifier by the tuned params
def findSGD(X_train, y_train):
    params = tunParamSelection(X_train,y_train)
    l1_param, alpha_param,iter_param = params.get("l1_ratio"), params.get("alpha"),params.get("max_iter")
    sgd_clf = SGDClassifier(tol=1e-4,max_iter=iter_param,alpha=alpha_param, l1_ratio=l1_param, loss="log", penalty="elasticnet")
    sgd_clf.fit(X_train,y_train)
    print('Fitting is done : )')
    return sgd_clf

# Tun params for svc classifier
def tunParamSelectionSVC(X_train,y_train):
    pipeline = Pipeline([('scaler',StandardScaler()),('pca', PCA()), ('supvm', SVC(kernel="rbf"))])
    tuned_parameters = {
        'pca__n_components': [2],
        'supvm__C': [0.1, 1, 10,30,100,500,1000],
        'supvm__gamma':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50]
    }
    
    for cv in tqdm(range(4,6)):
        model = GridSearchCV(pipeline, param_grid=tuned_parameters, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(model.best_params_)
    return model.best_params_   

# Fit svc classifier by the tuned params
def findSVC(X_train, y_train):
    params = tunParamSelectionSVC(X_train,y_train)
    pca_n_components, svm_gamma,svm_C = params.get("pca__n_components"), params.get("supvm__gamma"),params.get("supvm__C")
    svc_clf = SVC(kernel="rbf", C=svm_C,gamma = svm_gamma) # ,probability=True
    svc_clf.fit(X_train,y_train)
    print('Fitting is done : )')
    return svc_clf

clustered_codebook,train_features_from_pkl,test_features_from_pkl,train_D,train_L = learnBook('daisy',1000)

# scale and pca
scaler1 = StandardScaler()
train_D_scaled = scaler1.fit(train_D).transform(train_D)
pca1 = PCA(n_components = 4)
train_D_scaled_pca = pca1.fit(train_D_scaled).transform(train_D_scaled)

test_data = []

for image_features in test_features_from_pkl:
    bow_feature_vector = helpers.encodeImage(image_features.data, clustered_codebook)
    test_data.append(bow_feature_vector)
    
test_D = np.stack(test_data,axis=0)
test_D_fea = pca1.fit(test_D).transform(test_D)
test_D_scaled = scaler1.fit(test_D).transform(test_D)
test_D_scaled_pca = pca1.fit(test_D_scaled).transform(test_D_scaled)

X_train, X_val,y_train,y_val = splitData(train_D_scaled_pca,train_L)
feat_var = np.var(train_D_scaled_pca,axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))
print("Variance Ratio of the 4 principal components analysis: ", feat_var_rat)

# do prediction
svc_clf = findSVC(X_train, y_train)
y_pred = svc_clf.predict_proba(test_D_scaled_pca)
print('SVC Classification is done, waiting to generate the result...')
helpers.writePredictionsToCsv(y_pred, 'helper_out2.csv', label_strings)

sgd_clf = findSGD(X_train,y_train)
y_pred_sgd = sgd_clf.predict_proba(test_D_scaled_pca)
print('SGD Classification is done, waiting to generate the result...')
helpers.writePredictionsToCsv(y_pred_sgd, 'helper_out_sgd.csv', label_strings)