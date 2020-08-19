# to visualisation the gaze/pose in video, using the following command in command line
'''
sudo /media/ytikewk/76627FE2627FA591/dataset/OpenFace-master/build/bin
/FeatureExtraction -f /home/ytikewk/python_project/daisee_detect
/Validation/400022/4000221001/4000221001.avi
-out_dir /home/ytikewk/python_project -pose -vis-track
'''


# to output pose & gaze

# if only need pose / gaze, using the following func
# (in the same dirc)

# FeatureExtraction_pose
# FeatureExtraction_gaze


#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob, os, re
import xarray as xr
from  imblearn.ensemble import *
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix,make_scorer
from collections import Counter
import pydotplus
# from sklearn.datasets import load_iris
from sklearn import tree
#from getData import get_test_data,get_train_data,get_val_data
#from resultAnalysis import get_fault_instance,model_scores,all_datas
#from  trail import all_datas,model_scores,get_fault_instance
os.chdir('/home/ytikewk/python_project/daisee_detect')


# %%
###################################################
#  get data
###################################################

def get_val_data():
    val_label_full = pd.read_csv('/home/ytikewk/python_project/daisee_detect/ValidationLabels.csv')
    VAL_Y_engagement = val_label_full.iloc[:, 2]
    VAL_Y_engagement = np.array(VAL_Y_engagement).reshape(len(VAL_Y_engagement))
    val_label = np.zeros(len(VAL_Y_engagement))
    for i in range(len(VAL_Y_engagement)):
        if VAL_Y_engagement[i] > 1:
            val_label[i] = 1
        else:
            val_label[i] = 0
    #val_label = np.array(pd.DataFrame(index = val_label_full.iloc[:,0],data = val_label)).T
    val_data = pd.read_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/'
                                 'DataSet/val_data_process/data_frame_val_gp+au.csv', index_col=0)
    val_data = val_data.iloc[:, 42:]
    val_data_1 = val_data.iloc[:, -18 * 7:]
    val_data_2 = val_data.iloc[:,:-35*7]
    val_data = pd.concat([val_data_2,val_data_1],axis=1)


    return val_data,val_label



def get_train_data():
    train_label_full = pd.read_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/Labels/TrainLabels.csv')

    # del 2100552061.avi cause it not exist
    train_label_full = train_label_full[~ train_label_full['ClipID'].str.contains('2100552061.avi')]
    Y = train_label_full.iloc[:, 2]
    Y = np.array(Y).reshape(len(Y))
    train_label = np.zeros(len(Y))
    for i in range(len(Y)):
        if Y[i] > 1:
            train_label[i] = 1
        else:
            train_label[i] = 0                           # get binary class label Y_2

    #train_label = train_label

    train_data = pd.read_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/'
                             'DataSet/train_data_process/data_frame_gp+au.csv',index_col=0)

    train_data = train_data.iloc[:, 42:]
    train_data_1 = train_data.iloc[:, -18 * 7:]

    train_data_2 = train_data.iloc[:,:-35*7]
    train_data = pd.concat([train_data_2,train_data_1],axis=1)
    return train_data,train_label



def get_test_data():
    test_label_full = pd.read_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE'
                                  '/Labels/TestLabels.csv')

    Y = test_label_full.iloc[:, 2]
    Y = np.array(Y).reshape(len(Y))
    Y_2 = np.zeros(len(Y))
    for i in range(len(Y)):
        if Y[i] > 1:
            Y_2[i] = 1
        else:
            Y_2[i] = 0                           # get binary class label Y_2
    test_label = Y_2
    test_data = pd.read_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/DataSet/'
                            'test_data_process/test_data.csv',index_col=0)
    test_data_1 = test_data.iloc[:, -18 * 7:]
    test_data_2 = test_data.iloc[:,:-35*7]
    test_data = pd.concat([test_data_2,test_data_1],axis=1)
    return test_data,test_label



######################################################################################3
# result analysis
#######################################################################################

def all_data(train_data,train_label,test_data,test_label,val_data,val_label):
    full_data = pd.concat([train_data, test_data, val_data])
    full_label = np.append(train_label, test_label)
    full_label = np.append(full_label, val_label)
    return full_data,full_label


def get_fault_instance(val_y_predict,val_label):
    val_fault_rf = [] # predict engage but label is disengage
    val_fault_rf_inv = [] # predict disengage but label is engage
    val_correct_result = [] # correctly predict
    # find the .avi which always incorrect
    for i in range(len(val_y_predict)):
        if val_y_predict[i] - val_label[i] >= 1:  #
            val_fault_rf.append(i)
        if val_y_predict[i] - val_label[i] <= -1:
            val_fault_rf_inv.append(i)
        # if val_y_predict[i]==0 and val_label[i] == 0:
        #     val_correct_result.append(i)
    return val_fault_rf,val_fault_rf_inv#,val_correct_result


def model_score(brf,X_train,y_train,data,label):
    brf.fit(X_train,y_train)
    predict = brf.predict(data)
    #co = []
    co = confusion_matrix(predict,label)
    print(co)
    print("true disengage: " , co[0,0]/(co[0,0]+co[1,0]))
    print("true engage: " , co[1,1]/(co[0,1]+co[1,1]))
    print('accuracy: ', (co[0,0]+co[1,1])/(co[0,0]+co[1,0]+co[0,1]+co[1,1]))
    return predict,co

# define eng/diseng score func (recall / true positive rate)
def score_engage(label,predict):
    co = confusion_matrix(predict, label)
    score = co[1, 1] / (co[0, 1] + co[1, 1])

    return score

def score_disengage(label,predict):
    co = confusion_matrix(predict, label)
    score = co[0,0]/(co[0,0]+co[1,0])
    # print(Counter(predict))
    # print(Counter(label))
    # print(co)
    return score


# split the data for cross-validation(k = 10)
def data_split_10(full_data,full_label):
    cv_data = []
    cv_label = []
    d1,d2,l1,l2 = train_test_split(full_data,full_label,test_size=0.5) # half

    d3,d4,l3,l4 = train_test_split(d1,l1,test_size=0.2) # 分离出d4,l4
    d5,d6,l5,l6 = train_test_split(d3,l3,test_size=0.5)
    d7,d8,l7,l8 = train_test_split(d5,l5,test_size=0.5) # 分离出d7 d8
    d9,d10,l9,l10 = train_test_split(d6,l6,test_size=0.5) # 分离出d9, d10

    dd3,dd4,ll3,ll4 = train_test_split(d2,l2,test_size=0.2) # 分离出d4,l4
    dd5,dd6,ll5,ll6 = train_test_split(dd3,ll3,test_size=0.5)
    dd7,dd8,ll7,ll8 = train_test_split(dd5,ll5,test_size=0.5) # 分离出d7 d8
    dd9,dd10,ll9,ll10 = train_test_split(dd6,ll6,test_size=0.5) # 分离出d9, d10

    cv_data.append(d4), cv_label.append(l4)
    cv_data.append(dd4), cv_label.append(ll4)
    cv_data.append(d7), cv_label.append(l7)
    cv_data.append(dd7), cv_label.append(ll7)
    cv_data.append(d8), cv_label.append(l8)
    cv_data.append(dd8), cv_label.append(ll8)
    cv_data.append(d9), cv_label.append(l9)
    cv_data.append(dd9), cv_label.append(ll9)
    cv_data.append(d10), cv_label.append(l10)
    cv_data.append(dd10), cv_label.append(ll10)

    return cv_data,cv_label


def cross_val(brf,full_data,full_label):
    cv_data,cv_label = data_split_10(full_data,full_label)

    ds = [] # disengage scores
    es = [] # engage scores

    for i in range(10):
        # c1 = cv_data[0:i]
        # c2 = cv_data[i+1:]
        # cl1 = cv_label[0:i]
        # cl2 = cv_label[i+1:]
        X_train = pd.DataFrame()
        y_train = []
        for j in range(10):
            if j != i :
                X_train = pd.concat([X_train,cv_data[j]])
                y_train = np.append(y_train,cv_label[j])
        X_test = cv_data[i]
        y_test = cv_label[i]
        brf.fit(X_train,y_train)
        predict = brf.predict(X_test)
        ds.append(score_disengage(y_test,predict))
        es.append(score_engage(y_test,predict))

    print("disengage score", sum(ds)/10)
    print("engage score", sum(es)/10)

    return ds,es


# analysis dicision path
def get_decision_path(estimator,samples,filename,full_data):
    clf = estimator
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=full_data.columns,
                                    class_names=['disengg', 'engg'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    # empty all nodes, i.e.set color to white and number of samples to zero
    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '<br/>'.join(labels))
            node.set_fillcolor('white')

    # choose the instance
    decision_paths = clf.decision_path(samples)
    label_in_path = []
    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]
            node.set_fillcolor('green')
            labels = node.get_attributes()['label'].split('<br/>')
            label_in_path.append(labels)
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))

    graph.write_png(filename)
    return label_in_path


# get disengaged instance
def get_disengaged_instance(full_label,full_data):
    disengaged_instance = []
    for n, i in enumerate(full_label):
        if i == 0:
            disengaged_instance.append(full_data.index[n])
    return disengaged_instance

def get_correct_disengage(brf,full_data,full_label):
    cv_data, cv_label = data_split_10(full_data, full_label)

    ds = []  # disengage scores
    es = []  # engage scores
    disengaged_correct = []
    engaged_correct = []
    for i in tqdm(range(10)):
        X_train = pd.DataFrame()
        y_train = []
        for j in range(10):
            if j != i:
                X_train = pd.concat([X_train, cv_data[j]])
                y_train = np.append(y_train, cv_label[j])

        X_test = cv_data[i]
        y_test = cv_label[i]

        brf.fit(X_train, y_train)
        # print(i)
        predict = brf.predict(X_test)
        # print(confusion_matrix(predict, y_test))
        # print(score_disengage(y_test, predict))
        # print(Counter(y_test))
        # print(Counter(predict))
        # get disengaged prediction
        disengaged_predict = []
        # print(predict)
        for n, m in enumerate(predict):
            if m == 0 and y_test[n] == 0:
                # print(m)
                disengaged_predict.append(X_test.index[n])  # true positive one

            if m == 1 and y_test[n] == 1:
                engaged_correct.append(X_test.index[n])
        # get correct one
        # correct = np.intersect1d(disengaged_predict, disengaged_instance)
        disengaged_correct = np.append(disengaged_correct, disengaged_predict)


    return disengaged_correct,engaged_correct

# get the instance path
def clasBasedOnFeature(brf,full_data,disengaged_correct):
    # function trail
    nonFirst = []
    gazeFirst = []
    poseFirst = []
    AUFirst = []
    for a, b  in tqdm(enumerate(disengaged_correct)):
        # choose sample as the instance

        # file = "1100011004.avi"
        trail = full_data.loc[b, :]
        # brf.predict(np.array(trail).reshape(1, -1))

        #########################################################
        samples = np.array(trail).reshape(1, -1)
        #########################################################


        label_only_in_path_have = []
        label_only_in_path_dont_have = []

        for i in range(len(brf)):
            estimator = brf.estimators_[i]
            label_in_path = []

            # get the feature
            features = estimator.tree_.feature

            # get the path for the instance
            # path = estimator.decision_path(np.array(data_frame_val)[0,:].reshape(1,-1))
            # estimator.predict(np.array(X_test)[0,:].reshape(1,-1))
            path = estimator.decision_path(samples)
            path_array = path.toarray()

            # connect feature and path
            pf_num = []
            for n, i in enumerate(path_array[0]):
                if i == 1 and features[n] != -2:
                    pf_num.append(features[n])

            pf_label = []
            for n, i in enumerate(pf_num):
                pf_label.append(full_data.columns[i])

            label_in_path = pf_label
            # label including std/mean/min/max...
            for m in range(len(label_in_path)):
                # bund = label_in_path[m][0].rfind(' &le')
                label_only_in_path_have.append(label_in_path[m])

            # label dont have std/mean/min/max...
            for m in range(len(label_in_path)):
                bund = label_in_path[m].rfind('_')
                label_only_in_path_dont_have.append(label_in_path[m][:bund])


        detail_1 = Counter(label_only_in_path_have)
        x_haveMean = detail_1.most_common(n=10)

        detail = Counter(label_only_in_path_dont_have)
        x_dontHaveMean = detail.most_common(n=10)

        aa = 0

        if x_haveMean[0][0][1:5] == 'gaze':
            aa = 1
            # print('\n gaze: '+full_data.index[a])
            gazeFirst.append(disengaged_correct[a])

        if x_haveMean[0][0][1:5] == 'pose':
            aa = 1
            # print('\n pose: '+full_data.index[a])
            poseFirst.append(disengaged_correct[a])

        if x_haveMean[0][0][1:5] == 'AU45':
            aa = 1
            # print('\n AU: '+full_data.index[a])
            AUFirst.append(disengaged_correct[a])

        if aa == 0:
            nonFirst.append(disengaged_correct[a])

    return AUFirst,gazeFirst,poseFirst,nonFirst



def clasBasedOnFeature_donHaveMean(brf,full_data,disengaged_correct):
    # function trail

    nonFirst = []
    gazeFirst = []
    poseFirst = []
    AUFirst = []
    for a, b  in tqdm(enumerate(disengaged_correct)):
        # choose sample as the instance

        # file = "1100011004.avi"
        trail = full_data.loc[b, :]
        # brf.predict(np.array(trail).reshape(1, -1))

        #########################################################
        samples = np.array(trail).reshape(1, -1)
        #########################################################


        label_only_in_path_have = []
        label_only_in_path_dont_have = []

        for i in range(len(brf)):
            estimator = brf.estimators_[i]
            label_in_path = []

            # get the feature
            features = estimator.tree_.feature

            # get the path for the instance
            # path = estimator.decision_path(np.array(data_frame_val)[0,:].reshape(1,-1))
            # estimator.predict(np.array(X_test)[0,:].reshape(1,-1))
            path = estimator.decision_path(samples)
            path_array = path.toarray()

            # connect feature and path
            pf_num = []
            for n, i in enumerate(path_array[0]):
                if i == 1 and features[n] != -2:
                    pf_num.append(features[n])

            pf_label = []
            for n, i in enumerate(pf_num):
                pf_label.append(full_data.columns[i])

            label_in_path = pf_label
            # label including std/mean/min/max...
            for m in range(len(label_in_path)):
                # bund = label_in_path[m][0].rfind(' &le')
                label_only_in_path_have.append(label_in_path[m])

            # label dont have std/mean/min/max...
            for m in range(len(label_in_path)):
                bund = label_in_path[m].rfind('_')
                label_only_in_path_dont_have.append(label_in_path[m][:bund])


        detail_1 = Counter(label_only_in_path_have)
        x_haveMean = detail_1.most_common(n=10)

        detail = Counter(label_only_in_path_dont_have)
        x_dontHaveMean = detail.most_common(n=10)

        aa = 0

        if x_dontHaveMean[0][0][1:5] == 'gaze':
            aa = 1
            # print('\n gaze: '+full_data.index[a])
            gazeFirst.append(disengaged_correct[a])

        if x_dontHaveMean[0][0][1:5] == 'pose':
            aa = 1
            # print('\n pose: '+full_data.index[a])
            poseFirst.append(disengaged_correct[a])

        if x_dontHaveMean[0][0][1:5] == 'AU45':
            aa = 1
            # print('\n AU: '+full_data.index[a])
            AUFirst.append(disengaged_correct[a])

        if aa == 0:
            nonFirst.append(disengaged_correct[a])

    return AUFirst,gazeFirst,poseFirst,nonFirst


# %%
def make_2Label(label):
    l2 = np.zeros(len(label))
    for i in range(len(label)):
        if label[i] > 1:
            l2[i] = 1
        else:
            l2[i] = 0

    return l2



# %%
# get label
label4 = np.load('/home/ytikewk/python_project/daisee_detect/selected4pData_target.npy')
label2= make_2Label(label4)
print(Counter(label4))
print(Counter(label2))

# %% 19.00 in 14/8
# this data including 20 openpose point, openface,
# face landmark (different part shape,location)
data = pd.read_csv('/home/ytikewk/python_project/daisee_detect/dataprocess/finall_process.csv',index_col=0)
full_data = data
full_label = label2
# %%using finall process
data = pd.read_csv('/home/ytikewk/python_project/daisee_detect/final_process.csv',index_col=0)
full_data = data
full_label = label2
# %% get the split data(only face)
val_data,val_label = get_val_data()
train_data,train_label = get_train_data()
test_data,test_label = get_test_data()
test_data = pd.DataFrame([])
test_label = []
# fusion all the data
full_data, label2 = all_data(train_data,train_label,test_data,test_label,val_data,val_label)
full_label = label2
# %%
data = np.load('/home/ytikewk/python_project/daisee_detect/selected4pData_face.npy')

# %%-1 4pData + drop0
data = pd.read_csv('/home/ytikewk/python_project/daisee_detect/selected4pData_drop0.csv',index_col=0)
label = np.load('/home/ytikewk/python_project/daisee_detect/selected4pData_target.npy')

full_data = data


# %% normalisation (alternative)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data_norm = scaler.transform(data)

data_norm = pd.DataFrame(data_norm,index=data.index,columns=data.columns)
data_norm.to_csv('/home/ytikewk/python_project/daisee_detect/selected4pData_drop0_norm.csv')


# %%
# split data
# data_norm = pd.read_csv('/home/ytikewk/python_project/daisee_detect/selected4pData_drop0_norm.csv',index_col=0)
X_train,X_test,y_train,y_test = train_test_split(full_data,label2,test_size=0.25)
print(Counter(y_test))
print(Counter(y_train))
# %%
X_train,X_test,y_train,y_test = train_test_split(full_data,label4,test_size=0.25)

# os.chdir('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/tree_png')
print(Counter(y_test))
print(Counter(y_train))
# %%
###########################################
# result analysis
##########################################

#different method

print('bbc')
bbc = BalancedBaggingClassifier()
bbc_ds,bbc_es = cross_val(bbc,full_data,full_label)

print('brf')
brf = BalancedRandomForestClassifier()
brf_ds,brf_es = cross_val(brf,full_data,full_label)

print('eec')
eec = EasyEnsembleClassifier()
eec_ds,eec_es = cross_val(eec,full_data,full_label)

print('rbc')
rbc = RUSBoostClassifier()
rbc_ds,rbc_es = cross_val(rbc,full_data,full_label)

###################################
# balanced random fores have a better performance (0.7 vs 0.69)

# bbc
# disengage score 0.6290352664658471
# engage score 0.7269668354164707

# brf
# disengage score 0.7085299406791943
# engage score 0.6988126179654749

# eec
# disengage score 0.6612142411270588
# engage score 0.6961831261705418

# rbc
# disengage score 0.47591961405630095
# engage score 0.7603004021060058

###################################
# %%
# try to using oversampling and downsampling, but the result is not so good
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,RandomOverSampler
from imblearn.over_sampling import SMOTENC,SVMSMOTE,ADASYN,KMeansSMOTE
from sklearn.metrics import confusion_matrix

assert data_norm.shape[1] == 847
assert data_norm.shape[0] == len(full_label) #8921


xx_train,xx_test,yy_train,yy_test = train_test_split(data_norm,full_label,test_size=0.25)

################################
oversamp_strategy = BorderlineSMOTE(n_jobs=8)
xx,yy = oversamp_strategy.fit_resample(xx_train,yy_train)
print('strategy finish')
################################


rfc = RandomForestClassifier(n_jobs=8)
rfc.fit(xx_train,yy_train)
predx = rfc.predict(xx_test)
# print(score_engage(yy_test,predx))
# print(score_disengage(yy_test,predx))
print(confusion_matrix(yy_test,predx))


# %% finally using BRF
brf = BalancedRandomForestClassifier(n_jobs=9,n_estimators=500,oob_score=True,
                                     min_samples_split=7)
brf.fit(X_train,y_train)
predict_brf = brf.predict(X_test)

predict_prob_brf = brf.predict_proba(X_test)

print(confusion_matrix(y_test,predict_brf))
print(score_disengage(y_test,predict_brf))
print(score_engage(y_test,predict_brf))
# %%
# using lime to explain
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(np.array(X_test),class_names=['disengaged','engaged'],
                                 feature_names=data.columns)


idx = 88
exp = explainer.explain_instance(X_test[idx], predict_prob_brf[idx], num_features=6)
print('Document id: %d' % idx)
# print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
# print('True class: %s' % class_names[newsgroups_test.target[idx]])

# %%
# print the result
from sklearn.metrics import precision_recall_fscore_support as scores
y_2 = make_2Label(y_test)
p_2 = make_2Label(predict_brf)


precision, recall, fscore, support = scores(y_2, p_2)
# precision, recall, fscore, support = scores(y_test, predict_brf)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# %%
bbc = BalancedBaggingClassifier()
bbc.fit(X_train,y_train)
predict_bbc = bbc.predict(X_test)
predict_prob_bbc = bbc.predict_proba(X_test)
score_disengage(y_test,predict_bbc)

# %%
# plot the ROC compare figure
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_roc(labels, predict_brf, predict_eec,predict_bbc, predict_rus):
    false_positive_rate_brf,true_positive_rate_brf,thresholds_brf=roc_curve(labels, predict_brf)
    false_positive_rate_eec,true_positive_rate_eec,thresholds_eec=roc_curve(labels, predict_eec)
    false_positive_rate_bbc,true_positive_rate_bbc,thresholds_bbc=roc_curve(labels, predict_bbc)
    false_positive_rate_rus,true_positive_rate_rus,thresholds_rus=roc_curve(labels, predict_rus)

    roc_auc_brf = auc(false_positive_rate_brf, true_positive_rate_brf)
    roc_auc_eec = auc(false_positive_rate_eec, true_positive_rate_eec)
    roc_auc_bbc = auc(false_positive_rate_bbc, true_positive_rate_bbc)
    roc_auc_rus = auc(false_positive_rate_rus, true_positive_rate_rus)

    plt.figure()
    plt.title('ROC')
    plt.plot(false_positive_rate_brf, true_positive_rate_brf,'b',label='AUC_brf = %0.4f'% roc_auc_brf)
    plt.plot(false_positive_rate_eec, true_positive_rate_eec, 'r', label='AUC_eec = %0.4f' % roc_auc_eec)
    plt.plot(false_positive_rate_bbc, true_positive_rate_bbc, 'g', label='AUC_bbc = %0.4f' % roc_auc_bbc)
    plt.plot(false_positive_rate_rus, true_positive_rate_rus, 'y', label='AUC_rus = %0.4f' % roc_auc_rus)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')

plot_roc(y_test,predict_prob_brf[:,1],predict_prob_eec[:,1],predict_prob_bbc[:,1],predict_prob_rus[:,1])




# %%% predict specific instance
file = "1100011004.avi"
trail = full_data.loc[file,:]
brf.predict(np.array(trail).reshape(1,-1))


#########################################
# analysis the classifier feature importance (which feature(gaze) is more important
# using count value and importance value
# #######################################
# %%
# Count
# find the feature importance(if have 5 feature bigger than 0.01)
data_frame = X_test
importance = pd.DataFrame(brf.feature_importances_,index=data_frame.columns)

# nonzero importance describe (mean is 0.005)
importance[importance!=0].dropna().describe()

all_0_feature = []
for i in range(26):
    all_0 = 0
    for j in range(7):
        if np.array(importance)[7*i + j] >= 0.001:

            all_0 = all_0 + 1
    if all_0 > 5:
        all_0_feature.append(importance.index[7*i])

all_0_feature

# [' gaze_angle_x_mean',
#  ' gaze_angle_y_mean',
#  ' pose_Tx_mean',
#  ' pose_Tz_mean',
#  ' pose_Rx_mean',
#  ' pose_Ry_mean',cla
#  ' pose_Rz_mean']


# %%
# using value
importance_t = brf.feature_importances_
indices = np.argsort(importance_t)[::-1]
for f in range(len(X_train.columns)):
    if f < 10: # select first 10
        print("%2d) %-*s %f" % (f + 1, 30, X_train.columns[indices[f]], importance_t[indices[f]]))




#  1)  AU45_c_std                    0.025551
#  2)  gaze_angle_y_std              0.019705
#  3)  AU45_c_mean                   0.018768
#  4)  pose_Ry_std                   0.017004
#  5)  pose_Tx_min                   0.016761
#  6)  pose_Rx_min                   0.015519
#  7)  pose_Tz_50%                   0.015459
#  8)  pose_Rz_25%                   0.014300
#  9)  pose_Tz_min                   0.014080
# 10)  pose_Tz_std                   0.013974



###################################################
#
###################################################
# %% select instance which can be correctly classified

disengaged_instance = get_disengaged_instance(full_label,full_data)
disengaged_correct,engaged_correct = get_correct_disengage(brf,full_data,full_label)



# %%
# find the correct one in test set
in_test = []
in_test_indexNum = []
for n,i in enumerate(X_test.index):
    if X_test.index[n] in disengaged_correct:
        in_test.append(X_test.index[n])
        in_test_indexNum.append(n)

in_test_indexNum




# %% get the instance path
# function trail

file = '4110212062.avi'


trail = full_data.loc[file, :]
# brf.predict(np.array(trail).reshape(1, -1))

#########################################################
samples = np.array(trail).reshape(1, -1)
#########################################################


label_only_in_path_have = []
label_only_in_path_dont_have = []

for i in range(len(brf)):
    estimator = brf.estimators_[i]
    label_in_path = []

    # get the feature
    features = estimator.tree_.feature

    # get the path for the instance
    # path = estimator.decision_path(np.array(data_frame_val)[0,:].reshape(1,-1))
    # estimator.predict(np.array(X_test)[0,:].reshape(1,-1))
    path = estimator.decision_path(samples)
    path_array = path.toarray()

    # connect feature and path
    pf_num = []
    for n, i in enumerate(path_array[0]):
        if i == 1 and features[n] != -2:
            pf_num.append(features[n])

    pf_label = []
    for n, i in enumerate(pf_num):
        pf_label.append(X_test.columns[i])

    label_in_path = pf_label
    # label including std/mean/min/max...
    for m in range(len(label_in_path)):
        # bund = label_in_path[m][0].rfind(' &le')
        label_only_in_path_have.append(label_in_path[m])

    # label dont have std/mean/min/max...
    for m in range(len(label_in_path)):
        bund = label_in_path[m].rfind('_')
        label_only_in_path_dont_have.append(label_in_path[m][:bund])


detail_1 = Counter(label_only_in_path_have)
x_haveMean = detail_1.most_common(n=3)

detail = Counter(label_only_in_path_dont_have)
x_dontHaveMean = detail.most_common(n=3)

print(x_haveMean)
print(x_dontHaveMean)

if x_haveMean[0][0][1:3] == 'AU45':
    aa = 1
    # AUFirst.append(full_data.index[a])

# if aa == 0:
    # nonFirst.append(full_data.index[a])

# %% process all the data to find the influential feature for each sample
AUFirst,gazeFirst,poseFirst,nonFirst = clasBasedOnFeature(brf,
                                                          full_data,
                                                          disengaged_correct)

AUFirst1,gazeFirst1,poseFirst1,nonFirst1 = clasBasedOnFeature(brf,
                                                          full_data,
                                                          engaged_correct)







# %% output one tree classifier as .dot file
os.chdir('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/tree_png')
estimator = brf.estimators_[99] # select one tree from forest
onetree = tree.export_graphviz(estimator,
                     out_file=None,
                     feature_names = X_train.columns,
                     class_names= ['disengg','engg'],
                     filled = True)
graph = pydotplus.graph_from_dot_data(onetree)
graph.write_png("tree.png")

# dot file process
################################
# dot -Tpng tree.dot -o tree.png　（in command line）
###############################＃

# %% output one tree classifier decision path
estimator = brf.estimators_[99] # select one tree from forest
samples = np.array(X_test)[104,:].reshape(1,-1)
filename = "tree_path.png"

label_in_path = get_decision_path(estimator,samples,filename,full_data)



# %%

# %% binary classifier search
dis = make_scorer(score_disengage, greater_is_better=True)

param_test1 = {'n_estimators':range(1,300,10),
               'max_depth': range(3, 150, 2),
               'min_samples_split': range(5, 500, 20),
                #'min_samples_leaf': range(5, 500, 20),
                #'min_weight_fraction_leaf': np.arange(0,0.5,0.02),
                #'max_features': range(3, 51, 2),
               }
gsearch1 = RandomizedSearchCV(estimator=BalancedRandomForestClassifier(
                                                               #n_estimators=600,
                                                               n_jobs=2,
                                                               random_state=5
                                                               ),
                        #param_distributions=param_test1,
                        param_distributions=param_test1,
                        n_jobs=8,scoring=dis)
gsearch1.fit(full_data, full_label)
fault_1 = []
fault_1_inv = []
correct_result = []
gsearch1.best_params_, gsearch1.best_score_
