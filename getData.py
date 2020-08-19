import pandas as pd
import numpy as np
from tqdm import tqdm
import glob, os, re
import xarray as xr
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pandas import Series, DataFrame
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix
from collections import Counter

#%%
def process_data(label,name):
    process = []
    for idx in tqdm(range(len(label))):
        openface = pd.read_csv(label.iloc[idx, -2])
        openpose = pd.read_csv(label.iloc[idx, -1])
        pre = pd.concat([openface, openpose], axis=1)
        data = data_precess(pre)
        engg_np = np.array(data.stack())  # flatten
        process.append(engg_np)

    mul_index = data.stack().index
    split_index = []
    for i in range(len(mul_index)):
        split_index.append(str(mul_index[i][1] + '_' + mul_index[i][0]))

    file_name = np.array(label.iloc[:,0])

    da = pd.DataFrame(process, index=file_name, columns=split_index)
    da.to_csv('/home/ytikewk/python_project/daisee_detect/dataprocess/'+name+'.csv')

def get_face_feature(pre):
    fl = pre.iloc[:300, 299:299 + 68 + 68]

    face_x = fl.iloc[:, :17].T
    face_x = face_x.describe().iloc[1:3, :]
    face_y = fl.iloc[:, 68:17 + 68].T
    face_y = face_y.describe().iloc[1:3, :]

    eyebrow_l_x = fl.iloc[:, 17:22].T
    eyebrow_l_x = eyebrow_l_x.describe().iloc[1:3, :]
    eyebrow_l_y = fl.iloc[:, 17 + 68:22 + 68].T
    eyebrow_l_y = eyebrow_l_y.describe().iloc[1:3, :]
    eyebrow_r_x = fl.iloc[:, 22:27].T
    eyebrow_r_x = eyebrow_r_x.describe().iloc[1:3, :]
    eyebrow_r_y = fl.iloc[:, 22 + 68:27 + 68].T
    eyebrow_r_y = eyebrow_r_y.describe().iloc[1:3, :]

    nose_x = fl.iloc[:, 27:36].T
    nose_x = nose_x.describe().iloc[1:3, :]
    nose_y = fl.iloc[:, 27 + 68:36 + 68].T
    nose_y = nose_y.describe().iloc[1:3, :]

    eye_l_x = fl.iloc[:, 36:42].T
    eye_l_x = eye_l_x.describe().iloc[1:3, :]
    eye_l_y = fl.iloc[:, 36 + 68:42 + 68].T
    eye_l_y = eye_l_y.describe().iloc[1:3, :]

    eye_r_x = fl.iloc[:, 42:48].T
    eye_r_x = eye_r_x.describe().iloc[1:3, :]
    eye_r_y = fl.iloc[:, 42 + 68:48 + 68].T
    eye_r_y = eye_r_y.describe().iloc[1:3, :]

    mouth_x = fl.iloc[:, 48:60].T
    mouth_x = mouth_x.describe().iloc[1:3, :]
    mouth_y = fl.iloc[:, 48 + 68:60 + 68].T
    mouth_y = mouth_y.describe().iloc[1:3, :]

    face_features = pd.concat([face_x, face_y, eyebrow_l_x, eyebrow_l_y, eyebrow_r_x, eyebrow_r_y,
                               nose_x, nose_y, eye_l_x, eye_l_y, eye_r_x, eye_r_y, mouth_x, mouth_y],
                              axis=0).T
    face_features.columns = ['face_x_m', 'face_x_s', 'face_y_m', 'face_y_s', 'eyebrow_l_x_m', 'eyebrow_l_x_s'
        , "eyebrow_l_y_m", "eyebrow_l_y_s", 'eyebrow_r_x_m', 'eyebrow_r_x_s', 'eyebrow_r_y_m', 'eyebrow_r_y_s',
                             'nose_x_m', 'nose_x_s', 'nose_y_m', 'nose_y_s', 'eye_l_x_m', 'eye_l_x_s',
                             'eye_l_y_m', 'eye_l_y_s', 'eye_r_x_m', 'eye_r_x_s', 'eye_r_y_m', 'eye_r_y_s',
                             'mouth_x_m', 'mouth_x_s', 'mouth_y_m', 'mouth_y_s']
    return face_features
def sequ_divid(pre):
    p1 = pre.iloc[:75,:]
    p2 = pre.iloc[75:150, :]
    p3 = pre.iloc[150:225, :]
    p4 = pre.iloc[225:300, :]

    p1 = p1.describe()
    p1 = p1.loc[['mean', 'std', 'min', 'max'], :]
    p1.index = ['mean_0.25', 'std_0.25', 'min_0.25', 'max_0.25']
    p2 = p2.describe()
    p2 = p2.loc[['mean', 'std', 'min', 'max'], :]
    p2.index = ['mean_0.50', 'std_0.50', 'min_0.50', 'max_0.50']
    p3 = p3.describe()
    p3 = p3.loc[['mean', 'std', 'min', 'max'], :]
    p3.index = ['mean_0.75', 'std_0.75', 'min_0.75', 'max_0.75']
    p4 = p4.describe()
    p4 = p4.loc[['mean', 'std', 'min', 'max'], :]
    p4.index = ['mean_1.00', 'std_1.00', 'min_1.00', 'max_1.00']

    return p1, p2, p3,p4

def use_feature(pre):
    face_feture = get_face_feature(pre)

    el1_x = pre.iloc[:300, 13:13 + 28]
    el1_x_o = el1_x.iloc[:,8:20].T
    el1_x_o = np.array(el1_x_o.mean())
    el1_x_i = el1_x.iloc[:,20:].T
    el1_x_i = np.array(el1_x_i.mean())
    el1_y = pre.iloc[:300, 69:69 + 28]
    el1_y_o = el1_y.iloc[:,8:20].T
    el1_y_o = np.array(el1_y_o.mean())
    el1_y_i = el1_y.iloc[:,20:].T
    el1_y_i = np.array(el1_y_i.mean())

    el2_x = pre.iloc[:300, 13 + 28:13 + 28 + 28]
    el2_x_o = el2_x.iloc[:,8:20].T
    el2_x_o = np.array(el2_x_o.mean())
    el2_x_i = el2_x.iloc[:,20:].T
    el2_x_i = np.array(el2_x_i.mean())
    el2_y = pre.iloc[:300, 69 + 28:69 + 28 + 28]
    el2_y_o = el2_y.iloc[:,8:20].T
    el2_y_o = np.array(el2_y_o.mean())
    el2_y_i = el2_y.iloc[:,20:].T
    el2_y_i = np.array(el2_y_i.mean())
    eye_landmark = pd.DataFrame([el1_x_i, el1_x_o, el1_y_i, el1_y_o, el2_x_i, el2_x_o, el2_y_i,el2_y_o]
                            , index=('el1_x_i', 'el1_x_o','el1_y_i', 'el1_y_o',
                                     'el2_x_i','el2_x_o','el2_y_i','el2_y_o'))
    eye_landmark = eye_landmark.T
    eyeGaze = pre.iloc[:300, 5:11]
    headPose = pre.iloc[:300, 293:299]
    AUs = pre.iloc[:300, 679:696]
    pose = pre.iloc[:300,716:]
    pose = pose.iloc[:,[0*3,0*3+1,1*3,1*3+1,2*3,2*3+1,3*3,3*3+1,5*3,5*3+1,6*3,6*3+1,
                        15*3,15*3+1,16*3,16*3+1,17*3,17*3+1,18*3,18*3+1]]
    pose.columns = ['0_x','0_y','1_x','1_y','2_x','2_y','3_x','3_y','5_x','5_y','6_x','6_y',
                    '15_x','15_y','16_x','16_y','17_x','17_y','18_x','18_y',]
    procData = pd.concat([eyeGaze, eye_landmark, headPose,face_feture, AUs, pose], axis=1)

    return procData

def pose_feature(pre):
    pose = pre.iloc[:300,:]
    return pose

def data_precess(pre):
    select_feature = use_feature(pre)
    # select_feature = pose_feature(pre)
    p1, p2, p3, p4 = sequ_divid(select_feature)
    data = pd.concat([p1,p2,p3,p4])
    return data



# %%
label = pd.read_csv('/home/ytikewk/python_project/daisee_detect/labelPath.csv')
label = label.iloc[:,2:]
label

df1 = label #8925*8
df1=df1[~df1['ClipID'].isin(['2100552061.avi'])] # no video
df1=df1[~df1['ClipID'].isin(['205601028.avi'])] # no pose
df1=df1[~df1['ClipID'].isin(['205601027.avi'])] # no pose
df1=df1[~df1['ClipID'].isin(['205601025.avi'])] # no pose
label = df1

print(label) #shoudld be 8921*8


# %%
l1 = label.iloc[:1000,:]
l2 = label.iloc[1000:2000,:]
l3 = label.iloc[2000:3000,:]
l4 = label.iloc[3000:4000,:]
l5 = label.iloc[4000:5000,:]
l6 = label.iloc[5000:6000,:]
l7 = label.iloc[6000:7000,:]
l8 = label.iloc[7000:8000,:]
l9 = label.iloc[8000:,:]

# %%
l81 = l8.iloc[:400,:]
l82 = l8.iloc[400:700,:]
l83 = l8.iloc[700:,:]
l91 = l9.iloc[:300,:]
l92 = l9.iloc[300:600,:]
l93 = l9.iloc[600:,:]
# %%
from multiprocessing import Process
pp1 = Process(target=process_data,args=[l81,'l81'])
pp1.start()
pp2 = Process(target=process_data,args=[l82,'l82'])
pp2.start()
pp3 = Process(target=process_data,args=[l83,'l83'])
pp3.start()
pp4 = Process(target=process_data,args=[l91,'l91'])
pp4.start()
pp5 = Process(target=process_data,args=[l92,'l92'])
pp5.start()
pp6 = Process(target=process_data,args=[l93,'l93'])
pp6.start()
# %%
from multiprocessing import Process
ll1 = label.iloc[:2,:]
ll2 = label.iloc[2:4,:]
# %%

p1 = Process(target=process_data,args=[l1,'l1'])
p1.start()
p2 = Process(target=process_data,args=[l2,'l2'])
p2.start()
p3 = Process(target=process_data,args=[l3,'l3'])
p3.start()
p4 = Process(target=process_data,args=[l4,'l4'])
p4.start()
p5 = Process(target=process_data,args=[l5,'l5'])
p5.start()
p6 = Process(target=process_data,args=[l6,'l6'])
p6.start()
p7 = Process(target=process_data,args=[l7,'l7'])
p7.start()
p8 = Process(target=process_data,args=[l8,'l8'])
p8.start()
p9 = Process(target=process_data,args=[l9,'l9'])
p9.start()
#%%
os.chdir('/home/ytikewk/python_project/daisee_detect/dataprocess')
l1 = pd.read_csv('l1.csv',index_col=0)
l2 = pd.read_csv('l2.csv',index_col=0)
l3 = pd.read_csv('l3.csv',index_col=0)
l4 = pd.read_csv('l4.csv',index_col=0)
l5 = pd.read_csv('l5.csv',index_col=0)
l6 = pd.read_csv('l6.csv',index_col=0)
l7 = pd.read_csv('l7.csv',index_col=0)
l81 = pd.read_csv('l81.csv',index_col=0)
l82 = pd.read_csv('l82.csv',index_col=0)
l83 = pd.read_csv('l83.csv',index_col=0)
l91 = pd.read_csv('l91.csv',index_col=0)
l92 = pd.read_csv('l92.csv',index_col=0)
l93 = pd.read_csv('l93.csv',index_col=0)
data = pd.concat([l1,l2,l3,l4,l5,l6,l7,l81,l82,l83,l91,l92,l93],axis=0)
data.to_csv('finall_process')
# %%
import time
s = time.time()
# aa = process_data(l1)
aa = data_precess(pre)
e = time.time()
e-s
# %%

    # # print procedure
    # l = l + 1
    # if l % 300 == 0:
    #     print(l)

mul_index = engg_prece.index
split_index = []
for i in range(len(mul_index)):
    split_index.append(str(mul_index[i][1] + '_' + mul_index[i][0]))
# data_frame_val = pd.DataFrame(np.array(process)[:, 1:], index=np.array(process)[:, 0], columns=split_index)
# data_frame_val.to_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/DataSet/'
#                       'test_data_process/test_data.csv')

file_name = np.array(label.iloc[:,0])

    # prt_Data.append(np.array(data).reshape(-1))

# %%
da = np.array(prt_Data)
da.shape
target = np.array(label.iloc[:,2])

# %%
from collections import Counter
Counter(target)
# %%
np.save('/home/ytikewk/python_project/daisee_detect/selected4pData_face.npy',da)
np.save('/home/ytikewk/python_project/daisee_detect/final_process.npy',process)
# %%
np.save('/home/ytikewk/python_project/daisee_detect/selected4pData.npy',da)
np.save('/home/ytikewk/python_project/daisee_detect/selected4pData_target.npy',target)
# %%
data = np.load('/home/ytikewk/python_project/daisee_detect/selected4pData.npy')
label = np.load('/home/ytikewk/python_project/daisee_detect/selected4pData_target.npy')

da =  pd.DataFrame(data,index=file_name,columns=split_index)
da.to_csv('/home/ytikewk/python_project/daisee_detect/selected4pData.csv')

# %%
# dd = np.array(process)
# da =  pd.DataFrame(dd[:,1:],index=file_name,columns=split_index)
da.to_csv('/home/ytikewk/python_project/daisee_detect/final_process.csv')

# %%
ddad = pd.read_csv('/home/ytikewk/python_project/daisee_detect/final_process.csv',index_col=0)
# %%
aaa = pd.read_csv('/home/ytikewk/python_project/daisee_detect/selected4pData.csv',index_col=0)
# %% import test data
os.chdir('/home/ytikewk/Documents/data/face/openFace/test')
l = 0
process = []
test_label_full = pd.read_csv('/home/ytikewk/Documents/data/label/Labels-20200809T023223Z-001/'
                              'Labels/TestLabels.csv')
for f in tqdm(np.array(test_label_full.loc[:, 'ClipID'])):
    pre = pd.read_csv(f[:-3] + 'csv')
    ########################
    # pre = pre[~ pre[' success'].isin([0])]
    ########################
    # choose features which are not landmark
    part1 = pre.iloc[:, 11:13]
    part2 = pre.iloc[:, 293:299]
    part3 = pre.iloc[:, -35:]
    engg_pro = pd.concat([part1, part2, part3], axis=1).describe().iloc[1:, :]
    engg_prece = engg_pro.T.stack() # flatten

    engg_np = np.array(engg_prece)
    engg_np = np.append(f,engg_np) # add file name
    process.append(engg_np)

    # # print procedure
    # l = l + 1
    # if l % 300 == 0:
    #     print(l)
    break
mul_index = engg_prece.index
split_index = []
for i in range(len(mul_index)):
    split_index.append(str(mul_index[i][0] + '_' + mul_index[i][1]))
data_frame_val = pd.DataFrame(np.array(process)[:,1:],index=np.array(process)[:,0],columns=split_index)
data_frame_val.to_csv('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/DataSet/'
                      'test_data_process/test_data.csv')


# %%
#full_data, full_label = zip_data(train_data,train_label,test_data,test_label,val_data,val_label)

# assert test_data.columns != train_data.columns
# %%
# select 2d gaze (del the 3d gaze)
# # %% choose gaze,pose,aus
# gp = data_frame.iloc[:,:98] # gaze + pose
# gp_val = data_frame_val.iloc[:,:98]
# au = data_frame.iloc[:,378:]
# au_val = data_frame_val.iloc[:,378:]
# data_frame = pd.concat([gp,au],axis=1)
# data_frame_val = pd.concat([gp_val,au_val],axis=1)
# #data_frame.to_csv('../../train_data_process/data_frame_gp+au.csv',index=True)
# #data_frame_val.to_csv('../../val_data_process/data_frame_val_gp+au.csv')






#
#
# # %% import val X data
# os.chdir('/home/ytikewk/python_project/daisee_detect/validation_process/')
# l = 0
# process = []
# for f in tqdm(label.loc[:, 'ClipID']):
#     pre = pd.read_csv(f[:-3] + 'csv')
#     ########################
#     # pre = pre[~ pre[' success'].isin([0])]
#     ########################
#     # choose features which are not landmark
#     part1 = pre.iloc[:, 5:13]
#     part2 = pre.iloc[:, 293:299]
#     part3 = pre.iloc[:, 639:]
#     engg_pro = pd.concat([part1, part2, part3], axis=1).describe().iloc[1:, :]
#     engg_prece = engg_pro.T.stack() # flatten
#
#     engg_np = np.array(engg_prece)
#     engg_np = np.append(f,engg_np) # add file name
#     process.append(engg_np)
#
#     # print procedure
#     l = l + 1
#     if l % 300 == 0:
#         print(l)
#
# mul_index = engg_prece.index
# split_index = []
# for i in range(len(mul_index)):
#     split_index.append(str(mul_index[i][0] + '_' + mul_index[i][1]))
# data_frame_val = pd.DataFrame(np.array(process)[:,1:],index=np.array(process)[:,0],columns=split_index)
# data_frame_val.to_csv('../val_data_process/data_frame_val_all.csv')
#
#
#
# # %% import  X
# os.chdir('/media/ytikewk/76627FE2627FA591/dataset/DAiSEE/DAiSEE/DataSet/Train/train_process')
#
# train_process = []
# l = 0
# for f in train_label.loc[:, 'ClipID']:
#     pre = pd.read_csv(f[:-3] + 'csv')
#     # if np.array(pre).shape != (300, 714):
#     #     pre = pre.iloc[:300, :]
#     # assert np.array(pre).shape == (300,714)
#     #######################################
#     #pre = pre[~ pre[' success'].isin([0])]
#     #######################################
#
#     part1 = pre.iloc[:, 5:13]
#     part2 = pre.iloc[:, 293:299]
#     part3 = pre.iloc[:, 639:]
#     engg_pro = pd.concat([part1, part2, part3], axis=1).describe().iloc[1:, :]
#     engg_prece = engg_pro.T.stack()
#     engg_np = np.array(engg_prece)
#     engg_np = np.append(f,engg_np)
#     train_process.append(engg_np)
#     l = l + 1
#     if l == 1:
#         mul_index = engg_prece.index
#         split_index = []
#         for i in range(len(mul_index)):
#             split_index.append(str(mul_index[i][0] + '_' + mul_index[i][1]))
#
#     if l % 300 == 0:
#         print(l)
#
# # save the data
# data_frame = pd.DataFrame(np.array(train_process)[:,1:],index=np.array(train_process)[:,0],columns=split_index)
# data_frame.to_csv('../../train_data_process/data_frame_all.csv',index=True)
#
#
#
#
#
#
#
# # %%select featureimport pandas as pd
# import numpy as np
# from tqdm import tqdm
# import glob, os, re
# import xarray as xr
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.metrics import confusion_matrix
#
# from collections import Counter
# def select_feature(data_frame):
#     # 　gaze _0/_1 _x/_y/_z _std/_mean
#     '''
#     low_data = data_frame.iloc[:,:2]
#     low_data = pd.concat([low_data,data_frame.iloc[:,7:9]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,14:16]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,21:23]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,28:30]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,35:37]],axis=1)
#     '''
#     low_data = data_frame.iloc[:,42:45] # x_angel_mean/std/min
#     low_data = pd.concat([low_data,data_frame.iloc[:,48]],axis=1) # x_angel_max
#     low_data = pd.concat([low_data,data_frame.iloc[:,49:52]],axis=1) #y_angel_mean/std/min
#     low_data = pd.concat([low_data,data_frame.iloc[:,55]],axis=1) # y_angel_max
#
#     # head pose & rotation mean/std/min/max
#     low_data = pd.concat([low_data,data_frame.iloc[:,56:59]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,62:66]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,69:73]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,76:80]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,83:87]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,90:94]],axis=1)
#     low_data = pd.concat([low_data,data_frame.iloc[:,98]],axis=1)
#
#     # action_unit mean/std
#     # low_data = pd.concat([low_data,data_frame.iloc[:,55*7:55*7+2]],axis=1) # AU02_r outer brow raiser
#     # low_data = pd.concat([low_data,data_frame.iloc[:,64*7:64*7+2]],axis=1) # AU15_r lip corner depressor
#     # low_data = pd.concat([low_data,data_frame.iloc[:,69*7:69*7+2]],axis=1) # AU26_r jaw drop
#     # low_data = pd.concat([low_data,data_frame.iloc[:,70*7:70*7+2]],axis=1) # AU45_r blink
#     # low_data = pd.concat([low_data,data_frame.iloc[:,72*7:72*7+2]],axis=1) # AU02_c outer brow raiser
#     # low_data = pd.concat([low_data,data_frame.iloc[:,81*7:81*7+2]],axis=1) # AU15_c lip corner depressor
#     # low_data = pd.concat([low_data,data_frame.iloc[:,86*7:86*7+2]],axis=1) # AU26_c jaw drop
#     # low_data = pd.concat([low_data,data_frame.iloc[:,88*7:88*7+2]],axis=1) # AU45_c blink
#
#
#     low_data = pd.concat([low_data,data_frame.iloc[:,54*7:54*7+2]],axis=1) # AU01_r outer brow raiser
#     low_data = pd.concat([low_data,data_frame.iloc[:,55*7:55*7+2]],axis=1) # AU02_r outer brow raiser
#     low_data = pd.concat([low_data, data_frame.iloc[:, 56 * 7:56 * 7 + 2]], axis=1)  # AU04_r outer brow raiser
#     low_data = pd.concat([low_data, data_frame.iloc[:, 59 * 7:59 * 7 + 2]], axis=1)  # AU07_r outer brow raiser
#     low_data = pd.concat([low_data, data_frame.iloc[:, 61 * 7:61 * 7 + 2]], axis=1)  # AU10_r outer brow raiser
#     low_data = pd.concat([low_data,data_frame.iloc[:,63*7:63*7+2]],axis=1) # AU14_r lip corner depressor
#     low_data = pd.concat([low_data,data_frame.iloc[:,70*7:70*7+2]],axis=1) # AU45_r blink
#     low_data = pd.concat([low_data,data_frame.iloc[:,(53+18)*7:(53+18)*7+2]],axis=1) # AU01_r outer brow raiser
#     low_data = pd.concat([low_data,data_frame.iloc[:,(54+18)*7:(54+18)*7+2]],axis=1) # AU02_r outer brow raiser
#     low_data = pd.concat([low_data, data_frame.iloc[:, (55+18) * 7:(55+18) * 7 + 2]], axis=1)  # AU04_r outer brow raiser
#     low_data = pd.concat([low_data, data_frame.iloc[:, (58+18) * 7:(58+18) * 7 + 2]], axis=1)  # AU07_r outer brow raiser
#     low_data = pd.concat([low_data, data_frame.iloc[:, (60+18) * 7:(60+18) * 7 + 2]], axis=1)  # AU10_r outer brow raiser
#     low_data = pd.concat([low_data,data_frame.iloc[:,(62+18)*7:(62+18)*7+2]],axis=1) # AU14_r lip corner depressor
#     low_data = pd.concat([low_data,data_frame.iloc[:,(70+18)*7:(70+18)*7+2]],axis=1) # AU45_r blink
#     return low_data
#
# low_train = select_feature(data_frame)
# low_val = select_feature(data_frame_val)