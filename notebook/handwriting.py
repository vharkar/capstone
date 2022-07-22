#!/usr/bin/env python
# coding: utf-8

# In[3]:


# conda install seaborn


# In[2]:


# conda install -c conda-forge scikit-image


# In[3]:


# conda install -c conda-forge opencv


# In[4]:


#conda install -c conda-forge tensorflow


# In[5]:


# conda install -c conda-forge keras


# In[63]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Read the Data

# In[64]:


train_data = pd.read_csv('../images/written_name_train_v2.csv')
valid_data = pd.read_csv('../images/written_name_validation_v2.csv')


# ## Analyze the Data

# In[65]:


train_data.head()


# In[66]:


valid_data.head()


# In[69]:


train_data.shape


# In[70]:


valid_data.shape


# In[72]:


train_data.value_counts()


# In[73]:


valid_data.value_counts()


# In[74]:


valid_data.columns


# In[75]:


train_data.columns


# ## Check and remove nulls

# In[76]:


train_data.isnull().sum()


# In[77]:


valid_data.isnull().sum()


# In[78]:


train_data=train_data.dropna()


# In[79]:


valid_data=valid_data.dropna()


# In[80]:


train_data.isnull().sum()


# In[81]:


valid_data.isnull().sum()


# ## What do the files look like?

# In[8]:


train=train_data
valid=valid_data


# In[9]:


def drawTheImg(plt, img_dir):
 image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
 plt.imshow(image, cmap = 'gray')
 plt.title(train.loc[i, 'IDENTITY'], fontsize=12)
 plt.axis('off')


# In[10]:


plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '../images/train/' + train.loc[i,'FILENAME']
    drawTheImg(plt, img_dir)
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)


# ## Some names are unreadable - remove them

# In[11]:


unreadable = train[train['IDENTITY'] == 'UNREADABLE']
unreadable.reset_index(inplace = True, drop=True)

plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '../images/train/'+ unreadable.loc[i, 'FILENAME']
    drawTheImg(plt, img_dir)

plt.subplots_adjust(wspace=0.2, hspace=-0.8)


# In[12]:


train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']


# ## Convert all labels to uppercase

# In[13]:


train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()


# ## Reset Indexes, after cleaning

# In[14]:


train.reset_index(inplace = True, drop=True) 
valid.reset_index(inplace = True, drop=True)


# In[82]:


train.head()


# ## Preprocessing and preparing images for training

# In[15]:


def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


# In[16]:


train_size = 30000
valid_size= 3000


# In[17]:


train_x = []

for i in range(train_size):
    img_dir = '../images/train/' + train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    train_x.append(image)


# In[18]:


valid_x = []

for i in range(valid_size):
    img_dir = '../images/validation/' + valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    valid_x.append(image)


# ## CTC Prep

# In[19]:


train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)


# In[20]:


alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret


# In[21]:


name = 'MARYANNE'
print(name, '\n',label_to_num(name))


# * train_y contains the true labels converted to numbers and padded with -1. The length of each label is equal to max_str_len.
# * train_label_len contains the length of each true label (without padding)
# * train_input_len contains the length of each predicted label. The length of all the predicted labels is constant i.e number of timestamps - 2.
# * train_output is a dummy output for ctc loss.

# In[22]:


train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    train_label_len[i] = len(train.loc[i, 'IDENTITY'])
    train_y[i, 0:len(train.loc[i, 'IDENTITY'])]= label_to_num(train.loc[i, 'IDENTITY'])  


# In[23]:


valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])    


# In[24]:


print('True label : ',train.loc[100, 'IDENTITY'] , '\ntrain_y : ',train_y[100],'\ntrain_label_len : ',train_label_len[100], 
      '\ntrain_input_len : ', train_input_len[100])


# In[25]:


import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import adam_v2


# In[26]:


input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()


# In[58]:


model.layers


# In[27]:


# the ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# In[28]:


labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


# In[59]:


model_final.summary()


# ## Train the model

# In[30]:


# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam_v2.Adam(lr = 0.0001))

model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs=60, batch_size=128)


# In[60]:


model.summary()
model_final.summary()


# In[61]:


model_final.layers


# ## Check Model performance on validation set

# In[57]:


preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))


# In[32]:


y_true = valid.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

for i in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))


# ## Predict

# In[55]:


test = pd.read_csv('../images/written_name_test_v2.csv')

plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '../images/test/' + test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = preprocess(image)
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)


# ## Save Model

# In[37]:


# as a keras file
#model.save("../assets/handwriting-model-1.h5")


# In[89]:


# load model
from keras.models import load_model
model2 = load_model('../assets/handwriting-model-1.h5', compile=False)


# ## Read saved model and predict with it.

# In[90]:


test = pd.read_csv('../images/written_name_test_v2.csv')

plt.figure(figsize=(15, 10))
for i in range(20):
    ax = plt.subplot(5, 4, i+1)
    img_dir = '../images/test/' + test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = preprocess(image)
    image = image/255.
    pred = model2.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)


# In[ ]:




