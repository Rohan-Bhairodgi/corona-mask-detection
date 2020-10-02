#!/usr/bin/env python
# coding: utf-8

# In[ ]:


label_dict={"with_mask":0,"without_mask":1}
categories=["with_mask","without_mask"]
label=[0,1]
import os,cv2
data_path= ('C:\\Users\\Balaji\\Documents\\dataset')
features=[]
target=[]
for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resize=cv2.resize(gray,(100,100))
            features.append(resize)
            target.append(label_dict[category])
        except Exception as e:
            pass


# In[2]:


import numpy as np


# In[3]:


features=np.array(features)
features=features/255


# In[4]:


features.shape


# In[5]:


features=np.reshape(features,(features.shape[0],100,100,1))
features.shape


# In[6]:


target=np.array(target)
target.shape


# In[9]:


import tensorflow
from keras.utils import np_utils


# In[10]:


new_target= np_utils.to_categorical(target)


# In[ ]:





# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout 
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint 
model=Sequential() 
model.add(Conv2D(200,(3,3),input_shape=(100,100,1),activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(100,(3,3),activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten())
model.add(Dropout(0.5)) 
model.add(Dense(50,activation="relu")) 
model.add(Dense(2,activation="softmax")) 
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 
train_features,test_features,train_target,test_target=train_test_split(features,new_target,test_size=0.1) 
checkpoint=ModelCheckpoint("model-{epoch:03d}.model",save_best_only=True,mode="auto") 
history=model.fit(train_features,train_target,epochs=20,validation_split=0.2)


# In[25]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 
train_features,test_features,train_target,test_target=train_test_split(features,new_target,test_size=0.2) 
saveweights=ModelCheckpoint("newmaskdata.h5",monitor='val_loss',verbose=0,save_best_only=True,mode="min")
train=model.fit(train_features,train_target,callbacks=[saveweights],epochs=5,validation_split=0.2)


# In[ ]:





# In[26]:


import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model


# In[27]:


model = load_model('newmaskdata.h5')

labels_dict={0:'mask',1:'no mask'}
color_dict={0:(0,255,255),1:(0,0,255)}

source = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('C:/Users/Balaji/AppData/Local/programs/Python/Python38/Lib/site-packages/cv2/haar-cascade-files-master/haarcascade_frontalface_default.xml')


# In[ ]:


while (True):
    ret,img = source.read()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(gray,1.3,5)

    # Draw rectangles around each face
    for x,y,w,h in faces:
        
        face_img = gray[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result = model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)
   
    if (key == 27): 
        break

cv2.destroyAllWindows()
source.release()



# In[ ]:





# In[ ]:





# In[ ]:




