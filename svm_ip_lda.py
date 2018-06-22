import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn.svm import SVC
import pandas as pd
import cv2

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\xtraindataset_ip_featuresextracted_lda_shuffled.csv",header=None)
dataset = dataframe.values
in1=array(dataset)

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\ytraindataset_ip_featuresextracted_lda_shuffled.csv",header=None)
dataset = dataframe.values
in2=array(dataset)

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\xtestdataset_ip_featuresextracted_lda_shuffled.csv",header=None)
dataset = dataframe.values
in3=array(dataset)

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\ytestdataset_ip_featuresextracted_lda_shuffled.csv",header=None)
dataset = dataframe.values
in4=array(dataset)

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\indices_train_dataset_ip_featuresextracted_lda_shuffled.csv",header=None)
dataset = dataframe.values
in5=array(dataset)

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\indices_test_dataset_ip_featuresextracted_lda_shuffled.csv",header=None)
dataset = dataframe.values
in6=array(dataset)

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(in1, in2)
predictions = svm_model_linear.predict(in3)
 
cm = confusion_matrix(in4, predictions)
print(cm)
 
for i in range(len(predictions)):
 print(str(predictions[i])+"   "+str(in4[i]))
  
a=accuracy_score(in4,predictions)
print(a)

blank_image = np.zeros((145,145,3), np.uint8)

for x in range(len(in5)):
        ind=in5[x]
        i=ind/145
        j=ind%145
        blank_image[int(j)][int(i)]=(in2[int(x)]*17,in2[int(x)]*17,in2[int(x)]*17)
        
for x in range(len(in6)):
        ind=in6[x]
        i=ind/145
        j=ind%145
        blank_image[int(j)][int(i)]=(predictions[int(x)]*17,predictions[int(x)]*17,predictions[int(x)]*17)        

       
cv2.imshow('image',blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()  


