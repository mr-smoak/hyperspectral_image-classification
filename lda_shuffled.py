import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import array
import csv
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import safe_indexing, indexable
from itertools import chain

dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\x_ip.csv",header=None)
dataset = dataframe.values
X=array(dataset)

dataframe2 = pd.read_csv("C:\\Users\\User\\Desktop\\paperwork\\y_ip.csv",header=None)
dataset2 = dataframe2.values
y=array(dataset2)

lda = LinearDiscriminantAnalysis(n_components=16)
X_r2 = lda.fit(X, y).transform(X)

seed = 1

cv = ShuffleSplit(random_state=seed, test_size=0.33)
arrays = indexable(X_r2, y)
train, test = next(cv.split(X=X_r2))
iterator = list(chain.from_iterable((
    safe_indexing(a, train),
    safe_indexing(a, test),
    train,
    test
    ) for a in arrays)
)
xtrain, xtest, train_is, test_is, ytrain, ytest, _, _  = iterator

with open("C:/Users/User/Desktop/paperwork/dataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(X_r2)

with open("C:/Users/User/Desktop/paperwork/xtraindataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(xtrain)
    
with open("C:/Users/User/Desktop/paperwork/xtestdataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(xtest)    

with open("C:/Users/User/Desktop/paperwork/ytraindataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(ytrain)

with open("C:/Users/User/Desktop/paperwork/ytestdataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(ytest) 
    

with open("C:/Users/User/Desktop/paperwork/ytestdataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(ytest) 

with open("C:/Users/User/Desktop/paperwork/indices_train_dataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(map(lambda x: [x], train_is))
    output.close()
    
with open("C:/Users/User/Desktop/paperwork/indices_test_dataset_ip_featuresextracted_lda_shuffled.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(map(lambda x: [x], test_is))
    output.close()    