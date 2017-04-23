import csv
import random
import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt 

# Making randomization deterministic
random.seed(1)

# Constants
NUM_ENTRIES = [0,1,4,5,6,7,8,9,10,11]
MONTH = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
DAY = ['mon','tue','wed','thu','fri','sat','sun']
TRAIN_SIZE = 0.95

# Getting input from csv file
with open('forestfires.csv','rb') as f:
    reader = csv.reader(f)
    raw_data = list(reader)
raw_data = raw_data[1:]
area = np.array(raw_data)[:,12]

# Preprocessing
data = list()
for entry in raw_data:
    mod_entry = []
    for i in range(len(entry)):
        if i == 12:
             mod_entry.append(np.log(float(entry[i])+1))
        elif i in NUM_ENTRIES:
            mod_entry.append(float(entry[i]))
        elif i == 2:
            zero = np.zeros(12)
            zero[MONTH.index(entry[i])] = 1
            for j in zero:
                mod_entry.append(j)
        else:
            zero = np.zeros(7)
            zero[DAY.index(entry[i])] = 1
            for j in zero:
                mod_entry.append(j)
    data.append(mod_entry)
# Shuffling data
random.shuffle(data)
data = np.array(data)

# Standardization
# Computing the mean and variance along the columns  
mean = np.mean(data,axis = 0,dtype = np.float32)
var = np.var(data,axis = 0,dtype = np.float32)
out_mean = mean[29]
out_var = var[29]
for i in range(len(data)):
    data[i] = np.divide((data[i]-mean),np.sqrt(var))

X_train = data[0:int(TRAIN_SIZE*len(data)),:29]
Y_train = data[0:int(TRAIN_SIZE*len(data)),29]
X_test = data[int(TRAIN_SIZE*len(data)):,0:29]
Y_test = data[int(TRAIN_SIZE*len(data)):,29]

Y_orig = np.exp(Y_test*np.power(out_var,0.5) + out_mean) - 1
Yt_orig = np.exp(Y_train*np.power(out_var,0.5) + out_mean) - 1

print "No. of Testing samples: ", len(data[:,29])
#C = 3
#gamma = 

svr_rbf = SVR(kernel = 'rbf', C = 3, gamma = 1, epsilon = 0.01)
svm_model = svr_rbf.fit(X_train, Y_train)
param = svm_model.get_params()
h = svm_model.predict(X_test)
h_t = svm_model.predict(X_train)
h_orig = np.exp(h*np.power(out_var,0.5) + out_mean) - 1
ht_orig = np.exp(h_t*np.power(out_var,0.5) + out_mean) - 1

N = np.float32(len(Y_test))
MAD = (1/N)*np.sum(np.abs(Y_orig - h_orig))
RMSE = np.sqrt(np.sum((Y_orig - h_orig)**2)/N)

Nt = np.float32(len(Y_train))
MADt = (1/Nt)*np.sum(np.abs(Yt_orig - ht_orig))
RMSEt = np.sqrt(np.sum((Yt_orig - ht_orig)**2)/Nt)

print "MAD - testing data: ", MAD, " RMSE - testing data: ", RMSE
print "MAD - training data: ", MADt, " RMSE - training data: ", RMSEt

for key in param:
    print key

'''
lw = 2
plt.plot(Yt_orig, color = 'darkorange', label = 'Testing target')
plt.hold('on')
plt.plot(ht_orig, color = 'navy', lw = lw, label = 'Prediction')
plt.xlabel('Samples')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
'''