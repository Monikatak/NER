import re
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm 

def readconll(file):
    lines = [line.strip() for line in open(file)]
    while lines[-1] == '':  # Remove trailing empty lines
        lines.pop()
    s = [x.split('_') for x in '_'.join(lines).split('__')]  # Quick split corpus into sentences
    return [[y.split() for y in x] for x in s]


E = readconll('eng.train')
Y =[]
             
def wordshape(string):      #Extract the wordshape of the words in the sentences
    s = str()
    for i in range(0,len(string)):
        if(string[i].isupper() and string[i].isalpha()):
            s = s + 'X'
        elif(string[i].islower() and string[i].isalpha()):
            s = s + 'x'
        elif(string[i].isdigit()):
            s = s + 'd'
        else:
            s = s + string[i]
    return s
 
def extractFeatures(dat):      # Extract features of the train/test data
    f = []
    for x in readconll(dat):
        for j in range(0,len(x)):
            info = dict()
            info['currentword,'+ x[j][0]] = 1
            info['wordshape,' + wordshape(x[j][0])] = 1
            info['prefix,' + x[j][0][0:4]] =1
            info['suffix,' + x[j][0][-4:]] =1
            info['currentPOS,' + x[j][1]] = 1
            if(j>0):
                info['previousWord,'+ x[j-1][0]] =1
                info['previousPOS,' + x[j-1][1]] =1
                info['Prevwordshape,' + wordshape(x[j-1][0])] = 1
                if(x[j-1][0][0].isupper()):
                    info['Previousinitcaps'] = 1   # feature if the previous word in the sentence has intial letter as capital
            if(j<len(x)-1):
                info['nextWord,'+x[j+1][0]] =1
                info['nextPOS,'+ x[j+1][1]] =1
                if(x[j+1][0][0].isupper()):
                    info['Nextinitcaps'] = 1      # feature if the next word in the sentence has intial letter as capital
                info['Nextwordshape,' + wordshape(x[j+1][0])] = 1
            if(x[j][0][0].isupper()):
                info['initcaps'] = 1             # feature if the current word has first letter as capital
            if(x[j][0].isupper()):
                info['ALLUPPER'] = 1             # feature if all letters are in uppercase
            if(j == 0 and x[j][0][0].isupper() == True):
                info['FirstwordCaps'] =1           # feature if the first word of the sentence has first leller as capital
            if(re.match('[a-zA-z]+\-[a-zA-z]+',x[j][0])):
                info['ContainsHyphen'] =1        # fearure for the presence of hyphen
            if(re.match('[a-zA-z]*[0-9]+[a-zA-z]*',x[j][0])):
                info['ContainsDigit'] =1         # fearure for the presence of digits.
            f.append(info)
            if(dat == 'eng.train'):
                Y.append(x[j][3])            # Extract label classes from the training corpus
    return f
 
F_train = extractFeatures('eng.train')    # Extract features of Training data
F_test = extractFeatures('eng.testa')     # Extract feautures of Test data

vectorizer = DictVectorizer(sparse = True)
X = vectorizer.fit_transform(F_train)
clf = svm.LinearSVC()                             # use Linear SVM to fit the data
clf.fit(X, Y)  

Res = clf.predict(vectorizer.transform(F_test))  # predict the labels of the Test data features
   
fo1 =open("eng.guessa",'w+')
fo = open("eng.testa",'r')
lines = [line.strip() for line in fo]
while lines[-1] == '':  # Remove trailing empty lines
        lines.pop()
j = 0
for line in lines:
    if(line == ''):
        fo1.write(line+"\n")      # Append empty lines in eng.guessa as it is
        continue
    else:
        line = line + " " + Res[j]       # Append the predicted label to the last column in test data
        fo1.write(line + "\n")           # Write the output to the eng.guessa file
        j = j+1

fo.close()
fo1.close()



