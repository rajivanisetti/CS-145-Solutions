import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import seaborn as sns; sns.set()
import collections
import math
import sys
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
from sklearn.metrics import confusion_matrix

class NB_model():
    def __init__(self): 
        self.pi = {} # to store prior probability of each class 
        self.Pr_dict = None
        self.num_vocab = None
        self.num_classes = None
        self.total_class = None
    
    def fit(self, train_data, train_label, vocab, if_use_smooth=True):
        # get prior probabilities
        self.num_vocab = len(vocab['index'].tolist())
        self.get_prior_prob(train_label)
        self.Pr_dict = collections.defaultdict(lambda: collections.defaultdict(float))
        self.total_class = collections.defaultdict(int)                                      # Number of words within class
        word_count = collections.defaultdict(lambda: collections.defaultdict(int))      # Count of word within specific class 


        train_dict = train_data.to_dict()
        for idx in range(len(train_dict['classIdx'])):
            self.total_class[train_dict['classIdx'][idx]] += train_dict['count'][idx]
            word_count[train_dict['classIdx'][idx]][train_dict['wordIdx'][idx]] += train_dict['count'][idx]

        """      
        for word in vocab:                                          # Smoothing, if word not in word_count, make 1
            for classID in word_count:
                if word not in word_count[classID]:
                    word_count[classID][word] += 1
                    self.total_class[classID] += 1
        """

        for classID in word_count:
            for word in word_count[classID]:
                self.Pr_dict[classID][word] = float(word_count[classID][word] + 1) / float(self.total_class[classID] + self.num_vocab)
        
        # ================== YOUR CODE HERE ==========================
        # Calculate probability of each word based on class 
        # Hint: Store each probability value in matrix or dict: self.Pr_dict[classID][wordID] or Pr_dict[wordID][classID])
        # Remember that there are possible NaN or 0 in Pr_dict matrix/dict. Use smooth method
        
        # ============================================================

        print("Training completed!")
    
    def predict(self, test_data):
        test_dict = test_data.to_dict() # change dataframe to dict
        new_dict = {}
        prediction = []
        
        for idx in range(len(test_dict['docIdx'])):
            docIdx = test_dict['docIdx'][idx]
            wordIdx = test_dict['wordIdx'][idx]
            count = test_dict['count'][idx]
            try: 
                new_dict[docIdx][wordIdx] = count 
            except:
                new_dict[test_dict['docIdx'][idx]] = {}
                new_dict[docIdx][wordIdx] = count
                ''
        for docIdx in range(1, len(new_dict)+1):
            score_dict = {}
            #Creating a probability row for each class
            for classIdx in range(1,self.num_classes+1):
                score_dict[classIdx] = 0
                score_dict[classIdx] += math.log(self.pi[classIdx])
                for wordIdx in new_dict[docIdx]:
                    if self.Pr_dict[classIdx][wordIdx] != 0:
                        score_dict[classIdx] += new_dict[docIdx][wordIdx] * math.log(self.Pr_dict[classIdx][wordIdx])
                    else:
                        score_dict[classIdx] += new_dict[docIdx][wordIdx] * math.log(1/(self.total_class[classIdx] + self.num_vocab))   # If vocab word didn't come up during training,
                                                                                                                                        # smoothing means numerator ofdict value would be 1
                # ================== YOUR CODE HERE ==========================
                ### Implement the score_dict for all classes for each document
                ### Remember to use log addtion rather than probability multiplication
                ### Remember to add prior probability, i.e. self.pi

                # ============================================================
            max_score = max(score_dict, key=score_dict.get)
            prediction.append(max_score)
        return prediction
                    
    
    def get_prior_prob(self,train_label, verbose=True):
        unique_class = list(set(train_label))
        self.num_classes = len(unique_class)
        total = len(train_label)
        for c in unique_class:
            # ================== YOUR CODE HERE ==========================
            ### calculate prior probability of each class ####
            ### Hint: store prior probability of each class in self.pi

            # ============================================================
            count = 0
            for label in train_label:
                if c is label:
                    count += 1
            self.pi[c] = count / total
        if verbose:
            print("Prior Probability of each class:")
            print("\n".join("{}: {}".format(k, v) for k, v in self.pi.items()))

########### Data processing ##########
# read train/test labels from files
train_label = pd.read_csv('./dataset/train.label',names=['t'])
train_label = train_label['t'].tolist()
test_label = pd.read_csv('./dataset/test.label', names=['t'])
test_label= test_label['t'].tolist()
# read train/test documents from files
train_data = open('./dataset/train.data')
df_train = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
test_data = open('./dataset/test.data')
df_test = pd.read_csv(test_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
# read vocab
vocab = open('./dataset/vocabulary.txt') 
vocab_df = pd.read_csv(vocab, names = ['word']) 
vocab_df = vocab_df.reset_index() 
vocab_df['index'] = vocab_df['index'].apply(lambda x: x+1) 

#Add label column to original df_train (for better implementation)
docIdx = df_train['docIdx'].values
i = 0
new_label = []
for index in range(len(docIdx)-1):
    new_label.append(train_label[i])
    if docIdx[index] != docIdx[index+1]:
        i += 1
new_label.append(train_label[i])
df_train['classIdx'] = new_label

# output the following head of df_train dataframe if program correctly runned
#		docIdx	wordIdx	count	classIdx
#	0	1		1		4		1
#	1	1		2		2		1
#	2	1		3		10		1
#	3	1		4		4		1
#	4	1		5		2		1
df_train['classIdx'] = new_label
print(df_train.head()) # you may comment this line if you get correct results


########### Start to Train your model ##########
nbm = NB_model()
nbm.fit(df_train, train_label, vocab_df)

# make predictions on train set to validate the model
predict_train_labels = nbm.predict(df_train)
train_acc = (np.array(train_label) == np.array(predict_train_labels)).mean()
print("Accuracy on training data by my implementation: {}".format(train_acc))

# make predictions on test data
predict_test_labels = nbm.predict(df_test)
test_acc = (np.array(test_label) == np.array(predict_test_labels)).mean()
print("Accuracy on test data by my implementation: {}".format(test_acc))

# plot classification matrix
mat = confusion_matrix(test_label, predict_test_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig('./output/nbm_mine.png')


docIdx = df_test['docIdx'].values
wordIdx = df_test['wordIdx'].values
i = 0
new_label = []
p_labels = []
words = []
for index in range(len(docIdx)-1):
    new_label.append(test_label[i])
    p_labels.append(predict_test_labels[i])
    words.append(vocab_df['word'][wordIdx[index]])
    if docIdx[index] != docIdx[index+1]:
        i += 1
new_label.append(test_label[i])
p_labels.append(predict_test_labels[i])
words.append("final")
df_test['classIdx'] = new_label
df_test['predictIdx'] = p_labels
df_test['word'] = words

with open('misclassified.txt', 'w') as f:
    sys.stdout = f
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_test.query('docIdx == 4'))
