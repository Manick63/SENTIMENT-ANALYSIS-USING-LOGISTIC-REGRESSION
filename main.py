import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")
with open("full_set.txt") as f:
 content = f.readlines()
 content = [x.strip() for x in content]
 sentences = [x.split("\t")[0] for x in content]
 labels = [x.split("\t")[1] for x in content]
 y = np.array(labels, dtype='int8')
 y = 2 * y - 1
def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, '')
    return x
digits = [str(x) for x in range(10)] # Remove digits
digit_less = [full_remove(x, digits) for x in sentences]
punc_less = [full_remove(x, list(string.punctuation)) for x in
digit_less] # Remove punctuation
sents_lower = [x.lower() for x in punc_less] # Make everything lowercase
stop_set = set(['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to',
'of', 'it', 'from'])
sents_split = [x.split() for x in sents_lower]
sents_processed = [" ".join(list(filter(lambda a: a not in stop_set,
x))) for x in sents_split]
sents_processed[0:10]
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
preprocessor = None, stop_words = None, max_features = 4500)
data_features = vectorizer.fit_transform(sents_processed)
data_mat = data_features.toarray()
np.random.seed(0)
test_inds = np.append(np.random.choice((np.where(y==-1))[0], 250,
replace=False), np.random.choice((np.where(y==1))[0], 250,
replace=False))
train_inds = list(set(range(len(labels))) - set(test_inds))
train_data = data_mat[train_inds,]
train_labels = y[train_inds]
test_data = data_mat[test_inds,]
test_labels = y[test_inds]
print("train data: ", train_data.shape)
print("test data: ", test_data.shape)
clf = SGDClassifier(loss="log", penalty="none")
clf.fit(train_data, train_labels)
w = clf.coef_[0,:]
b = clf.intercept_
preds_train = clf.predict(train_data)
preds_test = clf.predict(test_data)
errs_train = np.sum((preds_train > 0.0) != (train_labels > 0.0))
errs_test = np.sum((preds_test > 0.0) != (test_labels > 0.0))
print ("Training error: ", float(errs_train)/len(train_labels))
print ("Test error: ", float(errs_test)/len(test_labels))
def margin_counts(clf, test_data, gamma):
    preds = clf.predict_proba(test_data)[:,1]
    margin_inds = np.where((preds > (0.5+gamma)) | (preds < (0.5-gamma)))[0]
    return float(len(margin_inds))
gammas = np.arange(0, 0.5, 0.01)
f = np.vectorize(lambda g: margin_counts(clf, test_data, g))
plt.plot(gammas, f(gammas) / 500.0, linewidth=2, color='green')
plt.xlabel('Margin', fontsize=14)
plt.ylabel('Fraction of points above margin', fontsize=14)
plt.show()
def margin_errors(clf, test_data, test_labels, gamma):
 preds = clf.predict_proba(test_data)[:,1]
 margin_inds = np.where((preds > (0.5+gamma)) | (preds < (0.5-
gamma)))[0]
 num_errors = np.sum((preds[margin_inds] > 0.5) !=
(test_labels[margin_inds] > 0.0))
 return float(num_errors)/len(margin_inds)
gammas = np.arange(0, 0.5, 0.01)
f = np.vectorize(lambda g: margin_errors(clf, test_data, test_labels,
g))
plt.plot(gammas, f(gammas), linewidth=2)
plt.ylabel('Error rate', fontsize=14)
plt.xlabel('Margin', fontsize=14)
plt.show()
vocab = np.array([z[0] for z in sorted(vectorizer.vocabulary_.items(),
key=lambda x:x[1])])
inds = np.argsort(w)
neg_inds = inds[0:50]
print("Highly negative words: ")
print([str(x) for x in list(vocab[neg_inds])])
pos_inds = inds[-49:-1]
print("Highly positive words: ")
print([str(x) for x in list(vocab[pos_inds])])
