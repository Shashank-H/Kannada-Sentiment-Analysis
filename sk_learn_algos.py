import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import string
import numpy as np
import matplotlib.pyplot as plt


accuracy_plot = []
recall_plot = []
f1_plot = []
precision_plot = []

corpus = pd.read_excel("data/dataset.xlsx")
print('Kannada Corpus : ', len(corpus))
review_positive = len([x for x in corpus['Sentence'] if x == 1])
review_negative = len([x for x in corpus['Sentence'] if x == -1])

no_punctuations = []



x = corpus['Sentence']
print(x)
y = corpus['Tag']

vect = CountVectorizer(ngram_range=(1, 1), max_df=.80, min_df=4)

# create X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

# Using training data to transform text into counts of features for each review
vect.fit(X_train)
X_train_dtm = vect.transform(X_train).toarray()
X_test_dtm = vect.transform(X_test)

attributes = {}


def Display(model, name):
    model.fit(X_train_dtm, y_train)
    y_predicted = model.predict(X_test_dtm.toarray())
    conf_matrix = confusion_matrix(y_test, y_predicted)

    tp_nb = conf_matrix[1, 1]
    tn_nb = conf_matrix[0, 0]
    fp_nb = conf_matrix[1, 0]
    fn_nb = conf_matrix[0, 1]

    accuracy = (tp_nb + tn_nb) / (tp_nb + tn_nb + fp_nb + fn_nb)
    print('Accuracy ', accuracy)
    precision = tp_nb / (tp_nb + fp_nb)
    print('Precision ', precision)
    recall = tp_nb / (tp_nb + fn_nb)
    print('Recall ', recall)
    f1_score = 2 * precision * recall / (precision + recall)
    print('F1_Score', f1_score)
    print("\n")
    attributes[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1score': f1_score
    }


def Plot(performance_arg, label):
    objects = ('SVM', 'Multinomial', 'GaussianNB','KNN-Classifier')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance_arg, align='center', alpha=1, color='r')
    plt.xticks(y_pos, objects)
    plt.ylabel(label)
    plt.title('Algorithm '+label+' Comparision')
    plt.savefig("outputs/graphs/"+label+".png");
    plt.show()


NB = MultinomialNB()
SVM = LinearSVC()
GN = GaussianNB()
KNN = KNeighborsClassifier()

Display(SVM, name="SVM")

Display(NB, name="Multinomial")

Display(GN, name="GaussianNB")

Display(KNN, name="KNN-Classifier")

print(attributes)

model_names = ['SVM', 'Multinomial', 'GaussianNB','KNN-Classifier']


for models in model_names:
    accuracy_plot.append(round(attributes[models]['accuracy'] * 100, 2))
    recall_plot.append(round(attributes[models]['recall'] * 100, 2))
    f1_plot.append(round(attributes[models]['f1score'] * 100, 2))
    precision_plot.append(round(attributes[models]['precision'] * 100, 2))


Plot(accuracy_plot, label="Accuracy")
Plot(recall_plot, label="Recall")
Plot(f1_plot, label="F1-Score")
Plot(precision_plot, label="Precision")



#LinePlot of all algorithms
ranges =[0,50,100,150]
plt.xlabel('Algorithms')
plt.ylabel('Value')
plt.xticks(ranges,['SVM','Multinomial','GaussianNB','KNN-Classifier'])

plt.scatter(ranges, accuracy_plot, color='g',label="Accuracy")
plt.scatter(ranges, recall_plot, color='orange',label="Recall")
plt.scatter(ranges, f1_plot, color='blue',label="f1_score")
plt.scatter(ranges, precision_plot, color='red',label="Precision")

plt.legend()
plt.title('Line plot')

plt.plot(ranges, accuracy_plot, color='g',label="Accuracy")
plt.plot(ranges, recall_plot, color='orange',label="Recall")
plt.plot(ranges, f1_plot, color='blue',label="f1_score")
plt.plot(ranges, precision_plot, color='red',label="Precision")

plt.savefig("outputs/graphs/line-plot.png")
plt.show()