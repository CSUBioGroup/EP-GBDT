import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import random,pickle,argparse
np.random.seed(202010086)
random.seed(202010086)

"""
type is the selected species, 
threshold is for the final ensemble classifier, 
num is the number of classifiers. 
"""
parser = argparse.ArgumentParser()
parser.add_argument('--type', required=True, help='"h" for human, "y" for yeast.')
parser.add_argument('--threshold', required=True, help='threshold for model select.')
parser.add_argument('--num', default='49', help='num of base models.')

args = parser.parse_args()

def get_base_model(model:str):
    assert model in pool
    if model=='SVM':
        return SVC(probability=True, kernel='rbf', degree=4) #
    elif model=='DT':
        return DecisionTreeClassifier()
    elif model=='LR':
        return LogisticRegression(penalty='l2', class_weight='balanced')
    elif model=='NB':
        return GaussianNB()
    elif model=='GBDT':
        return GradientBoostingClassifier(min_samples_leaf=1, max_leaf_nodes=63, max_depth=6, learning_rate=0.1, 
                                          n_estimators=500, n_iter_no_change=20)

if __name__=='__main__':
    ty,auc_thres,k = str(args.type),float(args.threshold),int(args.num)
    # load the data
    with open(f'data_{ty}.pkl', 'rb') as f:
        data = pickle.load(f)
    # X_train,y_train,X_test,y_test = data['X_train'],data['y_train'],data['X_test'],data['y_test']
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'].reshape(10789,1), data['X_test'], data['y_test'].reshape(630,1)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    print(np.sum(y_test))
    pool = ['GBDT']
    # train base models
    models,i = [],1

    while True:
        # use RandomUnderSampler to sample
        rus = RandomUnderSampler(random_state=random.randint(1000,9999))
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        # train
        locModel = random.choice(pool)
        clf = get_base_model(locModel)
        clf.fit(X_resampled, y_resampled)
        # predict
        ypro_pre = clf.predict_proba(X_test)
        y_pre = ypro_pre.argmax(axis=1).reshape(-1,1)
        # evaluate
        acc = (sum(y_pre==y_test)/len(y_pre))[0]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, ypro_pre[:, 1])
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, ypro_pre[:, 1])
        auc = metrics.auc(fpr, tpr)
        pre = metrics.precision_score(y_test, ypro_pre[:, 1]>0.5)
        rec = metrics.recall_score(y_test, ypro_pre[:, 1]>0.5)
        f1 = metrics.f1_score(y_test, ypro_pre[:, 1]>0.5)
        aupr = metrics.average_precision_score(y_test, ypro_pre[:, 1])
        print(f'Base Model {i}: {locModel}')
        print(f'ACC= {acc:.3f}; PRE= {pre:.3f}; REC= {rec:.3f}; F1= {f1:.3f}; AUC= {auc:.3f}; AUPR= {aupr:.3f}')
        if auc>auc_thres: models.append(clf)
        else: print('AUC is less than threshold, drop this model!')
        print(f'now: {len(models)}/{k}')
        print()
        if len(models)>=k:
            break
        i += 1
    print(f'Get {len(models)} base models.')
    
    # model ensemble and vote
    final_proba = np.mean([clf.predict_proba(X_test) for clf in models], axis=0)
    final_label = final_proba.argmax(axis=1).reshape(-1,1)
    # evaluate the ensembled results
    acc = (sum(final_label==y_test)/len(y_test))[0]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, final_proba[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, final_proba[:, 1])
    print(fpr)
    print("XXXXXXXXXXXXXXXX")
    print(tpr)
    print("XXXXXXXXXXXXXXXX")
    print(precision)
    print("XXXXXXXXXXXXXXXX")
    print(recall)
    auc = metrics.auc(fpr, tpr)
    pre = metrics.precision_score(y_test, final_proba[:, 1] > 0.5)
    rec = metrics.recall_score(y_test, final_proba[:, 1] > 0.5)
    f1 = metrics.f1_score(y_test, final_proba[:, 1] > 0.5)
    aupr = metrics.average_precision_score(y_test, final_proba[:, 1])
    print(f'The final evaluation: '
          f'ACC= {acc:.3f}; PRE= {pre:.3f}; REC= {rec:.3f}; F1= {f1:.3f}; AUC= {auc:.3f}; AUPR= {aupr:.3f}.')
    # save the unsembled models
    with open(f'SVC{int(round(1000*auc))}_{ty}.pkl', 'wb') as f:
        pickle.dump(models, f)


    with open(f'fpr_tpr.csv', 'a') as f:
        f.write('fpr, tpr\n')
        for i in range(len(fpr)):
            f.write('%0.6f, %0.6f\n' % (fpr[i], tpr[i]))

    with open(f'pre_rec.csv', 'a') as f:
        f.write('precision, recall\n')
        for i in range(len(precision)):
            f.write('%0.6f, %0.6f\n' % (precision[i], recall[i]))