import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def sklearn_models(X_train, X_test, y_train, y_test, model_type):
    if model_type == 'GaussianNB':
        model = GaussianNB()
    elif model_type == 'SVC':
        model = SVC(verbose=True)
    elif model_type == 'KNNeighbors':
        model = KNeighborsClassifier(verbose=True)
    else:
        model = KMeans()

    start = time.time()
    print('Started training: {}'.format(start))
    import pdb; pdb.set_trace()
    model.fit(X_train, y_train)
    end = time.time()
    print('Completed training model: {}'.format(end - start))

    start = time.time()
    print('Completed training model: {}'.format(start))
    score = model.score(X_test, y_test)
    end = time.time()
    print('Completed scoring: Time: {} Score: {}'.format(end-start, score))
    pdb.set_trace()
    return model