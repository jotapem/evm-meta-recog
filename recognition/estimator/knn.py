from sklearn.neighbors import KNeighborsClassifier as sk_KNN

class KNN(sk_KNN):
    def __init__(self, **kw):
        super().__init__(**kw)

        
    def predict_with_prop(self, X):
        Y_ = self.predict(X)
        Y_kdist, Y_kindexes = self.kneighbors(X)

        
        Y_better = list(map(lambda i: [Y_[i], str(Y_kdist[i][0])], range(len(Y_kdist))))
            
        return Y_better
            
