import numpy as np

class bf_KNN():
    def __init__(self, **kw):
        self.points = {}

    def _distance(self, x1, x2):
        return np.linalg.norm(x1-x2)
    
    def _add_point(self, X, y):
        if y not in self.points.keys():
            self.points[y] = []

        self.points[y] += [X]

    def _point_distances(self, x):
        dists = {}
        for cl in self.points.keys():
            d = list(map(lambda p: self._distance(x, p), self.points[cl]))
            dists[cl] = d,
        return dists

    def _closer_class(self, x):
        min_dist =  None
        p_cl = None

        for cl in self.points.keys():
            for point in self.points[cl]:
                dist = self._distance(point, x)
                if min_dist is None or min_dist > dist:
                    min_dist = dist
                    p_cl = cl

        return p_cl, min_dist

    def fit(self, X, y):
        #print(X.shape, y.shape)

        for i in range(len(y)):
            self._add_point(X[i], y[i])

    def predict(self, X):
        Y_ = []
        for x in X:
            dists = self._point_distances(x)
            #print(dists)
            #p_cl, dist = self._closer_class(dists)
            p_cl, dist = self._closer_class(x)
            Y_ += [p_cl]

        #print(Y_)
        return Y_
            
