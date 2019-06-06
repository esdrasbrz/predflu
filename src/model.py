from sklearn.metrics import mean_squared_error, r2_score

class AbstractModel:
    def __init__(self, model, pca=None):
        self.model = model
        self.pca = pca

    def fit(self, X, y):
        if self.pca:
            self.pca.fit(X)
            X = self.pca.transform(X)

        self.model.fit(X, y)

    def predict(self, X):
        if self.pca:
            X = self.pca.transform(X)

        return self.model.predict(X)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)
