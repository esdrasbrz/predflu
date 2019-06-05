from sklearn.metrics import mean_squared_error

class AbstractModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return mean_squared_error(y_true, y_pred)
