from model import AbstractModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import config as cfg
import os


MODELS = [
    {
        'name': 'LinearRegression',
        'model': LinearRegression()
    },
    {
        'name': 'RandomForestRegressor',
        'model': RandomForestRegressor(n_estimators=5, 
                                       random_state=25, 
                                       max_depth=21, 
                                       max_features=8,
                                       min_samples_leaf=1,
                                       min_samples_split=12)
    }
]


def load_data(data_path, mode='train'):
    X = pickle.load(open(os.path.join(data_path, 'X_{}.pkl'.format(mode)), 'rb'))
    y = pickle.load(open(os.path.join(data_path, 'y_{}.pkl'.format(mode)), 'rb'))

    return X, y


def train():
    X_train, y_train = load_data(cfg.DATA_PATH, 'train')
    X_test, y_test = load_data(cfg.DATA_PATH, 'test')
    X_2018, y_2018 = load_data(cfg.DATA_PATH, '2018')

    for model_meta in MODELS:
        print('Fit model {}'.format(model_meta['name']))
        model = AbstractModel(model_meta['model'])
        model.fit(X_train, y_train)

        print('Test score: {}'.format(model.score(X_test, y_test)))
        print('2018 score: {}'.format(model.score(X_2018, y_2018)))

        y_pred = model.predict(X_2018)
        plt.plot(y_2018, 'ro')
        plt.plot(y_pred, 'g-')
        plt.show()


if __name__ ==  '__main__':
    train()
