from model import AbstractModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import config as cfg
import os


MODELS = [
    {
        'name': 'LinearRegression',
        'model': LinearRegression(),
        'pca': None,
    },
    {
        'name': 'RandomForestRegressor',
        'model': RandomForestRegressor(n_estimators=5, 
                                       random_state=25, 
                                       max_depth=21, 
                                       max_features=8,
                                       min_samples_leaf=1,
                                       min_samples_split=12),
        'pca': None
    },
    {
        'name': 'MLP',
        'model': MLPRegressor(hidden_layer_sizes=(256,128,64),
                              activation='relu'),
        'pca': None,
    },
    {
        'name': 'MLP PCA 6',
        'model': MLPRegressor(hidden_layer_sizes=(256,128,64),
                              solver='adam',
                              activation='relu'),
        'pca': PCA(n_components=6),
    },
]


def load_data(data_path, mode='train'):
    X = pickle.load(open(os.path.join(data_path, 'X_{}.pkl'.format(mode)), 'rb'))
    y = pickle.load(open(os.path.join(data_path, 'y_{}.pkl'.format(mode)), 'rb'))

    return X, y


def train():
    X_train, y_train = load_data(cfg.DATA_PATH, 'train')
    X_2018, y_2018 = load_data(cfg.DATA_PATH, '2018')
    print(X_train.shape)

    for model_meta in MODELS:
        print('Fit model {}'.format(model_meta['name']))
        model = AbstractModel(model_meta['model'], pca=model_meta['pca'])
        model.fit(X_train, y_train)

        print('2018 score: {}'.format(model.score(X_2018, y_2018)))

        if cfg.PLOT_REGRESSION:
            y_pred = model.predict(X_2018)
            plt.plot(y_2018, 'ro')
            plt.plot(y_pred, 'g-')
            plt.show()


if __name__ ==  '__main__':
    train()
