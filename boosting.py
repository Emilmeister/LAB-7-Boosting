from __future__ import annotations

from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        y : array-like, форма (n_samples,)
            Массив целевых значений.
        predictions : array-like, форма (n_samples,)
            Предсказания текущего ансамбля.

        Примечания
        ----------
        Эта функция добавляет новую модель и обновляет ансамбль.
        """
        loss_to_optimize = -self.loss_derivative(y, predictions)

        base_model = self.base_model_class(**self.base_model_params)
        base_model.fit(x, loss_to_optimize)
        best_gamma = self.find_optimal_gamma(y, predictions, base_model.predict(x))

        self.gammas.append(best_gamma * self.learning_rate)
        self.models.append(base_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        x_train : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y_train : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        x_valid : array-like, форма (n_samples, n_features)
            Массив признаков для валидационного набора.
        y_valid : array-like, форма (n_samples,)
            Массив целевых значений для валидационного набора.
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        stop_count = 0

        for i in range(self.n_estimators):
            boot_indexes = np.random.choice(range(y_train.shape[0]), size=int(y_train.shape[0] * self.subsample),
                                            replace=True)
            x_boot_sample = x_train[boot_indexes]
            y_boot_sample = y_train[boot_indexes]
            train_predictions_boot_sample = train_predictions[boot_indexes]
            self.fit_new_base_model(x_boot_sample, y_boot_sample, train_predictions_boot_sample)
            train_predictions = self.predict_proba(x_train)[:, 1]
            valid_predictions = self.predict_proba(x_valid)
            val_score = self.score(x_valid, y_valid)

            if self.plot:
                print(f"ESTIMATOR {i + 1}: Validation score: {val_score}")

            self.history["val_score"].append(val_score)
            self.history["train_loss"].append(self.loss_fn(y_train, self.predict_proba(x_train)[:, 1]))
            self.history["valid_loss"].append(self.loss_fn(y_valid, self.predict_proba(x_valid)[:, 1]))

            if self.early_stopping_rounds is not None:
                if len(self.history["valid_loss"]) > 2 and self.history["valid_loss"][-1] > self.history["valid_loss"][
                    -2]:
                    stop_count += 1
                    if stop_count >= self.early_stopping_rounds:
                        print("Early stopping")
                        break
                else:
                    stop_count = 0

        if self.plot:
            plt.plot(self.history['val_score'])
            plt.title('validation score')
            plt.xlabel('n_estimators')
            plt.ylabel('score')
            plt.show()

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, n_classes)
            Вероятности для каждого класса.
        """
        margin = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            margin += gamma * model.predict(x)

        proba = self.sigmoid(margin)
        return np.array([1 - proba, proba]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        result = np.zeros(len(self.models[0].feature_importances_))
        for model, gamma in zip(self.models, self.gammas):
            result += gamma * model.feature_importances_

        return result/(len(self.models))
