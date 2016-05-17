from models.ml import Gym

# TODO Make this actually useful...
class TimeSeriesGym(Gym):

    def __init__(self, ts_cls, order):
        self.ts_cls = ts_cls
        self.order = order

    # Note we cannot use LOO training method since it ARMA library initializes params randomly
    # And fits with bfgs. Kalman filter is used to fit residual MA.
    def train(self, ys):
        return self.ts_cls.fit(ys, self.order)

    def select_best_model(self, ys):
        return self.train(ys)