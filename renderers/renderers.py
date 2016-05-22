import os
import numpy as np
from abc import abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
import utils
from models import ml
from models.models import Axis


class GraphWriter(object):

    def __init__(self, stock, output_dir=os.getcwd()):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.stock = stock
        self.output_dir = output_dir

    def residuals_plot(self, algo, residuals, dates):
        start_date_str = utils.pretty_date_str(dates[0])
        end_date_str = utils.pretty_date_str(dates[-1])
        title = '{algo} {stock} Residual Plot {start} - {finish}'.format(stock=self.stock, start=start_date_str,
                                                                  finish=end_date_str, algo=algo)
        residuals_mean = np.mean(residuals) * np.ones(residuals.shape)
        residuals_df = pd.DataFrame({'Dates': dates, 'Residuals': residuals, 'Mean': residuals_mean})
        rax = residuals_df.plot(x='Dates', y='Residuals', style='ro', title=title, c='blue')
        residuals_df.plot(x='Dates', y='Mean', c='Red', ax=rax)
        name = "{stock}-{algo}-residual-{start}-{finish}.png".format(stock=self.stock,
                                                                     start=utils.date_file_str(dates[0]),
                                                                     finish=utils.date_file_str(dates[-1]),
                                                                     algo=algo)
        plt.savefig(os.path.join(self.output_dir, name), bbox_inches='tight', figsize=(10, 10))
        plt.close()

    def comparison_plot(self, dates, algo1, y1_preds, algo2, y2_preds, y_measure_name, y_measure):
        title = '{stock} Next Day Online Prediction {start} - {finish}'\
            .format(stock=self.stock, start=utils.pretty_date_str(dates[0]), finish=utils.pretty_date_str(dates[-1]))
        df = pd.DataFrame({'Dates': dates, algo1: y1_preds.flatten(), algo2: y2_preds.flatten(),
                           y_measure_name: y_measure.flatten()})
        df.plot(x='Dates', y=[y_measure_name, algo1, algo2], title=title, color=['blue', 'green', 'red'], marker='x',
                figsize=(10, 10), linestyle=' ')
        plot_name = '{stock}-comparison-plot-{start}-{finish}.png'.format(stock=self.stock,
                                                                          start=utils.date_file_str(dates[0]),
                                                                          finish=utils.date_file_str(dates[-1]))
        plt.savefig(os.path.join(self.output_dir, plot_name), bbox_inches='tight', figsize=(10, 10))
        plt.close()

    def cse_plot(self,dates, algo1, algo1_preds, algo2, algo2_preds, ys, loss_func=ml.calculate_se, plot_actuals=False):
        title = "Cumulative Sum  Squared Error of {algo1} against {algo2}".format(algo1=algo1, algo2=algo2)
        err1 = loss_func(ys, algo1_preds)
        err2 = loss_func(ys, algo2_preds)
        cse1 = np.zeros(ys.shape)
        cse1[:, :, 0] = err1[:, :, 0]
        cse2 = np.zeros(ys.shape)
        cse2[:, :, 0] = err2[:, :, 0]
        for t in range(1, ys.shape[Axis.time]):
            cse1[:, :, t] = cse1[:, :, t-1] + err1[:, :, t]
            cse2[:, :, t] = cse2[:, :, t-1] + err2[:, :, t]

        lkey = 'Cumulative {algo} SE'
        algo1_lkey = lkey.format(algo=algo1)
        algo2_lkey = lkey.format(algo=algo2)
        prices_lkey = '{stock} Prices'.format(stock=self.stock)
        algo1_pred_lkey = '{algo1} Predictions'.format(algo1=algo1)
        algo2_pred_lkey = '{algo2} Predictions'.format(algo2=algo2)
        df = pd.DataFrame({'Dates': dates, algo1_lkey: cse1.flatten(), algo2_lkey: cse2.flatten(),
                           prices_lkey: ys.flatten(), algo1_pred_lkey: algo1_preds.flatten(),
                           algo2_pred_lkey: algo2_preds.flatten()})
        y_axis = [algo1_lkey, algo2_lkey]
        if plot_actuals:
            y_axis += [prices_lkey, algo1_pred_lkey, algo2_pred_lkey]
        df.plot(x='Dates', y=y_axis, title=title)
        plot_name = "{stock}-CSE-plot-{start}-{finish}.png".format(stock=self.stock, start=utils.date_file_str(dates[0]),
                                                                   finish=utils.date_file_str(dates[-1]))
        plt.savefig(os.path.join(self.output_dir, plot_name), bbox_inches='tight', figsize=(10, 10))
        plt.close()



class CSVReportWriter(object):

    def __init__(self, filename, output_dir=os.getcwd()):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.filename = os.path.join(output_dir, filename)
        self.render_title()

    def render_title(self):
        if os.path.exists(self.filename):
            return
        self.render("Date From", "Till Date", "Stock", "Algorithm", "AAE", "RMSE", "MAPE", "MSPE")

    def render(self, *args):
        mode = 'a' if os.path.exists(self.filename) else 'w'
        with open(self.filename, mode) as f:
            f.write(",".join(list(args)) +"\n")
            f.close()