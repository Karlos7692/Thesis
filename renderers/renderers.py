import os
import numpy as np
from abc import abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
import utils


class GraphWriter(object):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def residuals_plot(self, stock, residuals, start_date, end_date, tag=""):
        start_date_str = utils.date_str(start_date)
        end_date_str = utils.date_str(end_date)
        title = '{stock} Residual Plot {start} - {finish}'.format(stock=stock, start=start_date_str,
                                                                  finish=end_date_str)
        residuals_mean = np.mean(residuals) * np.ones(residuals.shape)
        residuals_df = pd.DataFrame({'Days': list(range(residuals.shape[0])), 'Residuals': residuals, 'Mean': residuals_mean})
        rax = residuals_df.plot(x='Days', y='Residuals', kind='scatter', title=title)
        residuals_df.plot(x='Days', y='Mean', c='Red', ax=rax)
        name = "{stock}-residual-{start}-{finish}{tag}.png".format(stock=stock, start=start_date_str,
                                                                   finish=end_date_str, tag=tag)
        plt.savefig(os.path.join(self.output_dir, name), bbox_inches='tight')


    def comparison_plot(self, y1_preds, y2_preds, y_measure):
        pass



class CSVReportWriter(object):

    def __init__(self, filename, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.filename = os.path.join(output_dir, filename)
        self.render_title()

    @abstractmethod
    def render_title(self):
        pass

    def render(self, *args):
        with open(self.filename, 'a') as f:
            f.write(",".join(args))
            f.close()


class Exp1CsvWriter(CSVReportWriter):

    def render_title(self):
        self.render(["AAE", "RMSE", "MAPE", "MSPE"])