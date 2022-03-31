""" by Stefan Schmohl, 2020 """

import os
import numpy as np
import pandas
import matplotlib.pyplot as plt



class MetricsHistory(pandas.DataFrame):
    """ Class to record and plot metrics during / after training """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @classmethod
    def from_logfile(cls, file_path):
        """ Reconstruct a class instance from a log file created by CSVLogger.

        """

        # Read first line as header line for column names:
        f = open(file_path, 'r')
        lines = f.read().splitlines()
        f.close()
        keys = lines[0].split(',')
        keys = [k.strip() for k in keys]

        # Remove header lines (can also apear mid file in case ot resumed training):
        for i, line in reversed(list(enumerate(lines, start=0))):
            if line[0].isalpha():
                del lines[i]

        # Read data into dict with keys from headerline
        data = np.genfromtxt(lines, delimiter=',')

        if len(data.shape) == 1:   # in case only one line in logfile
            data = data[None, :]

        history = {}
        for i, k in enumerate(keys):
            history[k] = data[:, i]

        mh = cls(history)
        return mh
        
        
    @property
    def last_epoch(self):
        return int(self['epoch'].iloc[-1])
        


    def plot_metric_type(self, metric_type, y_factor, ylabel, title,  
                         ylim=None, f_path=None, show=True, ylim_outlier=0):
        """ plot metrics identified by name-part ("metric_type") """
        fig, ax = plt.subplots()
        has_any = False
        y_ = []
        for metric in self.columns:
            if metric_type in metric:
                x = self.index.to_list()
                y = self[metric] * y_factor
                y_.append(y)
                ax.plot(x, y, label=metric)
                has_any = True
        if not has_any:
            # if there is no metric of metric type in self.columns
            return
        ax.set(xlabel='epoch', ylabel=ylabel, title=title)
        if 'loss' in metric_type:
            ax.legend(loc='upper right')
        elif 'm_ap' in metric_type:
            ax.legend(loc='lower right')
        elif 'acc' in metric_type:
            ax.legend(loc='lower right')
        else:
            ax.legend(loc='center right')
        ax.grid(True)
        plt.xlim(0, max(x) + max(x)*.1)
        
        if ylim is not None:
            plt.ylim(ylim)
        elif ylim_outlier != 0:
            # breakpoint()
            y_all = np.asarray(y_).flatten()
            std = np.std(y_all)
            mean = np.mean(y_all)
            y_all = y_all[y_all < mean + std*ylim_outlier]
            ylim = [0, np.max(y_all) * 1.1]
            plt.ylim(ylim)
            

        if f_path is not None:
            fig.savefig(f_path + '_' + metric_type + '.png', dpi=600)
            fig.savefig(f_path + '_' + metric_type + '.pdf', dpi=600)

        if show:
            fig.show()



    def plot_all_compressed(self, path=None, show=True):
        """ Plot all metrics, similar ones compressed into the same plot.

        E.g. 'acc' & 'acc_val'.
        Args:
            path:  Path to save files. None for not saving files.
            show:  Whether to display the plots.
        """
        if path is not None:
            path = os.path.join(path, 'metric_history')

        # acc:
        self.plot_metric_type('acc', 100, 'accuracy [%]',
                              'Accuracy Training History', 
                              [0, 100], path, show)

        # m_ap:
        self.plot_metric_type('m_ap', 1, 'mAP [-]',
                              'mAP Training History', 
                              [0, 1], path, show)

        # loss:
        self.plot_metric_type('loss', 1, 'loss [-]',
                              'Loss Training History', None, path, show,
                              ylim_outlier=5)

        # lr:
        self.plot_metric_type('lr', 1, 'learning rate [-]',
                              'Learning Rate Training History', 
                              None, path, show)

        # rest:
        for c in self.columns:
            if not ('acc' in c or 'm_ap' in c or 'loss' in c or 'lr' in c
                    or c == 'epoch'):
                self.plot_metric_type(c, 1, c, c + ' Training History', 
                                      None, path, show)

        # Save to file for later re-plot:
        if path is not None:
            self.to_pickle(path + '.pkl')
