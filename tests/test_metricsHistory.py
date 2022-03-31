import os
import unittest
import numpy as np
import numpy.testing as nptest

from flare.utils.metricsHistory import MetricsHistory as MH

# Supress matlab logging messages: (it will complain about transparency in eps)
import logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)



def p2tf(filename):
    return os.path.join(os.path.dirname(__file__), filename)



class TestMetricsHistory(unittest.TestCase):

    dir = './test_metricsHistory/'
    fname = p2tf('./data/test_metricsHistory_log.txt')
    columns = ['epoch', 'lr', 'loss', 'acc', 'loss_val', 'acc_val']
    epoch0 = [0, 0.05, 101, 0.6512501538, 0.9728303651, 0.6743673225]
    epoch1 = [1, 0.05, 0.6963822263, 0.8230942555, 0.7318430438, 0.7789006050]
    epoch9 = [9, 0.035, 0.346611629, 0.8914471942, 0.4436489566, 0.8603768080]
    epoch10 = [10, 0.03, 0.42, 0.88, 0.44, 0.86]


    def setUp(self):
        os.makedirs(self.dir, exist_ok=True)
        pass


    def test_load_from_logfile(self):
        hist = MH.from_logfile(self.fname)

        self.assertEqual(hist.columns.to_list(), self.columns)
        nptest.assert_allclose(hist.iloc[0].values, self.epoch0)
        nptest.assert_allclose(hist.iloc[1].values, self.epoch1)
        nptest.assert_allclose(hist.iloc[9].values, self.epoch9)
        nptest.assert_allclose(hist.iloc[10].values, self.epoch10)


    def test_plot_all_compressed(self):
        hist = MH.from_logfile(self.fname)
        hist.plot_all_compressed(self.dir, show=False)

        
    def test_append(self):
        hist = MH.from_logfile(self.fname)
        hist = hist.append({'epoch': 11, 'lr': .41, 'loss': .42, 'acc': 43,
                            'loss_val': .44, 'acc_val': 45}, ignore_index=True)

        self.assertEqual(hist.columns.to_list(), self.columns)
        nptest.assert_allclose(hist.iloc[0].values, self.epoch0)
        nptest.assert_allclose(hist.iloc[10].values, self.epoch10)
        nptest.assert_allclose(hist.iloc[11].values, [11, .41, .42, 43, .44, 45])


if __name__ == '__main__':
    unittest.main()
