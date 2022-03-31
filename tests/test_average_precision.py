import os
import unittest
import numpy as np
import numpy.testing as nptest


from flare.utils.average_precision import ap_interp
from flare.utils.average_precision import ap_pr
from flare.utils.average_precision import plot_pr_curve
from flare.utils.average_precision import plot_rt_curve



class TestApInterp(unittest.TestCase):

    def test_standard(self):
        ## Inputs:
        recall    = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.7, 0.8, 0.9, 1.0])
        precision = np.asarray([1.00, 0.50, 0.40, 0.67, 0.50, 0.50,
                                0.57, 0.44, 0.50, 0.47, 0.0])

        ## Expected results:
        ap_e = 0.619

        ## Results:
        ap = ap_interp(precision, recall)

        ## Tests:
        nptest.assert_allclose(ap, ap_e)




class TestApPr(unittest.TestCase):

    def test_standard(self):
        ## Inputs:
        gt         = [np.asarray([[1, 1, 1],
                                  [4, 4, 2.]])]
        detections = [np.asarray([[4, 3, 2],
                                  [1, 1, 1],  # trigger sorting with low score.
                                  [5, 3, 2],  # <- double detection
                                  [2, 2, 1.]])]
        scores = [np.asarray([0.9, 0.2, 0.8, 0.6])]

        ## Expected results:
        thresholds_e = np.asarray([0.9, 0.9, 0.8, 0.6, 0.2, 0.2])
        precision_e  = np.asarray([1,   1, 1/2, 1/3, 1/2, 0])
        recall_e     = np.asarray([0, 1/2, 1/2, 1/2,   1, 1])
        ap_e = 0.75

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)

        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)



    def test_1gt(self):
        ## Inputs:
        gt         = [np.asarray([[1, 1, 1.]])]
        detections = [np.asarray([[4, 3, 2],
                                  [1, 1, 1],  # trigger sorting with low score.
                                  [5, 3, 2],  # <- double detection
                                  [2, 2, 1.]])]
        scores = [np.asarray([0.9, 0.2, 0.8, 0.6])]

        ## Expected results:
        thresholds_e = np.asarray([0.9, 0.9, 0.8, 0.6, 0.2, 0.2])
        precision_e  = np.asarray([0, 0, 0, 0, 1/4, 0])
        recall_e     = np.asarray([0, 0, 0, 0,   1, 1])
        ap_e = 0.25

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)

        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)
        
        

    def test_ap0(self):
        ## Inputs:
        gt         = [np.asarray([[1, 2, 1.],
                                  [6, 4, 2.]])]
        detections = [np.asarray([[4, 1, 2.]])]
        scores = [np.asarray([0.4])]

        ## Expected results:
        thresholds_e = np.asarray([0.4, 0.4, 0.4])
        precision_e  = np.asarray([0, 0, 0.])
        recall_e     = np.asarray([0, 0, 0.])
        ap_e = 0.0

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)

        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)



    def test_ap1(self):
        ## Inputs:
        gt         = [np.asarray([[1, 3, 1],
                                  [5, 2, 2.]])]
        detections = [np.asarray([[1.5, 3, 1],
                                  [6.0, 2, 2]])]
        scores = [np.asarray([0.5, 0.7])]

        ## Expected results:
        thresholds_e = np.asarray([0.7, 0.7, 0.5, 0.5])
        precision_e  = np.asarray([1,   1, 1, 0.])
        recall_e     = np.asarray([0, 1/2, 1, 1.])
        ap_e = 1.0

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)

        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)



    def test_multiple_samples(self):
        ## Inputs:
        gt         = [np.asarray([[1, 1, 1],
                                  [4, 4, 2.]]),
                      np.asarray([[1, 2, 1],
                                  [6, 4, 2.]]),
                      np.asarray([[1, 3, 1],
                                  [5, 2, 2.]])]
        detections = [np.asarray([[4, 3, 2],
                                  [1, 1, 1],  # trigger sorting with low score.
                                  [5, 3, 2],  # <- double detection
                                  [2, 2, 1.]]),
                      np.asarray([[4, 1, 2.]]),
                      np.asarray([[1.5, 3, 1],
                                  [6.0, 2, 2]])]
        scores = [np.asarray([0.9, 0.2, 0.8, 0.6]),
                  np.asarray([0.4]),
                  np.asarray([0.5, 0.7])]

        ## Expected results:
        thresholds_e = np.asarray([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.2])
        precision_e  = np.asarray([1,   1, 1/2, 2/3, 1/2, 3/5, 1/2, 4/7, 0])
        recall_e     = np.asarray([0, 1/6, 1/6, 1/3, 1/3, 1/2, 1/2, 2/3, 2/3])
        ap_e = 0.473015873015873

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)
        
        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)



    def test_multiple_samples_1xNoDet(self):
        ## Inputs:
        gt         = [np.asarray([[1, 1, 1],
                                  [4, 4, 2.]]),
                      np.asarray([[1, 2, 1],
                                  [6, 4, 2.]]),
                      np.asarray([[1, 3, 1],
                                  [5, 2, 2.]])]
        detections = [np.asarray([[4, 3, 2],
                                  [1, 1, 1],  # trigger sorting with low score.
                                  [5, 3, 2],  # <- double detection
                                  [2, 2, 1.]]),
                      np.asarray([]),
                      np.asarray([[1.5, 3, 1],
                                  [6.0, 2, 2]])]
        scores = [np.asarray([0.9, 0.2, 0.8, 0.6]),
                  np.asarray([]),
                  np.asarray([0.5, 0.7])]

        ## Expected results:
        thresholds_e = np.asarray([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2])
        precision_e  = np.asarray([1,   1, 1/2, 2/3, 2/4, 3/5, 4/6, 0])
        recall_e     = np.asarray([0, 1/6, 1/6, 2/6, 2/6, 3/6, 4/6, 4/6])
        ap_e = 0.5

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)
        
        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)



    def test_multiple_samples_1xNoGt(self):
        ## Inputs:
        gt         = [np.asarray([[1, 1, 1],
                                  [4, 4, 2.]]),
                      np.asarray([]),
                      np.asarray([[1, 3, 1],
                                  [5, 2, 2.]])]
        detections = [np.asarray([[4, 3, 2],
                                  [1, 1, 1],  # trigger sorting with low score.
                                  [5, 3, 2],  # <- double detection
                                  [2, 2, 1.]]),
                      np.asarray([[4, 1, 2.]]),
                      np.asarray([[1.5, 3, 1],
                                  [6.0, 2, 2]])]
        scores = [np.asarray([0.9, 0.2, 0.8, 0.6]),
                  np.asarray([0.4]),
                  np.asarray([0.5, 0.7])]

        ## Expected results:
        thresholds_e = np.asarray([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.2])
        precision_e  = np.asarray([1,   1, 1/2, 2/3, 1/2, 3/5, 1/2, 4/7, 0])
        recall_e     = np.asarray([0, 1/4, 1/4, 2/4, 2/4, 3/4, 3/4, 4/4, 1])
        ap_e = 0.7095238095238094

        ## Results:
        ap, precision, recall, thresholds = ap_pr(detections, scores, gt, 0.3)

        ## Tests:
        nptest.assert_allclose(thresholds, thresholds_e)
        nptest.assert_allclose(precision, precision_e)
        nptest.assert_allclose(recall, recall_e)
        nptest.assert_allclose(ap, ap_e)




class TestPlotPrCurve(unittest.TestCase):

    f_path='./test_average_precision/pr_'
    
    def setUp(self):
        os.makedirs(self.f_path, exist_ok=True)
        pass
        

    def test_one_curve(self):
        ## Inputs:
        recall    = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.7, 0.8, 0.9, 1.0])
        precision = np.asarray([1.00, 0.50, 0.40, 0.67, 0.50, 0.50,
                                0.57, 0.44, 0.50, 0.47, 0.0])

        ## Results (for visual insprection as file):
        title = 'one_curve'
        f_path = self.f_path + title
        plot_pr_curve(precision[None, :], recall[None, :],
                      ['the label'], title=title,
                      f_path=f_path, show=False)


    def test_two_curves(self):
        ## Inputs:
        recall1    = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.7, 0.8, 0.9, 1.0])
        precision1 = np.asarray([1.00, 0.50, 0.40, 0.67, 0.50, 0.50,
                                0.57, 0.44, 0.50, 0.47, 0.0])
        recall2    = np.sort(np.random.random(40))
        precision2 = np.random.random(40)
        recall     = [recall1, recall2]
        precision  = [precision1, precision2]

        ## Results (for visual insprection as file):
        title = 'two_curves'
        f_path = self.f_path + title
        plot_pr_curve(precision, recall,
                      ['the label', 'random'], title=title,
                      f_path=f_path, show=False)




class TestPlotRtCurve(unittest.TestCase):

    f_path='./test_average_precision/rt_'
    
    def setUp(self):
        os.makedirs(self.f_path, exist_ok=True)
        pass


    def test_one_curve(self):
        ## Inputs:
        recall     = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.6, 0.7, 0.8, 0.9, 1.0])
        thresholds = np.asarray([1.00, 0.50, 0.40, 0.67, 0.50, 0.50,
                                 0.57, 0.44, 0.50, 0.47, 0.0])

        ## Results (for visual insprection as file):
        title = 'one_curve'
        f_path = self.f_path + title
        plot_rt_curve(recall[None, :], thresholds[None, :],
                      ['the label'], title=title,
                      f_path=f_path, show=False)


    def test_two_curves(self):
        ## Inputs:
        recall1     = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.6, 0.7, 0.8, 0.9, 1.0])
        thresholds1 = np.asarray([1.00, 0.50, 0.40, 0.67, 0.50, 0.50,
                                0.57, 0.44, 0.50, 0.47, 0.0])
        recall2     = np.sort(np.random.random(40))
        thresholds2 = np.random.random(40)
        recall      = [recall1, recall2]
        thresholds  = [thresholds1, thresholds2]

        ## Results (for visual insprection as file):
        title = 'two_curves'
        f_path = self.f_path + title
        plot_rt_curve(recall, thresholds,
                      ['the label', 'random'], title=title,
                      f_path=f_path, show=False)




if __name__ == '__main__':
    unittest.main()
