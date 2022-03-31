
import os
import unittest
import numpy as np
import numpy.matlib
import numpy.testing as nptest

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import flare.utils.conf_matrix as conf_matrix


NaN = float('NaN')


class TestCreateConfMatrix(unittest.TestCase):


    def setUp(self):
        pass


    def test_same_classes_nolabels(self):
        gt = [3, 2, 1, 3, 2, 0, 2]
        pr = [3, 1, 0, 3, 2, 1, 2]
        cm = [[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 2, 0],
              [0, 0, 0, 2]]

        cm_ = conf_matrix.create(gt, pr)
        nptest.assert_allclose(cm_, cm)


    def test_same_classes_contlabels(self):
        gt = [3, 2, 1, 3, 2, 0, 2]
        pr = [3, 1, 0, 3, 2, 1, 2]
        lpr = [0, 1, 2, 3]
        lpr = [0, 1, 2, 3]
        cm = [[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 2, 0],
              [0, 0, 0, 2]]

        cm_ = conf_matrix.create(gt, pr)
        nptest.assert_allclose(cm_, cm)


    def test_different_classes_nolabels(self):
        gt = [3, 2, 1, 3, 2, 5, 2, 4]   # important that 0 not inlcuded.
        pr = [3, 1, 0, 3, 2, 1, 2, 0]
        cm = [[1, 0, 0, 0],
              [0, 1, 2, 0],
              [0, 0, 0, 2],
              [1, 0, 0, 0],
              [0, 1, 0, 0]]

        cm_ = conf_matrix.create(gt, pr)
        nptest.assert_allclose(cm_, cm)


    def test_different_classes_contlabels(self):
        gt = [3, 2, 1, 3, 2, 5, 2, 4]   # important that 0 not inlcuded.
        pr = [3, 1, 0, 3, 2, 1, 2, 0]
        lgt = [0, 1, 2, 3, 4, 5, 6]
        lpr = [0, 1, 2, 3]
        cm = [[0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 2, 0],
              [0, 0, 0, 2],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0]]

        cm_ = conf_matrix.create(gt, pr, lgt, lpr)
        nptest.assert_allclose(cm_, cm)


    def test_different_classes_gappylabels(self):
        gt = [3, 2, 1, 3, 2, 5, 2, 4]   # important that 0 not inlcuded.
        pr = [3, 1, 0, 3, 2, 1, 2, 0]
        lgt = [1, 2, 3, 4, 5, 6, 9]
        lpr = [0, 1, 2, 3]
        cm = [[1, 0, 0, 0],
              [0, 1, 2, 0],
              [0, 0, 0, 2],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],]

        cm_ = conf_matrix.create(gt, pr, lgt, lpr)
        nptest.assert_allclose(cm_, cm)



class TestAnalyzeConfMatrix(unittest.TestCase):

    def setUp(self):
        pass

    def test_symmetrical(self):
        cm = np.asarray([[171,  23,  16,   0,  0],
                         [ 99, 213,   0, 111,  0],
                         [  0,   0, 420,   0,  0],
                         [  0,   0,   0,   0,  0],
                         [123,   0,   0,   0,  0]])
        sum_correct = 804
        sum_gt_correct = [171, 213, 420, 0, 0]
        sum_pr_correct = [171, 213, 420, 0, 0]
        sum_gt_all = [210, 423, 420, 0, 123]
        sum_pr_all = [393, 236, 436, 111, 0]
        acc = 0.6836734694
        recall = [0.8142857143, 0.5035460993, 1, NaN, 0]
        precision = [0.4351145038, 0.9025423729, 0.9633027523, 0, NaN]
        f1 = [0.56716418, 0.64643399, 0.98130841, NaN, NaN]

        result = conf_matrix.analize(cm, detailed=True)

        nptest.assert_allclose(result[4], sum_correct)
        nptest.assert_allclose(result[5], sum_gt_correct)
        nptest.assert_allclose(result[6], sum_pr_correct)
        nptest.assert_allclose(result[7], sum_gt_all)
        nptest.assert_allclose(result[8], sum_pr_all)
        nptest.assert_allclose(result[9], np.eye(5))
        nptest.assert_allclose(result[0], acc)
        nptest.assert_allclose(result[1], recall)
        nptest.assert_allclose(result[2], precision)
        nptest.assert_allclose(result[3], f1)
        nptest.assert_allclose(result[11], f1)
        nptest.assert_allclose(result[12], f1)


    def test_different_gt_classes(self):
        classes_gt = {0: 'Powerline', 1: 'Tree', 3: 'Fassade', 5: 'Roof', 6: 'Shrub'}
        classes_pr = {0: 'Powerline', 1: 'Vegetation', 2: 'Building'}
        cm = np.asarray([[180,  60,   0],
                         [  0, 210,  30],
                         [  0,  50, 200],
                         [ 90,  20, 400],
                         [  0,  40,  50]])
        label_map = {0:0, 1:1, 3:2, 5:2, 6:1}

        sum_correct = 1030
        sum_gt_correct = [180, 210, 200, 400, 40]
        sum_pr_correct = [180, 250, 600]
        sum_gt_all = [240, 240, 250, 510, 90]
        sum_pr_all = [270, 380, 680]
        acc = 0.7744360902
        recall = [0.75, 0.875, 0.8, 0.7843137255, 0.4444444444]
        precision = [0.6666666666, 0.6578947368, 0.8823529412]
        f1_pr = [0.7058823529, 0.7042253521, 0.8333333333]
        f1_gt = [0.70588235, 0.72413793, 0.75471698, 0.80808081, 0.30769231]
        matches_mask = np.asarray([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 1],
                                   [0, 1, 0]], dtype=bool)

        result = conf_matrix.analize(cm,
                                     list(classes_gt.keys()),
                                     list(classes_pr.keys()),
                                     label_map, detailed=True)

        nptest.assert_allclose(result[4], sum_correct)
        nptest.assert_allclose(result[5], sum_gt_correct)
        nptest.assert_allclose(result[6], sum_pr_correct)
        nptest.assert_allclose(result[7], sum_gt_all)
        nptest.assert_allclose(result[8], sum_pr_all)
        nptest.assert_allclose(result[9], matches_mask)
        nptest.assert_allclose(result[0], acc)
        nptest.assert_allclose(result[1], recall)
        nptest.assert_allclose(result[2], precision)
        nptest.assert_allclose(result[3],  f1_pr)
        nptest.assert_allclose(result[11], f1_pr)
        nptest.assert_allclose(result[12], f1_gt)


    def test_different_pr_classes(self):
        classes_gt = {0: 'Powerline', 3: 'Vegetation', 5: 'Building'}
        classes_pr = {0: 'Powerline', 1: 'Tree', 3: 'Fassade', 5: 'Roof', 6: 'Shrub'}
        cm = np.asarray([[180,   0,   0,  90,   0],
                         [ 60, 210,  50,  20,  40],
                         [  0,  30, 200, 400,  50]])
        label_map = {0:0, 3:[1,6], 5:[3,5]}

        sum_correct = 180+210+40+200+400
        sum_gt_correct = [180, 250, 600]
        sum_pr_correct = [180, 210, 200, 400, 40]
        sum_gt_all = [270, 380, 680]
        sum_pr_all = [240, 240, 250, 510, 90]
        acc = sum_correct / np.sum(cm[:])
        recall = [0.66666667, 0.65789474, 0.88235294]
        precision = [0.75, 0.875, 0.8, 0.78431373, 0.44444444]
        f1_pr = [0.70588235, 0.72413793, 0.75471698, 0.80808081, 0.30769231]
        f1_gt = [0.7058823529, 0.7042253521, 0.8333333333]
        matches_mask = np.asarray([[1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1],
                                   [0, 0, 1, 1, 0]], dtype=bool)

        result = conf_matrix.analize(cm,
                                     list(classes_gt.keys()),
                                     list(classes_pr.keys()),
                                     label_map, detailed=True)


        nptest.assert_allclose(result[4], sum_correct)
        nptest.assert_allclose(result[5], sum_gt_correct)
        nptest.assert_allclose(result[6], sum_pr_correct)
        nptest.assert_allclose(result[7], sum_gt_all)
        nptest.assert_allclose(result[8], sum_pr_all)
        nptest.assert_allclose(result[9], matches_mask)
        nptest.assert_allclose(result[0], acc)
        nptest.assert_allclose(result[1], recall)
        nptest.assert_allclose(result[2], precision)
        nptest.assert_allclose(result[3],  f1_pr)
        nptest.assert_allclose(result[11], f1_pr)
        nptest.assert_allclose(result[12], f1_gt)


    def test_ignore_gt_label(self):
        classes_gt = {0: 'Not Labeled', 1: 'Vegetation', 2: 'Building'}
        cm = np.asarray([[ 30, 440, 170],
                         [ 10, 210,  30],
                         [ 20,  50, 200]])

        sum_correct = 410
        sum_gt_correct = [30, 210, 200]
        sum_pr_correct = [30, 210, 200]
        sum_gt_all = [640, 250, 270]
        sum_pr_all = [60, 260, 230]
        acc = 0.7884615384
        recall = [NaN, 210/250, 200/270]
        precision = [30/60, 210/260, 200/230]
        f1 = [0.0857142857, 0.8235294118, 0.8]
        matches_mask = np.asarray([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]], dtype=bool)

        result = conf_matrix.analize(cm, list(classes_gt.keys()),
                                     detailed=True, ignore_labels=[0])

        nptest.assert_allclose(result[4], sum_correct)
        nptest.assert_allclose(result[5], sum_gt_correct)
        nptest.assert_allclose(result[6], sum_pr_correct)
        nptest.assert_allclose(result[7], sum_gt_all)
        nptest.assert_allclose(result[8], sum_pr_all)
        nptest.assert_allclose(result[9], matches_mask)
        nptest.assert_allclose(result[0], acc)
        nptest.assert_allclose(result[1], recall)
        nptest.assert_allclose(result[2], precision)
        nptest.assert_allclose(result[3], f1)
        nptest.assert_allclose(result[11], f1)
        nptest.assert_allclose(result[12], f1)



class TestPlotConfMatrix(unittest.TestCase):

    dir = './test_conf_matrix_plot/'
    classes = {0: 'c1', 1: 'c3', 3: 'c3', 4: 'c4'}
    cm = np.asarray([[180,  20,   0,   0],
                     [ 90, 210,   0,   0],
                     [  0,   0, 400,   0],
                     [100,   0,   0,   0]])


    def setUp(self):
        os.makedirs(self.dir, exist_ok=True)
        pass


    def test_vals_rel(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir, file_suffix='test_vals_rel',
            rel_vals=True, abs_vals=False, F1=False, show=False)


    def test_vals_abs(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir, file_suffix='test_vals_abs',
            rel_vals=False, abs_vals=True, F1=False, show=False)


    def test_vals_abs_and_rel(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir,
            file_suffix='test_vals_abs_and_rel',
            rel_vals=True, abs_vals=True, F1=False, show=False)


    def test_rel_precision_0(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir,
            file_suffix='test_rel_precision_0',
            rel_vals=True, abs_vals=True, rel_precision=0, F1=True, show=False)


    def test_class_freq_off(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir,
            file_suffix='test_class_freq_off',
            rel_vals=True, abs_vals=True, class_freq=False, F1=False, show=False)


    def test_F1(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir, file_suffix='test_F1',
            rel_vals=True, abs_vals=True, F1=True, show=False)


    def test_iou1(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir, file_suffix='test_IoU1',
            rel_vals=True, abs_vals=True, F1=True, show=False, iou=True)


    def test_iou2(self):
        conf_matrix.plot(
            self.cm, self.classes, path=self.dir, file_suffix='test_IoU2',
            rel_vals=True, abs_vals=True, F1=False, show=False, iou=True)


    def test_medium(self):
        classes = {0: 'c1', 1: 'c2', 3: 'c3', 4: 'c4',
                   5: 'c5', 6: 'c6', 7: 'c7', 8: 'c8'}
        cm = np.asarray([[180,  20,   0,   0],
                         [ 90, 210,   0,   0],
                         [  0,   0, 400,   0],
                         [100,   0,   0,   0]])
        cm = np.matlib.repmat(cm, 2, 2)

        conf_matrix.plot(
            cm, classes, path=self.dir,
            file_suffix='test_medium',
            rel_vals=True, abs_vals=True, F1=True, show=False)


    def test_large(self):
        classes = {0: 'c1', 1: 'c2', 3: 'c3', 4: 'c4',
                   5: 'c5', 6: 'c6', 7: 'c7', 8: 'c8',
                   12: 'c12', 43: 'c43', 55: 'c55', 69: 'c69'}
        cm = np.matlib.repmat(self.cm, 3, 3)

        conf_matrix.plot(
            cm, classes, path=self.dir,
            file_suffix='test_large',
            rel_vals=True, abs_vals=True, F1=True, show=False)


    def test_abs_max(self):
        classes = {0: 'c1', 1: 'c2', 3: 'c3', 4: 'c4',
                   5: 'c5', 6: 'c6', 7: 'c7', 8: 'c8',
                   12: 'c12', 43: 'c43', 55: 'c55', 69: 'c69'}
        cm = np.matlib.repmat(self.cm*550, 3, 3)

        conf_matrix.plot(
            cm, classes, path=self.dir,
            file_suffix='test_abs_max',
            rel_vals=True, abs_vals=True, abs_max=99999, F1=False, show=False)


    def test_different_gt_classes(self):
        classes_gt = {0: 'Powerline', 1: 'Tree', 3: 'Fassade',
                      5: 'Roof', 6: 'Shrub'}
        classes_pr = {0: 'Powerline', 1: 'Vegetation', 2: 'Building'}
        
        cm = np.asarray([[180,  60,   0],
                         [  0, 210,  30],
                         [  0,  50, 200],
                         [ 90,  20, 400],
                         [  0,  40,  50]])

        label_map = {0:0, 1:1, 3:2, 5:2, 6:1}

        conf_matrix.plot(
            cm, classes_gt, path=self.dir,
            file_suffix='test_different_gt_classes',
            rel_vals=True, abs_vals=True, F1=True,
            classes_pred=classes_pr, label_map=label_map, show=False)


    def test_different_pr_classes(self):
        classes_gt = {0: 'Powerline', 3: 'Vegetation', 5: 'Building'}
        classes_pr = {0: 'Powerline', 1: 'Tree', 3: 'Fassade', 5: 'Roof', 6: 'Shrub'}
        
        cm = np.asarray([[180,   0,   0,  90,   0],
                         [ 60, 210,  50,  20,  40],
                         [  0,  30, 200, 400,  50]])
                         
        label_map = {0:0, 3:[1,6], 5:[3,5]}

        conf_matrix.plot(
            cm, classes_gt, path=self.dir,
            file_suffix='test_different_pr_classes',
            rel_vals=True, abs_vals=True, F1=True,
            classes_pred=classes_pr, label_map=label_map, show=False)


    def test_ignore_gt_label(self):
        classes_gt = {0: 'Not Labeled', 1: 'Vegetation', 2: 'Building'}
        cm = np.asarray([[  30, 44000, 17000],
                         [  10,   210,    30],
                         [  20,    50,   200]])

        conf_matrix.plot(
            cm, classes_gt, path=self.dir,
            file_suffix='test_ignore_gt_label',
            rel_vals=True, abs_vals=True, F1=True,
            ignore_labels=[0], show=False)





class TesPrintConfMatrix(unittest.TestCase):

    dir = './test_conf_matrix_print/'

    def setUp(self):
        os.makedirs(self.dir, exist_ok=True)
        pass


    def test_symmetrical(self):
        classes = {0: 'c1', 1: 'c2', 3: 'c3', 4: 'c4', 5: 'c5'}
        cm = np.asarray([[171,  23,  16,   0,  0],
                         [ 99, 213,   0, 111,  0],
                         [  0,   0,4920,   0,  0],
                         [  0,   0,   0,   0,  0],
                         [123,   0,   0,   0,  0]])

        f = open(self.dir + 'conf_matrix_test_symmetrical.txt', 'w')
        conf_matrix.print_to_file(cm, f, classes, indent=4)
        f.close()


    def test_different_gt_classes(self):
        classes_gt = {0: 'Powerline', 1: 'Tree', 3: 'Fassade', 5: 'Roof',
                      6: 'Shrub'}
        classes_pr = {0: 'Powerline', 1: 'Vegetation', 2: 'Building'}
        cm = np.asarray([[180,  60,   0],
                         [  0,2190,  30],
                         [  0,  50, 200],
                         [ 90,  20, 400],
                         [  0,  40,  50]])*1234
        label_map = {0:0, 1:1, 3:2, 5:2, 6:1}

        f = open(self.dir + 'test_different_gt_classes.txt', 'w')
        conf_matrix.print_to_file(cm, f, classes_gt, classes_pr, label_map)
        f.close()


    def test_ignore_gt_label(self):
        classes_gt = {0: 'Not Labeled', 1: 'Vegetation', 2: 'Building'}
        cm = np.asarray([[  30, 44000, 17000],
                         [  10,   210,    30],
                         [  20,    50,   200]])

        f = open(self.dir + 'test_ignore_gt_label.txt', 'w')
        conf_matrix.print_to_file(cm, f, classes_gt, ignore_labels=[0])
        f.close()



if __name__ == '__main__':
    unittest.main()
