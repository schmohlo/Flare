import unittest
import numpy as np
import numpy.testing as nptest

import flare.nn.data_loading as stuff


NaN = float('nan')


class TestLabelMap(unittest.TestCase):

    data_classes  = {8: 'Powerline', 1: 'Low veg.',    2: 'Imp. surface',
                     0: 'PkW',       4: 'Fence/Hedge', 5: 'Roof',
                     6: 'Facade',   11: 'Shrub',       9: 'Truck'}

    model_classes = {0: 'Powerline', 1: 'Low veg.',    2: 'Imp. surface',
                     3: 'Car',       4: 'High veg.',   5: 'Building'}

    class_mapping = {'Powerline'   : 'Powerline',
                     'Low veg.'    : 'Low veg.',
                     'Imp. surface': 'Imp. surface',
                     'PkW'         : 'Car',
                     'Truck'       : 'Car',
                     'Shrub'       : 'High veg.',
                     'Roof'        : 'Building',
                     'Facade'      : 'Building',
                     'Fence/Hedge' : 'High veg.'}

    def test_map_and_inv(self):
        map, inv = stuff.create_label_map(self.data_classes, self.model_classes,
                                          self.class_mapping)

        map_ = [3, 1, 2, NaN, 4, 5, 5, NaN, 0, 3, NaN, 4]
        inv_ = [8, 1, 2, 0, 11, 5]

        nptest.assert_array_equal(map, map_)
        nptest.assert_array_equal(inv, inv_)


    def test_maped_labels(self):
        map, inv = stuff.create_label_map(self.data_classes, self.model_classes,
                                          self.class_mapping)

        labels_data = np.asarray(list(self.data_classes.keys()))
        labels_mapped  = [0, 1, 2, 3, 4, 5, 5, 4, 3]
        labels_revesed = [8, 1, 2, 0, 11, 5, 5, 11, 0]

        nptest.assert_array_equal(map[labels_data], labels_mapped)
        nptest.assert_array_equal(inv[map[labels_data].astype(int)], labels_revesed)



if __name__ == '__main__':
    unittest.main()
