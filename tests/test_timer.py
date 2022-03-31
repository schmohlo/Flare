import unittest
import numpy as np
import numpy.testing as nptest

from flare.utils.timer import Timer



class TestTimer(unittest.TestCase):


    def test_calls(self):
        # Quick and dirty trying mostly all combos.

        t = Timer()

        with self.assertRaises(AssertionError): start = t.start()
        with self.assertRaises(AssertionError): t.resume()

        t.stop()
        with self.assertRaises(AssertionError): t.stop()
        t.resume()
        with self.assertRaises(AssertionError): t.start()
        with self.assertRaises(AssertionError): t.resume()

        t.restart()
        with self.assertRaises(AssertionError): t.start()
        with self.assertRaises(AssertionError): t.resume()


    def test_s2smhd(self):

        t = Timer()

        s_ = 12
        m_ = 4
        h_ = 5
        d_ = 9

        s = s_ + m_*60 + h_*60*60 + d_*60*60*24

        smhd = Timer.s2smhd(s)

        self.assertEqual(smhd['s'], s_)
        self.assertEqual(smhd['m'], m_)
        self.assertEqual(smhd['h'], h_)
        self.assertEqual(smhd['d'], d_)


if __name__ == '__main__':
    unittest.main()
