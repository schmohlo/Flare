""" by Stefan Schmohl, 2020 """

import numpy as np
import time


class Timer:

    def __init__(self):
        self._start_time = 0
        self._stop_time = 0
        self._delay = 0
        self._state = 'stopped'  # 'started', 'stopped'
        self.start()


    @staticmethod
    def s2smhd(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return {'s': s, 'm': m, 'h': h, 'd': d}


    @staticmethod
    def string(seconds=None, precision=2):
        smhd = Timer.s2smhd(seconds)
        fstring = '{:.0f}d {:.0f}h {:.0f}m {:.' + str(int(precision)) + 'f}s'
        return fstring.format(smhd['d'], smhd['h'], smhd['m'], smhd['s'])


    def time(self):
        if self._state == 'stopped':
            ref_time = self._stop_time
        if self._state == 'started':
            ref_time = time.time()
        return ref_time - self._start_time - self._delay


    def time_smhd(self):
        return self.s2smhd(self.time())
        
        
    def time_string(self, precision=2):
        return self.string(self.time(), precision)


    def start(self):
        assert self._state == 'stopped'
        self._start_time = time.time()
        self._stop_time = 0
        self._delay = 0
        self._state = 'started'
        return time.time()
    
     
    def stop(self):
        assert self._state == 'started'
        self._stop_time = time.time()
        self._state = 'stopped'
        return self.time()
        
        
    def restart(self):
        t = self.time()
        self.stop()
        self.start()
        return t


    def resume(self):
        assert self._state == 'stopped'
        self._delay += time.time() - self._stop_time
        self._state = 'started'
        return self.time()
