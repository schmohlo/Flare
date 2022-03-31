"""

by Stefan Schmohl, 2020
"""

import numpy as np
from typing import List
from . import las
from . import pts



class PointCloud:


    def __init__(self, data, fields):
        
        assert data.shape[1] == len(fields)
        
        self.data = data
        self.fields = fields.copy()
        
        # TODO:
        self.attr = []      
        self.import_offset = None      # in einem der oberen zusammenfassen?
        self.import_precision = None   # in einem der oberen zusammenfassen?
        self.import_reduced = False
        #tmat?
        
    
    def __len__(self):
        return self.data.shape[0]
    
    @property
    def shape(self):
        return self.data.shape;
        
    @property
    def extent(self):
        mi = np.min(self.get_coords, 0)
        ma = np.max(self.get_coords, 0)
        return np.abs(ma - mi)
        
        
    def field_indices(self, field_names: List[str]):
        if type(field_names) != list:
            raise TypeError('field_names is not a list')
        return [self.fields.index(f) for f in field_names]
    
    
    def get_fdata(self, field_names: List[str]):
        if not isinstance(field_names, list):
            field_names = [field_names]
        cols = self.field_indices(field_names)
        return self.data[:, cols]
    
    
    def set_fdata(self, fdata, field_names: List[str]):
        cols = self.field_indices(field_names)
        self.data[:, cols] = fdata
    
        
    def get_xyz(self):
        return self.get_fdata(['x','y','z'])
    
    
    def set_xyz(self, coords):
        self.set_fdata(coords, ['x','y','z'])
        
    
    def delete_fields(self, field_names: List[str]):
        cols = self.field_indices(field_names)
        self.data = np.delete(self.data, cols, 1)
        for f in field_names:
            self.fields.remove(f)
   
   
    def add_fields(self, fdata, field_names: List[str]):
        assert type(field_names) == list
        for f in field_names:
            if f in self.fields:
                raise ValueError('Field {} already exists!'.format(f))
        self.data = np.append(self.data, fdata, axis=1)
        self.fields += field_names
    
    
    def get_entries(self, indices):
        d = self.data[indices]
        return d


