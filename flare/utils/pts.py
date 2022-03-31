""" Writing and reading .pts ASCII files

by Stefan Schmohl, 2020 
"""

import os
import math
import numpy as np
import pandas as pd



def write(file_path, data, fields, precision=0.01, header=True, offset=None):
    """ 
    Write point cloud to PTS ASCII file.
    
    Args:
        file_path (string):
            Path to PTS file.
        data (ndarray):
            NxM numpy array of N points with M fields.
        fields ([strings]):
            List of data fields to extract from PTS file. X,y,z allways have
            to be included.
        precision:
            Floating point precision for writing.
        header:
            If True, write headerline into file.
        offset:
            Offset inherint in the coordinates of date. Will be added back onto
            the coordinates before writing to file.
    """

    # Add extension if not allready in file path:
    if file_path[-4:] != '.pts':
        file_path += '.pts'

    # Sort data and cols so that xyz are in the beginning (for convenience)
    # and by the way add offset if given.
    coord_cols = [fields.index(c) for c in ['x','y','z']]
    data = data.astype('float64')  # make sure data can handle coords+offset
    coords = data[:, coord_cols]
    if offset is not None:
        coords += offset
    data = np.hstack((coords, np.delete(data, coord_cols, 1)))
    fields = ['x', 'y', 'z'] + [c for c in fields if c not in ['x', 'y', 'z']]

    if header:
        h = ' '.join(fields)
    else:
        h = None
        
    p = math.ceil(-math.log(precision, 10))
    
    np.savetxt(file_path, data, fmt='%.{}f'.format(p),
               header=h,
               comments='//')



def read(file_path, fields=None, col_names=None, header='infer', sep='\s+', 
         comment=None, reduced=False, IDT='float64'):
    """
    Read point cloud from PTS ASCII file.
    
    Args:
        file_path (string):
            Path to PTS file.
        fields ([strings]):
            List of data fields to extract from PTS file. X,y,z are allways 
            read. None to read all available fields. [] for only coords.
        col_names:
            List of column names to use. If the file contains a header row, then
            you should explicitly pass header=0 to override the column names. 
            Duplicates in this list are not allowed.
        header:
            Row number(s) to use as the column names, and the start of the data. 
            Default behavior is to infer the column names: if no names are 
            passed the behavior is identical to header=0 and column names are 
            inferred from the first line of the file, if column names are passed 
            explicitly then the behavior is identical to header=None. Explicitly 
            pass header=0 to be able to replace existing names. Note that this 
            parameter ignores commented lines and empty lines if 
            skip_blank_lines=True, so header=0 denotes the first line of data 
            rather than the first line of the file.
            Removes '//' from header line.
        sep:
            Seperator of columns. Default is all whitespaces.
        reduced:
            If True, returns the cooords after substracting an offset (center 
            of gravity) for maximum possible precision given IDT.
            If False, returns original coords in IDT precision.
            Warning: when using IDT=float32, False may leed to precision loss,
            because float32 only garantues 6-9 correct decimal digits!
        IDT:
            Internal data type. 'float32' or 'float64'.

    Returns:
        data:
            Numpy array holding all points and their scalar fields values. 
            Shape: (#points, #fields); dtype: IDT
        fields:
            List of strings naming the data columns. 'x','y','z' will always be
            the first three fields. The order of the other fields depends on the
            input file.
        offset:
            Only if reduced=True.
    """

    # Extract the stuff from file (pandas is about 10x faster than numpy):
    data_frame = pd.read_csv(file_path, sep=sep, dtype='float64', header=header, 
                             names=col_names, comment=comment)
    
    # Prepare data and field names = column names, including the removal of
    # possible pts header line indicator ('//') from first field name:
    data = data_frame.values
    cols = list(data_frame.columns)
    cols = [c.lower() for c in cols]
    cols[0] = cols[0].replace('/', '')

    # Sort data and cols so that xyz are in the beginning (for convenience):
    coord_cols = [cols.index(c) for c in ['x','y','z']]
    coords = data[:, coord_cols]
    if reduced:
        offset = np.mean(coords, axis=0)
        coords -= offset
    data = np.hstack((coords, np.delete(data, coord_cols, 1)))
    cols = ['x', 'y', 'z'] + [c for c in cols if c not in ['x', 'y', 'z']]
    
    # Remove columns that are not in fields:
    if fields is not None:
        fields = [f.lower() for f in fields]
        fields = fields + ['x', 'y', 'z'] # just to be sure they are allways in.
        for f in cols.copy():
            if f not in fields:
                data = np.delete(data, cols.index(f), 1)
                cols.remove(f)
    
    if reduced:
        return data.astype(IDT), cols, offset
    else:
        return data.astype(IDT), cols
