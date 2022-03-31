""" 
Wrapper for laspy package for reading and writing LAS files.


Why?
----
Especially designed for reading to / writing from float32 memory, which is the 
typicall format for deep learning. 

This is becaue floats are faster on GPUs than doubles. Also labels need to be 
floating point in DL frameworks anyways.

Also voxelation can lead to floating numbers for fields which are usually 
integers, for example number of returns.

And lastly CloudCompare also uses float internaly, so this us probably a good
idea.


How?
----
- Data type controlled by IDT ('internal data type', global variable).
	- IDT = 'float32': laspy data type  9, Float,  4 bytes
	- IDT = 'float64': laspy data type 10, Double, 8 bytes

- Field names are all made lowercase because laspy reads them lowercase.
  Edit 20.04.2020:
    - Since laspy v1.7.0, laspy appears to leave upper-case field names as is 
      on import. However, it can't access upper-case extra-dims!
      => stick to 1.6.0

- Dimensions = scalar fields = point features.

- To save intermediate results in LAS files for internal usage, save all fields
  as floats with 
	- read_from_float()
	- write_to_float()


Issues:
-------
- can not guarantee that classification byte flags are going to work when 
  writing to file!  (they probably won't....)


by Stefan Schmohl, 2020
"""


import os
import warnings
import math
import re
import numpy as np
import laspy




def read(file_path, fields=None, reduced=False, decomp_class_byte=False, 
         ignore_empty=True, IDT='float64'):
    """
    Reads data from LAS file.

    Returns a numpy array of type defined by IDT.
    
    Returned coordinates are global (=after applying shift and offset to the LAS
    internal integer coordinates). 'x','y','z' will always be the first three
    data fields. The internal data type 'IDT' is intended to be float32.
    
    Args:
        file_path (string):
            Path to LAS file.
        fields ([strings]):
            List of data fields to extract from LAS file. Coordinates 'x','y',
            'z' will always be in the first three fields. None to read all 
            available fields, depeding on the LAS point format. [] only coords.
        reduced:
            If True, loads the internal las cooords only multiplied by their 
            scale. Does not restore the offset. For maximum possible precision 
            given IDT.
            If False, adds offset from las file to the coords and thereby 
            restores their original values. 
            Warning: when using IDT=float32, False may leed to precision loss,
            cause float32 only garantues 6-9 correct decimal digits!
        decomp_class_byte:
            If False, use the whole classification byte as class label. Only 
            effects LAS format <=5. False allows for class labels <=255, 
            otherwise only <=31. See also LAS or laspy doc. The cost is not
            having the fields 'synthetic', 'key_point' or 'withheld' available.
        ignore_empty: 
            IF True, ignore fields which have all values = 0.
        IDT:
            Internal data type. 'float32' or 'float64'.

    Returns:
        data:
            Numpy array holding all points and their scalar fields values. 
            Shape: (#points, #fields); dtype: IDT
        fields:
            List of strings naming the data columns. 'x','y','z' will always be
            the first three fields. The order of the other fields depends only
            on laspy.
        precision:
            Precision ('scale') in which coords had been stored in the LAS file.
        offset:
            Offset with which coords had been stored in the LAS file.
    """
    
    in_file = laspy.file.File(file_path, mode='r')
    pointset = in_file.points["point"]
    
    ## Determine readable fields:
    # Lowercase the field names, because laspy imports lowercase anyway and
    # later we want "x" and not "X" (org. coords vs. LAS integers):
    dims = [name.lower() for name in pointset.dtype.names]
    if fields is not None:
       fields = [field.lower() for field in fields]
   
    ## Split sub-byte fields to its sub-fields, depending on LAS point format:
    if in_file.header.data_format_id <= 5:
        if 'raw_classification' in dims and decomp_class_byte:
            dims += ['classification',
                     'synthetic',
                     'key_point',
                     'withheld']
            dims.remove('raw_classification')
        if 'flag_byte' in dims:
            dims += ['return_num',
                     'num_returns',
                     'scan_dir_flag', 
                     'edge_flight_line']
            dims.remove('flag_byte')
    else:
        if 'classification_flags' in dims:
            dims += ['synthetic',
                     'key_point',
                     'withheld',
                     'overlap', 
                     'scanner_channel',
                     'scan_dir_flag',
                     'edge_flight_line']
            dims.remove('classification_flags')
        if 'flag_byte' in dims:
            dims += ['return_num',
                     'num_returns']
            dims.remove('flag_byte')
    
    ## Remove empty or non-requested fields:
    for dim in dims.copy():  # copy, because removes are updated live.
        if ignore_empty:
            mini = np.min(getattr(in_file, dim))
            maxi = np.max(getattr(in_file, dim))
            if mini == 0 and maxi == 0:
                dims.remove(dim)
                continue
        if fields is not None:
            if dim not in fields and \
               (dim != 'raw_classification' or 'classification' not in fields): 
                dims.remove(dim)
                
    # Make sure x,y,z, are the first three fields (for convenience and to undo 
    # a previous removal which happened in case of one axis beeing 0):
    dims = _remove_from_list(dims, ['x', 'y', 'z'])


    ## Extraction of data:
    data = np.zeros((len(pointset), len(dims)+3), dtype=IDT)
    if reduced:
        data[:,0] = in_file.X * in_file.header.scale[0]
        data[:,1] = in_file.Y * in_file.header.scale[1]
        data[:,2] = in_file.Z * in_file.header.scale[2]
    else:
        data[:,0] = in_file.x
        data[:,1] = in_file.y
        data[:,2] = in_file.z
      
    for i, dim in enumerate(dims):
        data[:,i+3] = getattr(in_file,  dim)

    dims = ['x', 'y', 'z'] + dims

    # Rename raw_classification, which has been kept to get the full class.byte:
    if 'raw_classification' in dims:
        dims = ['classification' if dim == 'raw_classification' else dim for dim in dims]

    precision = in_file.header.scale
    offset = in_file.header.offset
    in_file.close()
    
    return data, dims, precision, offset




def read_from_float(file_path, fields=None, reduced=False,
                    ignore_empty=True, dtype='float32'):
    """             
    Reads data from a LAS file where the fields are named "..._float32/64".
    
    If field names are given, it will only read those specficially named fields
    to prevent confusion with fields of the same name but without '_floatXX'. 
    For more details, see read().
    
    Args:
        fields ([strings]):  
            Same as for read(), WITHOUT the float suffix!
    """
    
    suffix = '_' + dtype

    if fields is not None:
        fields_ = [f + suffix if f not in ['x', 'y', 'z'] else f for f in fields]
    else:
        fields_ = fields
    
    data, dims, precision, offset = read(file_path, fields_,
                                         reduced,
                                         decomp_class_byte=False, 
                                         ignore_empty=ignore_empty,
                                         IDT=dtype)
       
    # Delete all fields without the suffix (happens if fields=None, e.g. 
    # return_num & num_returns with defualt values = 1, or other custom fields).
    for f in dims.copy():
        if not re.search(suffix, f) and f not in ['x', 'y', 'z']:
            data = np.delete(data, dims.index(f), 1)
            dims.remove(f)
    
    # Remove suffix from output field names:
    dims = [re.sub(suffix, '', dim) for dim in dims]

    return data, dims, precision, offset




def write(file_path, data, fields, precision=0.01, offset=None,
          reduced=False, decomp_class_byte=False, file_version=1.2, 
          point_format=2, IDT='float32'):
    """
    Writes a pointcloud data array into a LAS file.
    
    ATTENTION: Many fields in LAS are predefined with a limited number of bits!
    It is therefore possible to lose information, for example for classification
    or number of returns. Use extra fields by choosing other names or use 
    write_to_float() instead.
    
    Extra fields are in IDT or float64.
    
    Args:
        data:
            Numpy array holding all points and their scalar fields values. 
            Shape: (#points, #fields); dtype: IDT
        fields:
            List of strings naming the data columns. Needs 'x','y','z' in any
            case. Will be made lowercase, because laspy supports mostly 
            lowercase and to prevent confusion between laspy's 'X' and 'x'.
        precision:
            Precision in which point coordinates are stores in LAS file 
            ("scale"). Input single scalar or list of size 3. Default is 0.01,
            which usually equals 1 centimeter (if input is in meters).
        offset:
            Offset with which point coordinates shell be stored in the LAS file.
            Input list of size 3. Default is None, which calculates the center 
            of gravity as offset.
        reduced:
            If True, this means that the offset has already been substracted
            from the coordinates. Coords will only be divided by the precision 
            before saving.
            If False, offset will be subtracted from coords before saving.
        decomp_class_byte:
            If True (and point_format <=5), the classification byte will compose
            the classification values as well as the classification flags. The 
            classifiation value will be converted to byte and occupy only 5 bits
            in the classificaion byte. Maximum class label is thereby 31.
            If False (or point_format > 5), the whole classification byte will
            contain only the input classification values. This allows for class
            labels up to 255 (still converted to byte, though).
            WARNING: the fields 'synthetic', 'key_point', and 'withheld' will be
            ignored if False and point_format <= 5!
        IDT:
            Internal data type. 'float32' or 'float64'.
    """

    predef_subbyte_fields = ['classification', 'return_num', 'num_returns',
                             'scan_dir_flag']  
    
    assert not (file_version < 1.4 and point_format > 5), \
        "Point format 6 and higher only supported in LAS v1.4"

    fields = [field.lower() for field in fields]
 
    out_header = laspy.header.Header(file_version, point_format)
    out_file   = laspy.file.File(file_path, mode='w', header=out_header)
    hM         = laspy.header.HeaderManager(out_header, out_file.reader)


    ## Extract coords from data and fields to handle them seperatly:
    coord_cols = [fields.index(f) for f in ['x','y','z']]
    coords = data[:,coord_cols]
    data = np.delete(data, coord_cols, 1)
    fields = _remove_from_list(fields, ['x', 'y', 'z'])
    
    
    ## Setting up the dimensions must be done before writing any data.
    # But not x,y,z. Handle them extra. Because laspy....
    # For data tpes see https://laspy.readthedocs.io/en/latest/tut_part_3.html .
    for f in fields:
        if f not in [x.name for x in out_file._reader.point_format.specs]:
            if f in predef_subbyte_fields:
				# Creating custom 'classification' field would unlink it from
                # raw_classification. Same for some other predefined fields.
                # This happens only in subbyte fields.
                continue  
            if IDT == 'float32':
                dt = 9
            elif IDT == 'float64':
                dt = 10  
            else:
                dt = 10
            if len(f) > 32:
                raise ValueError("Output .las dimension description larger " +
                                 "than 32 characters.")
            out_file.define_new_dimension(f, data_type=dt, description=f)
            # out_file.addProperty(c)  # do i realy need this?

    
    ## Coords
    # must be writen first, cause laspy will check field lengths acording
    # to the number of point later based on this.
    if offset is None:
        offset = np.mean(coords, axis=0)
    hM.set_offset(offset)
    
    if isinstance(precision, (int, float)):
        hM.scale = [precision]*3
    elif type(precision) is list and len(precision) == 3:
        hM.scale = precision
    else:
        hM.scale = [0.001]*3
        print("Warning: No walid las precision. Saving file with scale=0.001.")

    if reduced:
        out_file.X = coords[:, 0] / hM.scale[0]
        out_file.Y = coords[:, 1] / hM.scale[1]
        out_file.Z = coords[:, 2] / hM.scale[2]
    else:
        out_file.set_x_scaled(coords[:, 0])  # also does offset!
        out_file.set_y_scaled(coords[:, 1])
        out_file.set_z_scaled(coords[:, 2])
    
    
    ## Classification field:
    if point_format <= 5 and 'classification' in fields:
        class_values = data[:,fields.index('classification')]
        if np.max(class_values) > 255:
            warnings.warn("Classification values over 255. Labels > 255 will be set to 255. Consider using other field name.")
            class_values[class_values > 255] = 255 
        class_byte = class_values.astype('uint8')
        
        if decomp_class_byte:
            ## Prevent overflow:
            if np.max(class_byte) > 31:
                warnings.warn("Classification values over 31. Lables > 31 will be set to 31. Consider using decomp_class_byte=False or using other field name.")
                class_byte[class_byte > 31] = 31 
        else:
            ## Remove class flags to prevent overriding the class_byte later on:
            cflag_names = ['synthetic', 'key_point', 'withheld']
            # prevent "not in list" errors:
            cflags = [f for f in cflag_names if f in fields] 
            cflag_cols = [fields.index(f) for f in cflags]                          
            data = np.delete(data, cflag_cols, 1)
            fields = _remove_from_list(fields, cflags)

        out_file.raw_classification = class_byte
        
        # delete classification from data and fields, because already written:
        data = np.delete(data, fields.index('classification'), 1)
        fields = _remove_from_list(fields, 'classification')
	
    for f in fields:
        field_data = data[:,fields.index(f)]
        setattr(out_file, f, field_data.astype(getattr(out_file, f).dtype))

    out_file.close()




def write_to_float(file_path, data, fields, precision=0.01, offset=None,
                   reduced=False, IDT='float32'):
    """
    Writes a pointcloud data array into a LAS file with custom float fields.
    
    Fields are named '.._float32' or '..._float64' depending on IDT.
    This is less efficient however, because depending on the point format each
    point in a LAS file has at least some certain fields. For more details, see 
    write().
    """
    
    # Just by renaming the fields to non-standard field names, laspy will save
    # them in new, not standard dimensions:
    fields = [field.lower() for field in fields]
    fields = [f + '_' + IDT if f not in ['x', 'y', 'z'] else f for f in fields]

    # point_format = 0 to save memory, because it's the format with the least 
    # number of fields.
    # decomp_class_byte = False because classification is stored in float anyway.
    write(file_path, data, fields, precision, offset, reduced,
          decomp_class_byte=False, file_version=1.2, point_format=0)





def _remove_from_list(list_, values):
    """ Remove one or more values from list.
    
    More robust then list.remove() for some cases, because it does not throw an
    error when value is not found. If you know that the (single) value is in the 
    list, list.remove(value) might be more elegant.
    """
    
    if type(values) is not list:
        values = [values]
    list_ = [v for v in list_ if v not in values]
    return list_

