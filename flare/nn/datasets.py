""" Dataset classes, derived from torch.Dataset

Datasets are the internal data storage for Dataloaders, which in turn are
responsible for feeding a network sample by sample / mini-batch by mini-batch.

A single sample is drawn by __getitem__().
A mini-batch is created by calling collate() on a list of samples.

Those output functions are tailord for the SCN framework as well as my own
model subclasses.

by Stefan Schmohl, 2020
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from torch.utils.data import Dataset
from . import _utils as nnutils
from flare.utils import Timer
from flare.utils import import_class_from_string
from matplotlib.patches import Circle as mplcircle
from matplotlib.patheffects import withStroke



def dataset_from_config(config, samples, label_map, set, *args):
    try:
        dataset = import_class_from_string(config['train']['dataset'])
    except KeyError:
        # for backwards-compatibility:
        dataset = VoxelCloudDataset
    dataset = dataset.from_config(config, samples, label_map, set, *args)
    return dataset




class VoxelCloudDataset(Dataset):

    @classmethod
    def from_config(cls, config, samples, label_map, set, *args):
        """ """
        dataset_config = config['train'].get('dataset_'+set, None)
        if dataset_config is None:
            threshold = 1
            augmentation = None
        else:
            threshold = dataset_config['threshold']
            try:
                augmentation = dataset_config['augmentation']
            except KeyError:
                augmentation = None
        dataset = cls(samples,
                      label_map,
                      config['model']['input_features'],
                      config['data_'+set]['gt_field'],
                      config['train']['datasets']['transforms'],
                      threshold,
                      augmentation)
        return dataset


    def __init__(self, samples, label_map, input_features, gt_field,
                 transforms=None, threshold=1, augmentation=None):
        """ Dataset for pytorch Dataloader for SparseConvNet

        When dataloading, returns in 'data' key coords together with the
        features in one tensor. When multiple samples are merged to a mini-
        batch, a 4th coord for the sample index inside that mini-batch is added.

        See scn.InputLayer from github.com/facebookresearch/SparseConvNet

        Attention: internal samples share the same memory with the input
                   samples and will be altered by the Dataset!

        Args:
            samples (list):
                List of samples, each sample (dict) having as items a
                'voxel_cloud' with a point cloud object inside (and better also
                a 'name' item)
            label_map (ndarray):
                Used for mapping the original ground truth labels to the
                specificaions of the network.
            input_features (list[str]):
                list of model input features by name in the correct order!
            gt_field (str):
                Scalar field of the points cloud to use as ground truth.
            transforms (list[str]):
                List of transformations to apply to each sample voxel cloud:
                    - 'reduce_coords':
                        => min(xyz) = [0, 0, 0]
                    - 'reduce_z':
                        => min(z) = 0
                    - 'norm_features':
                        => per feature and sample € [0,..,1]
                    - 'set_features_to_1':
                        => usefull for training only on geometry.
        """

        print('\nInit Dataset ...')

        del_counter=0
        for i in reversed(range(len(samples))):
            num_of_voxels = len(samples[i]['voxel_cloud'])
            if num_of_voxels < threshold:
                del samples[i]
                del_counter += 1
        print('  Removed {:d} samples < {:d} elements.'.format(del_counter, threshold))

        nf = input_features + [gt_field] + ['x', 'y', 'z']   # "numb of fields"

        for sample in samples:
            vc = sample['voxel_cloud']

            for f in nf:
                if f not in vc.fields:
                    e_str = 'field \'{}\' does not exist in sample \'{}\'.\n'
                    e_str += 'Sample has fields: ' + str(vc.fields)
                    raise Exception(e_str.format(f, sample['name']))

            if label_map is not None:
                ind = vc.field_indices([gt_field])
                vc.data[:, ind] = label_map[vc.data[:, ind].astype(int)]
                if np.isnan(np.min(vc.data[:, ind])):
                    print("  Warning: unable map some data labels to model labels! => nan")

            if transforms is not None:
                if 'reduce_coords' in transforms:
                    coords = vc.get_xyz()
                    min_coords = np.min(coords, 0)
                    vc.set_xyz(coords - min_coords)
                    # Do the same for tree coords if present:
                    if 'tree_set' in sample.keys():
                        tr = sample['tree_set']
                        tr.set_xyz(tr.get_xyz() - min_coords)

                if 'reduce_z' in transforms:
                    ind_z = vc.field_indices(['z'])
                    min_z = np.min(vc.data[:, ind_z])
                    vc.data[:, ind_z] -= min_z
                    # Do the same for tree coords if present:
                    if 'tree_set' in sample.keys():
                        tr = sample['tree_set']
                        ind_z = tr.field_indices(['z'])
                        tr.data[:, ind_z] -= min_z

                if 'set_features_to_one' in transforms:
                    shp = vc.get_fdata(input_features).shape
                    vc.set_fdata(np.ones(shp), input_features)

                if 'norm_features' in transforms:
                    data = vc.get_fdata(input_features)
                    mins = np.min(data, 0)
                    maxs = np.max(data, 0)
                    diff = maxs - mins
                    diff[diff==0] = 1  # prevent diff by zero.
                    vc.set_fdata((data - mins) / diff, input_features)

        self.feature_fields = input_features
        self.gt_field = gt_field
        self.samples = samples
        self.augmentation = augmentation



    def __len__(self):
        return len(self.samples)



    def __getitem__(self, index):
        vc = self.samples[index]['voxel_cloud']

        coords = vc.get_xyz()
        labels = vc.get_fdata([self.gt_field])
        features = vc.get_fdata(self.feature_fields)
        
        # Augmentation (coords must stay positive!!)
        if self.augmentation is not None:
            if 'scaling' in self.augmentation.keys():
                s = random.uniform(*self.augmentation['scaling'])
                coords *= s
            if 'translation' in self.augmentation.keys():
                t = np.random.randint(*self.augmentation['translation'], (1,2))
                coords[:,0:2] += t
            if 'flip' in self.augmentation and self.augmentation['flip']:
                # (coords should already normed 0..127; must stay >0!)
                if random.randint(0,1):
                    m = np.max(coords[:,0])
                    coords[:,0]  = -1 * coords[:,0]  + m
                if random.randint(0,1):
                    m = np.max(coords[:,1])
                    coords[:,1]  = -1 * coords[:,1]  + m
                # print(np.min(coords,0))
                # print(np.max(coords,0))
                # print('----')

        # No need to convert coords to long, done in scn.InputLayer.
        return {'data':  [torch.from_numpy(coords),
                          torch.from_numpy(features)],
                'labels': torch.from_numpy(labels.astype(np.int64))}



    @staticmethod
    def collate(batch):
        """ Creates a mini-batch ready to use in scn.InputLayer  """
        coords = []
        featrs = []
        labels = []

        if not isinstance(batch, list):  # in case mini-batch-size = 1
            batch = [batch]

        batch_size = len(batch)

        for sample in batch:
            coords.append(sample['data'][0])
            featrs.append(sample['data'][1])
            labels.append(sample['labels'])

        # Add batch-relative sample index as additional coord (see scn.InputLayer):
        coords = [torch.cat([x, i * torch.ones((x.shape[0], 1))], dim=1)
                  for i, x in enumerate(coords)]

        return {'data':  [torch.cat(coords, dim=0),
                          torch.cat(featrs, dim=0),
                          batch_size],
                'labels': torch.squeeze(torch.cat(labels, dim=0))}






#=#############################################################################
## 3D -> 2D                                                                   #
#                                                                             #
#=#############################################################################


class VoxelCloudDataset_3D2D(VoxelCloudDataset):
    """
    Altough ground truth is a 2D grid ("image"), it is flattened to 1D to
    allow usage of standard classifier. "shape" needed to restore original 2D.
    """

    @classmethod
    def from_config(cls, config, samples, label_map, set, *args):
        dataset_config = config['train'].get('dataset_'+set, None)
        if dataset_config is None:
            threshold = 1
        else:
            threshold = dataset_config['threshold']
        dataset = cls(config['model']['network']['net-params']['input_size'],
                      samples,
                      label_map,
                      config['model']['input_features'],
                      config['data_'+set]['gt_field'],
                      config['train']['datasets']['transforms'],
                      threshold,
                      config['model']['ignore_label'])
        return dataset



    def __init__(self, input_size, samples, label_map, input_features, gt_field,
                 transforms=None, threshold=1, default_label=-100):

        VoxelCloudDataset.__init__(self, samples, label_map, input_features,
                                   gt_field, transforms, threshold)

        self.default_label=default_label
        self.shape2D = input_size[0:2]

        ## Make 2D Ground-Truth:
        print("  Making labels 2D ...")
        timer = Timer()

        for sample in samples:
            vc = sample['voxel_cloud']

            coords = vc.get_xyz()
            features = vc.get_fdata(self.feature_fields)

            labels = vc.get_fdata([self.gt_field]).astype(np.int64)

            labels_2D, height = nnutils.project_z_numpy(coords, labels, 'max',
                                                        self.default_label, self.shape2D)

            labels_2D = np.squeeze(labels_2D)  # remove channel-dimension
            sample['labels_2D'] = labels_2D
            sample['height'] = height

        print("    Finished in", timer.time_string())



    def __getitem__(self, index):
        vc = self.samples[index]['voxel_cloud']

        coords = vc.get_xyz()
        features = vc.get_fdata(self.feature_fields)

        labels = self.samples[index]['labels_2D']
        shape  = tuple(labels.shape)
        labels = torch.flatten(torch.from_numpy(labels))

        # No need to convert coords to long, done in scn.InputLayer.
        return {'data':  [torch.from_numpy(coords),
                          torch.from_numpy(features)],
                'labels': labels,
                'shape':  shape}



    @staticmethod
    def collate(batch):
        """ Creates a mini-batch ready to use in scn.InputLayer  """
        coords = []
        featrs = []
        labels = []
        shapes = []

        if not isinstance(batch, list):  # in case mini-batch-size = 1
            batch = [batch]

        batch_size = len(batch)

        for sample in batch:
            coords.append(sample['data'][0])
            featrs.append(sample['data'][1])
            labels.append(sample['labels'])
            shapes.append(sample['shape'])

        # Add batch-relative sample index as additional coord (see scn.InputLayer):
        coords = [torch.cat([x, i * torch.ones((x.shape[0], 1))], dim=1)
                  for i, x in enumerate(coords)]

        return {'data':  [torch.cat(coords, dim=0),
                          torch.cat(featrs, dim=0),
                          batch_size],
                'labels': torch.squeeze(torch.cat(labels, dim=0)),
                'shapes': shapes}




class VoxelCloudDataset_2D(VoxelCloudDataset):
    """
    For true 2D Networks. Creates 2D feature maps at __init__.
    
    Altough ground truth is a 2D grid ("image"), it is flattened to 1D to
    allow usage of standard classifier. "shape" needed to restore original 2D.
    """

    @classmethod
    def from_config(cls, config, samples, label_map, set, *args):
        dataset_config = config['train'].get('dataset_'+set, None)
        if dataset_config is None:
            threshold = 1
        else:
            threshold = dataset_config['threshold']
        dataset = cls(config['model']['network']['net-params']['input_size'],
                      samples,
                      label_map,
                      config['model']['input_features'],
                      config['data_'+set]['gt_field'],
                      config['train']['datasets']['transforms'],
                      threshold,
                      config['model']['ignore_label'])
        return dataset



    def __init__(self, input_size, samples, label_map, input_features, gt_field,
                 transforms=None, threshold=1, default_label=-100):

        VoxelCloudDataset.__init__(self, samples, label_map, input_features,
                                   gt_field, transforms, threshold)

        self.default_label=default_label
        self.shape2D = [int(i) for i in input_size[0:2]]

        ## Make 2D Ground-Truth:
        print("  Making labels & features 2D ...")
        timer = Timer()

        for sample in samples:
            vc = sample['voxel_cloud']

            coords   = vc.get_xyz()
            features = vc.get_fdata(self.feature_fields)
            labels   = vc.get_fdata([self.gt_field]).astype(np.int64)

            labels_2D, height = nnutils.project_z_numpy(coords, labels, 'max',
                                                        self.default_label, self.shape2D)

            features_2D, _ = nnutils.project_z_numpy(coords, features, 'max',
                                                     0, self.shape2D)

            labels_2D = np.squeeze(labels_2D)  # remove channel-dimension
            features_2D = np.moveaxis(features_2D, 2, 0)  #wxhxc => cxwxh
            sample['labels_2D']   = labels_2D
            sample['features_2D'] = features_2D
            sample['height']      = height

        print("    Finished in", timer.time_string())



    def __getitem__(self, index):
        vc = self.samples[index]['voxel_cloud']

        coords   = vc.get_xyz()
        features = self.samples[index]['features_2D']
        labels   = self.samples[index]['labels_2D']
        shape    = tuple(labels.shape)
        #labels   = torch.flatten(torch.from_numpy(labels))

        return {'data':  [torch.from_numpy(coords),
                          torch.from_numpy(features)],
                'labels': torch.from_numpy(labels),
                'shape':  shape}



    @staticmethod
    def collate(batch):
        """ Creates a mini-batch ready to use in scn.InputLayer  """
        coords = []
        featrs = []
        labels = []
        shapes = []

        if not isinstance(batch, list):  # in case mini-batch-size = 1
            batch = [batch]

        batch_size = len(batch)

        for sample in batch:
            coords.append(sample['data'][0])
            featrs.append(sample['data'][1])
            labels.append(sample['labels'])
            shapes.append(sample['shape'])

        coords = [torch.cat([x, i * torch.ones((x.shape[0], 1))], dim=1)
                  for i, x in enumerate(coords)]

        return {'data':  [torch.cat(coords, dim=0),
                          torch.stack(featrs)],
                'labels': torch.stack(labels),
                #'labels': torch.squeeze(torch.cat(labels, dim=0)),
                'shapes': shapes}





#=#############################################################################
## Tree-Detection                                                             #
#                                                                             #
#=#############################################################################






class VoxelCloudDataset_Tree_2D(VoxelCloudDataset_2D):
    """
    TODO
    """


    @classmethod
    def from_config(cls, config, samples, label_map, set, voxel_size):
        dataset_config = config['train'].get('dataset_'+set, None)
        if dataset_config is None:
            threshold = 1
            augmentation = None
            gt_augmentation = None
        else:
            threshold = dataset_config['threshold']
            try:
                augmentation = dataset_config['augmentation']
            except KeyError:
                augmentation = None
            try:
                gt_augmentation = dataset_config['gt_augmentation']
            except KeyError:
                gt_augmentation = None
        dataset = cls(config['model']['network']['net-params']['input_size'],
                      samples,
                      label_map,
                      config['model']['input_features'],
                      config['data_'+set]['gt_field'],
                      config['train']['datasets']['transforms'],
                      threshold,
                      config['model']['ignore_label'],
                      voxel_size,
                      augmentation, gt_augmentation)
        return dataset



    def __init__(self, input_size, samples, label_map,
                 input_features, gt_field, transforms=None, threshold=1,
                 default_label=-100, voxel_size=None,
                 augmentation=None, gt_augmentation=None):

        ## Adjust Tree GT zu voxel-size:
        print("  Transforming tree gt in voxel-space ...")
        timer = Timer()

        for sample in samples:
            sample['tree_set'].data /= voxel_size

        print("    Finished in", timer.time_string())

        VoxelCloudDataset_2D.__init__(self, input_size, samples, label_map,
                                      input_features, gt_field, transforms,
                                      threshold, default_label)

        self.augmentation    = augmentation
        self.gt_augmentation = gt_augmentation


    def __getitem__(self, index):
        vc = self.samples[index]['voxel_cloud']
        tr = self.samples[index]['tree_set']

        trees    = tr.data.copy()
        coords   = vc.get_xyz().copy()
        features = self.samples[index]['features_2D'].copy()
        labels   = self.samples[index]['labels_2D'].copy()
        shape    = tuple(labels.shape)
        
        #fig, ax = plt.subplots()
        #plt.imshow(labels)
        #fig, ax = plt.subplots()
        #plt.imshow(features[-1,:,:])
        #for tree in trees:
            #ax.add_artist(mplcircle((tree[1], tree[0]), radius=tree[3], edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125), path_effects=[withStroke(linewidth=5, foreground='w')]))
        #plt.show()  

        # Augmentation (coords must stay positive!!)
        if self.augmentation is not None:
            if 'scaling' in self.augmentation.keys():
                s = random.uniform(*self.augmentation['scaling'])
                coords *= s
                trees[:,0:4] *= s   #x,y,z,radius

                features_zoomed = ndimage.zoom(features, (1, s, s), order=0)
                labels_zoomed   = ndimage.zoom(labels,   (s, s),    order=0)
                
                if s > 1:
                    features = features_zoomed[:, 0:shape[0], 0:shape[1]]
                    labels   = labels_zoomed  [   0:shape[0], 0:shape[1]]
                else:
                    sshape   = labels_zoomed.shape
                    features = np.zeros_like(features)
                    labels   = np.zeros_like(labels)
                    features[:, 0:sshape[0], 0:sshape[1]] = features_zoomed
                    labels  [   0:sshape[0], 0:sshape[1]] = labels_zoomed

            if 'translation' in self.augmentation.keys():
                t = np.random.randint(*self.augmentation['translation'], (1,2))
                coords[:,0:2] += t
                trees[:,0:2] += t
                
                t = t.squeeze()
                features = ndimage.shift(features, (0, t[0], t[1]), order=0)
                labels   = ndimage.shift(labels,   (   t[0], t[1]), order=0)

            if 'flip' in self.augmentation and self.augmentation['flip']:
                # (coords should already normed 0..127; must stay >0!)
                if random.randint(0,1):
                    m = shape[0]
                    coords[:,0] = m - coords[:,0]
                    trees[:,0]  = m - trees[:,0]
                    
                    features = np.flip(features, axis=1).copy()
                    labels   = np.flip(labels,   axis=0).copy()
                    
                if random.randint(0,1):
                    m = shape[1]
                    coords[:,1] = m - coords[:,1]
                    trees[:,1]  = m - trees[:,1] 
                    
                    features = np.flip(features, axis=2).copy()
                    labels   = np.flip(labels,   axis=1).copy()
                                  
            #fig, ax = plt.subplots()
            #plt.imshow(labels)
            #fig, ax = plt.subplots()
            #plt.imshow(features[-1,:,:])
            #for tree in trees:
                #ax.add_artist(mplcircle((tree[1], tree[0]), radius=tree[3], edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125), path_effects=[withStroke(linewidth=5, foreground='w')]))
            #plt.show()  


        # GT-Augmentation
        if self.gt_augmentation is not None:
            if 'scaling' in self.gt_augmentation.keys():
                l,h = self.gt_augmentation['scaling']
                s = np.random.random((len(tr),1))
                s = (h-l) * s + l
                trees[:,[3,4]] *= s   #radius, height
            if 'translation' in self.gt_augmentation.keys():
                t = np.random.randint(*self.gt_augmentation['translation'], (len(tr),2))
                # relative to radius (max(r)=20m=40v)
                trees[:,0:2] += t * trees[:,3,None] / 40 

        # Only for visualization intended:
        labels_2Dsem = self.samples[index]['labels_2D']
        labels_2Dsem = torch.from_numpy(labels_2Dsem)
        height = self.samples[index]['height']
        height = torch.from_numpy(height)

        # No need to convert coords to long, done in scn.InputLayer.       
        return {'data':  [torch.from_numpy(coords),
                          torch.from_numpy(features)],
                'labels': torch.from_numpy(labels.astype(np.int64)),
                'trees': [torch.from_numpy(trees)],
                'labels_2Dsem': [labels_2Dsem],
                'height': [height],
                'shape':  shape}


    @staticmethod
    def collate(batch):
        """ Creates a mini-batch ready to use in scn.InputLayer  """
        coords = []
        featrs = []
        labels = []
        trees  = []
        labels_2Dsem = []
        height = []
        shapes = []

        if not isinstance(batch, list):  # in case mini-batch-size = 1
            batch = [batch]

        batch_size = len(batch)

        for sample in batch:
            coords.append(sample['data'][0])
            featrs.append(sample['data'][1])
            labels.append(sample['labels'])
            trees.extend(sample['trees'])
            labels_2Dsem.extend(sample['labels_2Dsem'])
            height.extend(sample['height'])
            shapes.append(sample['shape'])

        # Add batch-relative sample index as additional coord (see scn.InputLayer):
        coords = [torch.cat([x, i * torch.ones((x.shape[0], 1))], dim=1)
                  for i, x in enumerate(coords)]

        return {'data':  [torch.cat(coords, dim=0),
                          torch.stack(featrs),
                          batch_size],
                'labels': torch.stack(labels),
                'trees' : trees,
                'labels_2Dsem': torch.stack(labels_2Dsem),
                'height': torch.stack(height),
                'shapes': shapes}










class VoxelCloudDataset_Tree(VoxelCloudDataset_3D2D):
    """
    TODO
    """


    @classmethod
    def from_config(cls, config, samples, label_map, set, voxel_size):
        dataset_config = config['train'].get('dataset_'+set, None)
        if dataset_config is None:
            threshold = 1
            augmentation = None
            gt_augmentation = None
        else:
            threshold = dataset_config['threshold']
            try:
                augmentation = dataset_config['augmentation']
            except KeyError:
                augmentation = None
            try:
                gt_augmentation = dataset_config['gt_augmentation']
            except KeyError:
                gt_augmentation = None
        dataset = cls(config['model']['network']['net-params']['input_size'],
                      samples,
                      label_map,
                      config['model']['input_features'],
                      config['data_'+set]['gt_field'],
                      config['train']['datasets']['transforms'],
                      threshold,
                      config['model']['ignore_label'],
                      voxel_size,
                      augmentation, gt_augmentation)
        return dataset



    def __init__(self, input_size, samples, label_map,
                 input_features, gt_field, transforms=None, threshold=1,
                 default_label=-100, voxel_size=None,
                 augmentation=None, gt_augmentation=None):

        ## Adjust Tree GT zu voxel-size:
        print("  Transforming tree gt in voxel-space ...")
        timer = Timer()

        for sample in samples:
            sample['tree_set'].data /= voxel_size

        print("    Finished in", timer.time_string())

        VoxelCloudDataset_3D2D.__init__(self, input_size, samples, label_map,
                                        input_features, gt_field, transforms,
                                        threshold, default_label)

        self.augmentation    = augmentation
        self.gt_augmentation = gt_augmentation


    def __getitem__(self, index):
        vc = self.samples[index]['voxel_cloud']
        tr = self.samples[index]['tree_set']

        coords = vc.get_xyz()
        labels = vc.get_fdata([self.gt_field])
        features = vc.get_fdata(self.feature_fields)

        # Augmentation (coords must stay positive!!)
        if self.augmentation is not None:
            if 'scaling' in self.augmentation.keys():
                s = random.uniform(*self.augmentation['scaling'])
                coords *= s
                tr.data[:,0:4] *= s   #x,y,z,radius
            if 'translation' in self.augmentation.keys():
                t = np.random.randint(*self.augmentation['translation'], (1,2))
                coords[:,0:2] += t
                tr.data[:,0:2] += t
            if 'flip' in self.augmentation and self.augmentation['flip']:
                # (coords should already normed 0..127; must stay >0!)
                if random.randint(0,1):
                    m = np.max(coords[:,0])
                    coords[:,0]  = -1 * coords[:,0]  + m
                    tr.data[:,0] = -1 * tr.data[:,0] + m
                if random.randint(0,1):
                    m = np.max(coords[:,1])
                    coords[:,1]  = -1 * coords[:,1]  + m
                    tr.data[:,1] = -1 * tr.data[:,1] + m
                # print(np.min(coords,0))
                # print(np.max(coords,0))
                # print('----')

        # GT-Augmentation
        if self.gt_augmentation is not None:
            if 'scaling' in self.gt_augmentation.keys():
                l,h = self.gt_augmentation['scaling']
                s = np.random.random((len(tr),1))
                s = (h-l) * s + l
                tr.data[:,[3,4]] *= s   #radius, height
            if 'translation' in self.gt_augmentation.keys():
                t = np.random.randint(*self.gt_augmentation['translation'], (len(tr),2))
                # relative to radius (max(r)=20m=40v)
                tr.data[:,0:2] += t * tr.data[:,3,None] / 40 

        # Only for visualization intended:
        labels_2Dsem = self.samples[index]['labels_2D']
        labels_2Dsem = torch.from_numpy(labels_2Dsem)
        height = self.samples[index]['height']
        height = torch.from_numpy(height)

        # No need to convert coords to long, done in scn.InputLayer.
        return {'data':  [torch.from_numpy(coords),
                          torch.from_numpy(features)],
                'labels': torch.from_numpy(labels.astype(np.int64)),
                'trees': [torch.from_numpy(tr.data)],
                'labels_2Dsem': [labels_2Dsem],
                'height': [height]}



    @staticmethod
    def collate(batch):
        """ Creates a mini-batch ready to use in scn.InputLayer  """
        coords = []
        featrs = []
        labels = []
        trees = []
        labels_2Dsem = []
        height = []

        if not isinstance(batch, list):  # in case mini-batch-size = 1
            batch = [batch]

        batch_size = len(batch)

        for sample in batch:
            coords.append(sample['data'][0])
            featrs.append(sample['data'][1])
            labels.append(sample['labels'])
            trees.extend(sample['trees'])
            labels_2Dsem.extend(sample['labels_2Dsem'])
            height.extend(sample['height'])

        # Add batch-relative sample index as additional coord (see scn.InputLayer):
        coords = [torch.cat([x, i * torch.ones((x.shape[0], 1))], dim=1)
                  for i, x in enumerate(coords)]

        return {'data':  [torch.cat(coords, dim=0),
                          torch.cat(featrs, dim=0),
                          batch_size],
                'labels': torch.squeeze(torch.cat(labels, dim=0)),
                'trees' : trees,
                'labels_2Dsem': torch.stack(labels_2Dsem, dim=0),
                'height': torch.stack(height, dim=0)}


