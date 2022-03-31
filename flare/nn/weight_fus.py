""" Weight functions for torch loss functions

by Stefan Schmohl, 2020
"""

import torch



class WeightLookup():
    def __init__(self, weights, voxel_size=1):
        self.weights = weights      # torch.tensor
        self.factor = voxel_size
        
    def __call__(self, values):  # torch.tensor
        i = torch.floor((values * self.factor)).long()
        l = [i[:,j] for j in range(i.shape[1])] # make list of tensors for 2d indexing.
        try:
            return self.weights[l]
        except:
            breakpoint()
            pass
    def to(self, device):
        self.weights = self.weights.to(device)
            





def equal(dataset, classes=None):
    """ Weighting function to weight each class equal.
    
    Basically only to preserve constistency with other weight functions.
    """
    return None



def inv_class_freq(dataset, classes=None):
    """ Calcs for each class in dataset a the inverted class frequency
    
    Args:
        dataset:    
            A torch.dataset subclass. Make sure each sample is a dict that 
            must contain a 'label' key. It's value is a torch tensor.
        classes:
            A list of classes (class labels) to look for. Usefull, if there is
            the posibility of not all classes to be in the dataset, resulting
            in a too short weight vector.
    Returns:
        inv_freq:
            Tensor of consecutive class weights, sorted from lowest label to 
            highest.
    """
    class_weights = {}
    spc = _samples_per_class(dataset)
    samples_total = sum(list(spc.values()))

    for label, num_samples in spc.items():
        frequency = num_samples / samples_total
        class_weights[label] = 0 if frequency == 0 else 1 / frequency

    if classes == None:
        inv_freq = [i[1] for i in sorted(class_weights.items())]
        inv_freq = torch.tensor(inv_freq)
    else:
        inv_freq = torch.zeros(len(classes))   # should be max(classes)+1
        for c, w in class_weights.items():
            inv_freq[c] = w
    
    return inv_freq



def inv_class_freq_sqrt(dataset, classes=None):
    """ Like inv_class_freq, but it's square root. """
    return torch.sqrt(inv_class_freq(dataset, classes))



def _samples_per_class(dataset):
    spc = {}
    for sample in dataset:
        labels = sample['labels'].squeeze()
        unique, counts = torch.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            key = int(u.item())
            if key not in spc.keys():
                spc[key] = 0
            spc[key] += c.item()
    return spc
