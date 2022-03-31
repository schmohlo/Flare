
# Configuration Files (.json)

**Note:** partially out-dated!


## General Remarks
The training and prediction scripts are mainly controlled via json configs.
This allows to store them in seperate folders for each experiment / project. 
Also, this makes it easy and fast to switch between different setups. 

Each trained model is documented by the configs used for training, as they are 
copied into the respective model folders, thereby enabeling the training to be
continued later with the same settings.



## Config - Model

### Description
- **input_features:** Input to the network. Order is important. Data channels
  will be thinned out and fed into the network accordingly by the data-loader.
  Make sure the data channels are properly named trough the **field_renaming** 
  in the data config. **Note:** all in lower case.
- **classes:** Class labels the network will output, together with their
  meaning. Make sure it is a gapless order from 0 to n.  
- **ignore_label:** Class label to ignore in training loss. In this example it 
  is the "car" classes. Optional. Default is -100.
- **network:** Network definition of the model.
    - **architecture:** Name of the network class in `flare.nn.nets`
    - **net-params:** Constructor inputs for given architecture. See respective
      docstring.
    
#### Example:
``` json
{
    "input_features": ["intensity", "return_num", "num_returns"],
    "classes":
    {
        "0": "Powerline", 
        "1": "Low veg.",
        "2": "Imp. surface",
        "3": "Car",
        "4": "Fence/Hedge",
        "5": "Roof",
        "6": "Facade", 
        "7": "Shrub",
        "8": "Tree"
    },
	"ignore_label": 3
    "network":
    {
        "architecture": "SUNet",
        "net-params":
         {
                "input_size"      : [512, 512, 512],
                "filter_size"     : 3,
                "bias"            : true,
                "residual_blocks" : false,
                "downsample"      : [2, 2],
                "reps"            : 2,
                "nPlanes"         : [32, 64, 96, 128, 160, 192]
        }
    }
}
```



## Config - Data
Describes which data to use and how to fit it to a model. One file for each 
(sub-) dataset, like V3D train, val and test. You may want to create different
versions for different model configs.

### Description
- **dir:** folder in `PATH_2_SAMPLES` from which to get the input data.
- **gt_field:** data field / channel to use as ground truth, if existent.
- **field_renaming:** Dict (data:model) that maps original data field names 
  to a (common) set of field names used by the model. Only needed for those 
  fields for which renaming is necessary. **Note:** all in lower case.
- **field_norming:** How to normalized data fields when loading for a model.
  Use the old field names.  Optional.
- **class_mapping:** Dict (data:model) that maps data classes to model classes.
  Data classes are descriped by a json file inside the data folder.   
  Usefull for training, if you want to merge some classes, or put (multiple) 
  classes into the ***"ignore"*** class (see model config).  
  Usefull for inference, if a model was trained on fewer classes, because it 
  will trigger the creation of an assymetric confusion matrix for evaluation.

#### Example:
``` json
{
    "dir": "V3D_Val_CIR__vs050__a0_30_60_90_120_150_180_210_240_270_300_330__32x32__o00_00",
    
    "gt_field"        : "classification",
    "field_renaming": 
    {
        "amplitude"   : "intensity"
        "r"           : "red"
    },
	"field_norming":
    {
        "r":
        {
            "old_min": 0,
            "old_max": 255,
            "new_min": -1,
            "new_max": 1
        },
    },
    "class_mapping":
    {
        "Powerline"   : "Powerline", 
        "Low veg."    : "Vegetation",
        "Imp. surface": "Imp. surface",
        "Car"         : "ignore", 
        "Fence/Hedge" : "Vegetation", 
        "Roof"        : "Building",
        "Facade"      : "Building", 
        "Shrub"       : "Vegetation",       
        "Tree"        : "Vegetation"
    }
}
```



## Config - Training

### Description (selection)
- **weight_fu:** Function name to calculate weights per class for the loss 
  function. See `flare.nn.weight_fus` for possible options.
- **optim_paras:** Allows the definition of paras for multiple optimizers. The
  Idea is that in practice for the same problem valid paremeters for different 
  optimizers are found and the user wants to save them all for later use or 
  documentation.
- **monitor:** Metric to monitor in respective callback. Usually `"acc"`, 
  `"loss"`, `"acc_val"` or `"loss_val"`, depending to the model sub-class.
- **datasets:** Parameters for the specific dataset used in the code. See it's
  docstring for more details.

#### Example:
``` json
{
    "max_epochs"  : 200,
    "loss_fu"     : "torch.nn.CrossEntropyLoss",
    "weight_fu"   : "inv_class_freq",
    
    "optimizer"   : "torch.optim.SGD",
    "optim_paras" : 
    {
        "torch.optim.SGD":
        {
            "lr"          : 5e-2,
            "weight_decay": 2e-2,
            "momentum"    : 0.9,
            "nesterov"    : true
        }
    },
    
    "EarlyStopping": 
    {
        "monitor"  : "acc_val",
        "min_delta": 0.002,
        "patience" : 20
    },
    
    "ReduceLROnPlateau": 
    {
        "monitor"       : "acc_val",
        "factor"        : 0.7,
        "patience"      : 4,
        "verbose"       : true,
        "threshold"     : 0.002,
        "threshold_mode": "abs"
    },
    
    "ModelCheckpoint": 
    {
        "monitor"       : "acc_val",
        "save_best_only": true,
        "period"        : 1
    },

    "datasets":
    {
        "transforms"      : ["reduce_coords"],
    },

    "loader_train": 
    {
        "batch_size" : 32,
        "shuffle"    : true,
        "num_workers": 4,
        "drop_last"  : true
    },

    "loader_val":
    {
        "batch_size" : 32,
        "shuffle"    : true,
        "num_workers": 4,
        "drop_last"  : false
    }
}
```
