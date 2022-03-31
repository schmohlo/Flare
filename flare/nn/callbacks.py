""" This module contains callback function classes to use during training.

Each callback class has a function to call at the start of training, at the end
of each training epoch and finally one to call at the end of training.
Most of these callbacks are borrowed from or at least inspired by keras:
    https://github.com/keras-team/keras/blob/master/keras/callbacks.py

by Stefan Schmohl, 2019-2021
"""

from globals import *

import os
import numpy as np
import torch
import torch.optim.lr_scheduler as lrs



class Callback:

    def __init__(self):
        pass

    def on_train_begin(self, model):
        pass

    def on_epoch_end(self, model):
        pass

    def on_train_end(self, model):
        pass






class ModelAttributeScheduler(Callback):

    def __init__(self, attribute, start, step_size, end, patience=1):

        self.attribute = attribute
        self.start     = start
        self.step_size = step_size
        self.end       = end
        self.patience  = patience
        
        # self.last_epoch = 0
        
        if self.step_size < 0:
            self.reach = np.greater_equal
        else:
            self.reach = np.less_equal
        self.counter = 0


    def on_train_begin(self, model):
        setattr(model, self.attribute, self.start)
            
            
    def step(self, model, epoch=None):

        # if epoch is None:
            # epoch = self.last_epoch + 1
        # self.last_epoch = epoch

        if self.counter >= self.patience:
            new_value = getattr(model, self.attribute) + self.step_size
            if self.reach(new_value, self.end):
                setattr(model, self.attribute, new_value)
                print('  Epoch % 5d: %s changed to %f' % 
                      (epoch, self.attribute, getattr(model, self.attribute)))
                self.counter = 0
            else:
                print('  Epoch % 5d: %s already at end value: %f' % 
                      (epoch, self.attribute, getattr(model, self.attribute)))
        else:
            print('  Epoch % 5d: expecting to change %s in %d epochs.' %
                  (epoch, self.attribute, self.patience - self.counter))
            self.counter += 1


    def on_epoch_end(self, model):
        epoch = model.history['epoch'].tolist()[-1]
        self.step(model, epoch)







class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    Customization of
    https://github.com/keras-team/keras/blob/master/keras/callbacks.py

    Args
        path: string, dir path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`torch.save(the_model.state_dict(), PATH)`), else the full
            model is saved (`torch.save(the_model, PATH)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, path, monitor='loss_val', verbose=1,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
                 
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.last_saved_epoch = None
              
        self.filepath = os.path.join(path, 'epoch{epoch:05.0f}__%s{%s:.4f}' %
                                     (monitor, monitor))

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            elif 'm_ap' in self.monitor:
                self.monitor_op = np.greater
            elif 'loss' in self.monitor:
                self.monitor_op = np.less
            else:
                self.monitor_op = np.less
    
        self.best = np.Inf if self.monitor_op == np.less  else -np.Inf


    def on_train_begin(self, model):
        pass
            
            
    def on_epoch_end(self, model):
        logs = {}
        for c in model.history.columns:
            logs[c] = model.history[c].tolist()[-1]
        epoch = logs['epoch']
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:

            filepath = self.filepath.format(**logs)
            if self.save_best_only:
                current = logs[self.monitor]
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('  Epoch % 5d: %s improved from %0.4f,'
                              ' saving model to %s' %
                              (epoch, self.monitor, self.best, filepath))
                    self.best = current
                    if self.save_weights_only:
                        torch.save(model.net.state_dict(), filepath)
                    else:
                        torch.save(model.net, filepath)
                    self.last_saved_epoch = epoch
                    self.epochs_since_last_save = 0
                else:
                    if self.verbose > 0:
                        print('  Epoch % 5d: %s has not improved from %0.4f'
                              ' in %d epochs, not saving model.' %
                              (epoch, self.monitor, self.best,
                               self.epochs_since_last_save))
            else:
                if self.verbose > 0:
                    print('  Epoch % 5d: saving model to %s' %
                          (epoch, filepath))
                if self.save_weights_only:
                    torch.save(model.net.state_dict(), filepath)
                else:
                    torch.save(model.net, filepath)
                self.last_saved_epoch = epoch
                self.epochs_since_last_save = 0



class ReduceLROnPlateau(lrs.ReduceLROnPlateau, Callback):
    """ Wraps torch ReduceLROnPlateau but has lr_metric as attribute """

    def __init__(self, optimizer, monitor,
                 mode='auto', factor=0.1, patience=10, verbose=False,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0,
                 eps=1e-8):

        if mode == 'min':
            mode = 'min'
        elif mode == 'max':
            mode = 'max'
        else:
            if 'acc' in monitor:
                mode = 'max'
            elif 'm_ap' in monitor:
                mode = 'max'
            elif 'loss' in monitor:
                mode = 'min'
            else:
                mode = 'min'

        super().__init__(optimizer, mode, factor, patience, verbose, threshold,
                         threshold_mode, cooldown, min_lr, eps)

        self.mode = mode
        self.monitor = monitor
        self.verbose2 = verbose

        self.best = np.Inf if self.mode == 'min' else -np.Inf
        

    def on_train_begin(self, model):
        pass
            
            
    def step(self, metrics, epoch=None):
        current = metrics
        old = self.best
        s = ''

        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(float(current), self.best):
            self.best = current
            self.num_bad_epochs = 0
            s += ('  Epoch % 5d: %s improved from %0.4f,' %
                  (epoch, self.monitor, old))
        else:
            self.num_bad_epochs += 1
            s += ('  Epoch % 5d: %s has not improved from %0.4f in %d epochs,' %
                  (epoch, self.monitor, self.best, self.num_bad_epochs))

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            s += ' reducing lr on plateau.'
            do_r = True
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        else:
            s += (' expecting to reduce lr in %d epochs.' %
                  (self.patience - self.num_bad_epochs + 1))
            do_r = False

        if self.verbose2:
            print(s)
        if do_r:
            self._reduce_lr(int(epoch))


    def on_epoch_end(self, model):
        current = model.history[self.monitor].tolist()[-1]
        epoch = model.history['epoch'].tolist()[-1]
        self.step(current, epoch)



class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    Args:
        file_path: filepath of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
    """


    def __init__(self, file_path, separator=', '):
        super().__init__()
        self.fp = file_path
        self.sep = separator
        
        
    def on_train_begin(self, model):
        f = open(self.fp, 'a')
        f.write(self.sep.join(list(model.history.columns)) + '\n')
        f.close()


    def on_epoch_end(self, model):
        f = open(self.fp, 'a')
        latest_metrics = []
        for c in model.history.columns:
            values = model.history[c].tolist()
            latest_metrics.append('%.10f' % values[-1])
        f.write(self.sep.join(latest_metrics) + '\n')
        f.close()



class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    Customization of
    https://github.com/keras-team/keras/blob/master/keras/callbacks.py

    Args:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """


    def __init__(self, monitor='loss', min_delta=0, patience=10, verbose=1, 
                 mode='auto'):
        
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self._stopped_epoch = -1

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            elif 'm_ap' in self.monitor:
                self.monitor_op = np.greater
            elif 'loss' in self.monitor:
                self.monitor_op = np.less
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        

    def on_train_begin(self, model):
        pass


    def on_epoch_end(self, model):
        current = model.history[self.monitor].tolist()[-1]
        epoch = model.history['epoch'].tolist()[-1]
        
        if self.monitor_op(current - self.min_delta, self.best):
            old = self.best
            self.best = current
            self.wait = 0
            print('  Epoch % 5d: %s improved from %0.4f,'
                  ' resetting early stopping.' %
                  (epoch, self.monitor, old))
        else:
            if self.wait >= self.patience:
                self._stopped_epoch = epoch
                model.stop_training = True
            self.wait += 1
            print('  Epoch % 5d: %s has not improved from %0.4f'
                  ' in %d epochs, expecting to stop early in %d epochs.' %
                  (epoch, self.monitor, self.best, self.wait,
                   self.patience - self.wait + 1))


    def on_train_end(self, model):
        if self._stopped_epoch >= 0:
            print('  Early Stopping has been triggered.')

