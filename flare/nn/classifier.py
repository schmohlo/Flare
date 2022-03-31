""" Generic (hopefully...) wrapper for PyTorch networks for classification.

by Stefan Schmohl, 2020
"""

import os
import torch
import sparseconvnet as scn
import matplotlib.pyplot as plt

from torch.nn.functional import softmax

from .model import Model
from flare.utils import Timer
from flare.utils import clean_accuracy
from flare.utils import conf_matrix
from flare.utils.metricsHistory import MetricsHistory
from flare.nn.lr_scheduler import LinearLR




class Classifier(Model):
    """ Generic (hopefully) wrapper for PyTorch networks for classification.

    Contains all necessary functionallity to "easily" train and test a
    PyTorch classifier."
    """


    @classmethod
    def from_config(cls, config, net, optimizer, loss_fu, callbacks, device):
        instance = cls(
            net,
            config['model']['input_features'],
            config['model']['classes'],
            optimizer,
            loss_fu,
            callbacks,
            device)
        return instance



    def __init__(self, net, input_features, classes, optimizer, loss_fu,
                 callbacks=[], device=None):
        """
        Args:
            net:            PyTorch Module = the network to train / test.
                            Allready initalized (with previous state or rand)
            input_features: List of strings that define the names of the input
                            channels of the model.
            classes:        Dict{label:name}, defines the names of the model's
                            output classes.
            optimizer:      PyTorch optimizer for training. May be None for
                            testing.
            loss_fu:        PyTorch loss function for training and test. In
                            test needed for loss, if required, but in any case
                            for its ignore_index.
            callbacks:      List of callback objects for training. See
                            flare.nn.callbacks.
            device:         PyTorch device. 'cuda' or 'cpu'
        """

        super().__init__(net, input_features, device)

        self.classes = classes
        self.loss_fu = loss_fu.to(self.device)

        # Properties used during training (class members for easer exchange
        # between methods):
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.history = MetricsHistory()
        self.stop_training = False  # Triggered by EarlyStopping



    def train(self, data_loader, val_data_loader=None, max_epochs=100,
              start_epoch=0, verbose_step=1):
        """ Train this classification network.

        Network state chechkpoints will be stored according to callbacks.

        Args:
            data_loader:        PyTorch data_loader for training data.
            val_data_loader:    PyTorch data_loader for validation data.
            max_epochs:         Maximum number of epochs to train. Default=100.
            start_epoch:        Number of the first epoch. Used to abort
                                training with max_epochs and correct logs, e.g.
                                if training is resumed from previous state.
            verbose_step:       Number of mini-batches between console status
                                updates. 0 for none.
        Returns:
            history:            MetricsHistory object.
            time:               Training time in seconds.
        """

        if val_data_loader is None:
            self.history = self.history.reindex(columns=['epoch', 'lr',
                                                         'running_loss',
                                                         'running_acc'])
        else:
            self.history = self.history.reindex(columns=['epoch', 'lr',
                                                         'running_loss',
                                                         'running_acc',
                                                         'val_loss', 'val_acc'])

        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.stop_training = False  # Triggered by EarlyStopping
        
        lr_scheduler = None  # local lr_scheduler for warmup in first epoch.

        timer = Timer()
        
        for epoch in range(start_epoch, start_epoch + max_epochs):
        
            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(data_loader) - 1)
                lr_scheduler = LinearLR(
                    self.optimizer, 
                    start_factor=warmup_factor, 
                    total_iters=warmup_iters
                )

            time_epoch = timer.time()
            # Init running metrics:
            loss_verbose = 0.0
            acc_verbose = 0.0
            loss_epoch = 0.0
            acc_epoch = 0.0

            # In case a SCN is used. From the scn repo example for sem seg:
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0

            for i, batch in enumerate(data_loader, 0):

                data = self._batch_to_device(batch['data'], data_loader)
                gt = batch['labels'].to(self.device)

                # Set training mode to true. Certain modules have different
                # behaviors in train/eval mode.
                self.net.train()
                # Zero gradients buffer to prevent accumulation:
                self.optimizer.zero_grad()
                # Forward pass:
                outputs = self.net(data)
                loss = self.loss_fu(outputs, gt)
                # Backpropagation:
                loss.backward()
                # Parameter update:
                self.optimizer.step()

                # Monitoring:
                _, pred = torch.max(outputs.detach(), 1)
                acc = clean_accuracy(pred, gt, self.loss_fu.ignore_index)
                loss_epoch += loss.item()
                acc_epoch += acc

                # Print every some mini-batches the training status:
                if verbose_step:
                    loss_verbose += loss.item()
                    acc_verbose += acc
                    if i % verbose_step == verbose_step - 1:
                        print('')
                        print('  Epoch %5d, %5d:   lr: %8.6f  acc: %5.2f%%  loss: %.4f ' %
                              (epoch, i + 1, self.optimizer.param_groups[0]['lr'],
                               acc_verbose / verbose_step * 100,
                               loss_verbose / verbose_step), end='')
                        loss_verbose = 0.0
                        acc_verbose = 0.0
                        
                if lr_scheduler is not None:
                    lr_scheduler.step()


            ## End and evaluate epoch:
            print('\n  ' + '-'*77)
            loss_epoch /= len(data_loader)
            acc_epoch /= len(data_loader)

            self._update_history(epoch, loss_epoch, acc_epoch, val_data_loader)
            self._verbose_epoch(epoch, timer.time() - time_epoch, timer.time())

            for cb in self.callbacks:
                cb.on_epoch_end(self)
            print('  ' + '-'*77)

            # Supposed to help against memory runout:
            torch.cuda.empty_cache()

            if self.stop_training:
                # Triggered by EarlyStopping callback
                break

        ## End Training:
        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.history, timer.time()




    ###########################################################################
    # Inference methods                                                       #
    #                                                                         #
    ###########################################################################



    def test(self, data_loader, probs=True, gt=False, loss=False, acc=False,
             numpy=False, merge=False, coords=False):
        """ Returns predcitions (labels, softmax-probs, etc.) and evaluations.

        Attention: all samples are merged into one global "batch"! So do better
                   not shuffle your dataloader when predicting.

        Args:
            data_loader:    PyTorch data_loader for data.
            probs:          Whether to return softmax pseudo probabilities.
            gt:             Whether to return matching ground truth labels.
            loss:           Whether to return overall loss.
            acc:            Whether to return overall accuracy (clean).
            numpy:          Whether to return in numpy array or PyTorch tensor.
            merge:          Whether to return labels, gt and probs as single,
                            merged vector or as a list containing results for
                            each individual sample.
        Returns:
            labels [, probs] [, gt] [, loss] [, acc]
        """

        milfs_  = []
        probs_  = []
        labels_ = []
        gt_     = []
        loss_   = []
        acc_    = []
        numel_  = []   # to save memory.
        coords_ = []


        ignore = self.loss_fu.ignore_index

        # Set training mode to false / set eval mode. Certain modules like
        # Dropout have different behaviors in training/evaluation mode.
        self.net.eval()
        with torch.no_grad():  # prevent gradient calculation

            for batch in data_loader:
                data = self._batch_to_device(batch['data'], data_loader)
                outputs_gpu = self.net(data).detach()
                outputs = outputs_gpu.cpu()  # save GPU memory

                labels_batch = torch.argmax(outputs, dim=1)
                labels_.append(labels_batch)

                if hasattr(self.net, 'milf'):
                    milfs_.append(self.net.milf.cpu())

                if probs:
                    probs_batch = softmax(outputs, dim=1)
                    probs_.append(probs_batch)

                if loss or acc or gt:
                    gt_batch_gpu = batch['labels'].to(self.device).detach()
                    gt_batch = gt_batch_gpu.cpu()
                    if loss or acc:
                        numel_batch = gt_batch.numel()
                        numel_.append(numel_batch)
                    if gt:
                        gt_.append(gt_batch)

                if loss:
                    loss_batch = self.loss_fu(outputs_gpu, gt_batch_gpu).detach().cpu()
                    loss_.append(loss_batch * numel_batch)

                if acc:
                    acc_batch = clean_accuracy(labels_batch, gt_batch, ignore)
                    acc_batch = torch.tensor(acc_batch)
                    acc_.append(acc_batch * numel_batch)
                    
                if coords:
                    coords_.append(data[0])

        if merge:
            labels_ = torch.cat(labels_)
            if probs:
                probs_ = torch.cat(probs_)
            if gt:
                gt_ = torch.cat(gt_)
            if hasattr(self.net, 'milf'):
                milfs_ = torch.cat(milfs_)
            if coords:
                coords_ = torch.cat(coords_)

        if loss or acc:
            numel_ = torch.sum(torch.tensor(numel_))
        if loss:
            loss_ = torch.stack(loss_)
            loss_ = (torch.sum(loss_) / numel_).item()
        if acc:
            acc_ = torch.stack(acc_)
            acc_ = (torch.sum(acc_) / numel_).item()

        output = (labels_,)
        if probs:
            output += (probs_,)
        if gt:
            output += (gt_,)
        if loss:
            output += (loss_,)
        if acc:
            output += (acc_,)
        if probs and hasattr(self.net, 'milf'):
            output += (milfs_, )
        if coords:
            output += (coords_, )

        if numpy:
            output1 = ()
            for e in output:
                if isinstance(e, torch.Tensor):
                    output1 += (e.numpy(),)
                else:
                    output1 += (e,)
            output = output1
        return output



    def predict(self, data_loader, probs, gt, numpy=False, merge=False):
        """ Shortcut for self.test without evaluation.
        Returns:
            labels
            props if probs = True
            gt    if gt    = True
        """
        return self.test(data_loader, probs, gt, loss=False, acc=False,
                         numpy=numpy, merge=merge)



    def evaluate(self, data_loader):
        """ Shortcut for self.test without probs nor gt. """
        _, loss, acc = self.test(data_loader, probs=False, gt=False, loss=True,
                                 acc=True)
        return loss, acc



    def post_training_eval(self, dataloaders, config, path2model, path2mcps,
                           label_maps, label_maps_inv, show=False):
        # Load last saved epoch (which is the best epoch if specified in config):
        self.load_net_from_checkpoint(path2mcps)

        test_metrics = {}

        for key, dataloader in dataloaders.items():
            # It is important to get acc from the classifier directly, because may
            # differ from later calculations due to ignore class and class mapping.
            labels, gt, loss, acc = self.test(data_loader=dataloader,
                                              probs=False, gt=True,
                                              loss=True, acc=True,
                                              numpy=True, merge=True)

            # -- I want to see the original labels for gt. ---
            # In theory, yes. But inverse Mapping may merge unique gt classes
            # that have been merged into single model class. So the conf matrix
            # is misleading. Better run seperate inference, where this is handled
            # more cleanly.
            # gt = label_maps_inv[key][gt]

            test_metrics[key+'_acc'] = acc
            test_metrics[key+'_loss'] = loss

            # Plot & Print Confusion Matrix
            cm = conf_matrix.create(gt.astype(int), labels,
                                    list(config['model']['classes'].keys()),
                                    list(config['model']['classes'].keys()))

            # data_classes = config['data_'+key]['classes']
            # l_map = {l: label_maps[key][l] for l in data_classes.keys()}
            if 'ignore_label' in config['model']:
                ignore_label = config['model']['ignore_label']
                ignore_labels_gt = [ignore_label]
                # ignore_labels_gt =  [i for i,v in enumerate(label_maps[key])
                                     # if v == ignore_label]
            else:
                ignore_labels_gt = None

            conf_matrix.plot(cm, config['model']['classes'], path2model,
                             file_suffix='_'+str(key),
                             # classes_pred=config['model']['classes'],
                             # label_map=l_map,
                             ignore_labels=ignore_labels_gt, show=show)

            f = open(os.path.join(path2model, 'conf_matrix__{}.txt'.format(key)), 'w')
            conf_matrix.print_to_file(cm, f,
                                      config['model']['classes'],
                                      # config['model']['classes'],
                                      # l_map,
                                      indent=4,
                                      ignore_labels=ignore_labels_gt)
            f.close()

            print('  Overall acc on {:14,d} {:5s} voxels: {:5.2f} %'.format(
                  len(gt), key, acc * 100))

        return test_metrics

    ###########################################################################
    # Utility methods                                                         #
    #                                                                         #
    ###########################################################################



    def _update_history(self, epoch, loss, acc, val_data_loader=None):
        """ Add metrics from current epoch to metrics history. """

        lr = self.optimizer.param_groups[0]['lr']
        if val_data_loader is None:
            self.history = self.history.append({'epoch': epoch, 'lr': lr,
                                                'running_loss': loss,
                                                'running_acc': acc})
        else:
            loss_val, acc_val = self.evaluate(val_data_loader)
            self.history = self.history.append(
                {'epoch': epoch, 'lr': lr,
                 'running_loss': loss, 'running_acc': acc,
                 'val_loss': loss_val, 'val_acc': acc_val},
                ignore_index=True)

