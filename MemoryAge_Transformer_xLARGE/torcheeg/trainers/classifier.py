from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import os

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


def classification_metrics(metric_list: List[str], num_classes: int):
    '''
    The function returns a MetricCollection containing all the metric calculators that were specified in metric_list.
    These can then be used during the training, validation, and testing phases to evaluate the model's performance.
    '''

    allowed_metrics = ['precision', 'recall', 'f1score', 'accuracy']

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1_score', 'accuracy'"
            )
        
    # A dictionary metric_dict is created to map each metric name to its corresponding torchmetrics class.    
    metric_dict = {
        'accuracy':
        torchmetrics.Accuracy(task='multiclass',
                              num_classes=num_classes,
                              top_k=1),
        'precision':
        torchmetrics.Precision(task='multiclass',
                               average='macro',
                               num_classes=num_classes),
        'recall':
        torchmetrics.Recall(task='multiclass',
                            average='macro',
                            num_classes=num_classes),
        'f1score':
        torchmetrics.F1Score(task='multiclass',
                             average='macro',
                             num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.
        ALL the details of operation related to the model, training, validation, and testing are encapsulated in this class.
        'I provide the model, the hyperparams, and the model will do the training and testing for me.'

        Example:
            trainer = ClassifierTrainer(model)      # give the model to the trainer
            trainer.fit(train_loader, val_loader)   # let the trainer do the training and validation
            trainer.test(test_loader)               # let the trainer do the testing

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories (classes) in the dataset. 
                               The output layer does not need to have a softmax activation function (why?).
            num_classes (int, optional): The number of categories in the dataset. 
                                         If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. 
                                         (defualt: :obj:`None`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy'. (default: :obj:`["accuracy"]`)
        
        method: fit
        method: test
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["f1score","accuracy"],
                 save_attention: bool = True,
                 save_test_f1: bool = False):

        # call the __init__() method of the parent class (pl.LightningModule)
        super().__init__()   

        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self._ce_fn = nn.CrossEntropyLoss()   # for the classification task

        # prepare the metric calculators
        self.init_metrics(metrics, num_classes)

        self.save_attention = save_attention
        self.save_test_f1 = save_test_f1

        self.save_hyperparameters()           # save the hyperparameters to the checkpoint file


    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        '''
        initializing the metrics that will be used during the training, validation, and testing phases of the model. 
        this function is called in the constructor (__init__ method) of the ClassifierTrainer class.
        '''

        # Initializes mean metric calculators for different LOSS. 
        # This will help in keeping track of the AVERAGE loss during the training phase.
        self.train_loss = torchmetrics.MeanMetric()
        # and the same for validation and testing
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        # Calls the classification_metrics function to initialize the metrics (e.g., accuracy and F1 score) 
        # that will be used during training. It uses the provided metrics list and num_classes to do so.
        # metrics = ["accuracy", "f1score"]
        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)


    def fit(self, train_loader: DataLoader, val_loader: DataLoader, fold_idx: int, max_epochs: int = 300, *args, **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch 
                                       (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch 
                                       (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            fold_idx (int): The index of the current fold. used for naming the best model for each fold.
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
                              in each epoch, the model will be trained on all the batches in the train_loader.
                                             and the model will be evaluated on all the batches in the val_loader.
        '''

        # early stopping
        # if the validation loss does not decrease for 5 epochs, the training will be stopped.
        early_stop_callback = EarlyStopping(min_delta=0.00,
                                            monitor='val_loss',
                                            patience=5,
                                            verbose=True,
                                            mode='min')

        # save the best model by early stopping
        # save the model with the minimum (mode='min', save_top_k=1) validation loss (monitor='val_loss')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath='./processed_dataset/Model_Checkpoints',
                                              filename=f'Best_model_Fold{fold_idx}',
                                              save_top_k=1, 
                                              mode='min')

        # the offcial pytorch lightning trainer
        trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback],
                             devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             enable_progress_bar = False,       # disable the progress bar
                             *args,
                             **kwargs)
        
        # trainer.fit():
        # NOTE: The 'self' parameter contains all the information about the ClassifierTrainer instance, which includes the model (self.model), 
        # as well as other configurations like metrics, learning rate, etc.
        # When you pass 'self' along with the train_loader and val_loader to trainer.fit(), PyTorch Lightning's Trainer is aware of 
        # everything it needs to train the model. For example:
        # - self.model
        # - train_loader and val_loader
        # - self.train_metrics and self.val_metrics
        # - The optimizer, learning rate, etc., which would usually be defined in other methods like configure_optimizers in the LightningModule
        return trainer.fit(self, train_loader, val_loader)


    def test(self, test_loader: DataLoader, fold_idx: int , *args, **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        -> _EVALUATE_OUTPUT: in the function signature indicates the type hint for the return value of the function.

                             '_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader'

                             _EVALUATE_OUTPUT is a type alias that specifies the expected type of the return value for the test method. 
                             It is defined as a list of dictionaries, where each dictionary contains key-value pairs 
                             with the keys as strings and the values as floats.

                             Suppose you have two test data loaders. The output might look like: 
                             [{'test_loss': 0.41, 'test_accuracy': 0.85}, {'test_loss': 0.35, 'test_accuracy': 0.88}].

        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch 
                                      (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            fold_idx (int): Retrive the best model for the current fold.
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             enable_progress_bar = False,     # disable the progress bar
                             *args,
                             **kwargs)
        
        return trainer.test(self, test_loader, ckpt_path=f'./processed_dataset/Model_Checkpoints/Best_model_Fold{fold_idx}.ckpt')

    # -------------------------- Lifecycle of Methods in fit and test: ----------------------
    # The following functions are part of PyTorch Lightning's lifecycle for the LightningModule class, of which your ClassifierTrainer is a subclass.
    # Here is how these methods are generally called:
    # - fit Method
    #   When you call the fit method, it internally calls the PyTorch Lightning's trainer.fit(), which manages the training process. 
    #   In each training loop iteration, the following methods are called automatically:
    #   - training_step: Called once for each batch of training data.
    #   - on_train_epoch_end: Called at the end of every training epoch.
    #   - validation_step: Called once for each batch of validation data if a validation loader is provided.
    #   - on_validation_epoch_end: Called at the end of every validation epoch.
    #   The fit method also sets up the optimizer by calling configure_optimizers.
    #   
    #   Note that on_validation_epoch_end method is executed before the on_train_epoch_end method. 
    #        The reason for this behavior is that PyTorch Lightning's workflow is designed to execute the validation epoch end hook 
    #        before the training epoch end hook. This is not a bug, but a design choice made by the PyTorch Lightning team.
    #        Therefore, you will get the validation metrics before the training metrics for each

    # - test Method
    #   When you call the test method, it internally calls PyTorch Lightning's trainer.test(), which manages the testing process. 
    #   In each testing loop iteration, the following methods are called automatically:
    #   - test_step: Called once for each batch of testing data.
    #   - on_test_epoch_end: Called at the end of the testing process.

    # - Other Methods
    #   - forward: This is called implicitly within the training_step, validation_step, and test_step methods when you do self(x).
    #   - configure_optimizers: Called once at the beginning of the training to configure the optimizers.
    #   - predict_step: This is for prediction/inference and isn't automatically called in the training/testing loop. 
    #                   It has to be manually triggered, typically via PyTorch Lightning's Trainer.predict() method if you are using it for inference.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        batch: a tuple of (x, y), where x is the input and y is the label
        batch_idx: the index of the current batch
        Returns a torch.Tensor which should represent the loss for this training step.
        '''
        # x is the input, y is the label
        x, y = batch

        # y_hat is the output of the model
        # self(x) is equivalent to self.forward(x)
        y_hat = self(x)

        # calculate the loss
        loss = self._ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),    # self.train_loss is a torchmetrics.MeanMetric object, calculate the RUNNING mean of the loss values among the shown batches
                 prog_bar=True,            # whether to show the progress bar
                 on_epoch=False,           # Do not aggregate this log on the epoch level. 
                 logger=False,             # Do not log this value for logger backends like TensorBoard.
                 on_step=True)             # Log this value on every step.
        # the metrics are being "logged" to the progress bar in every batch (on_step=True) that appears in the console during training, 
        # but not being sent to any external logging services since (logger=False).
        # the metrics is the RUNNING mean of the loss values among the shown batches (self.train_loss(loss)),
        # but no aggregation is done on the epoch level (on_epoch=False).

        # additional metrics stored in self.train_metrics are calculated and logged. 
        for i, metric_value in enumerate(self.train_metrics.values()):
            # metric_value is a torchmetrics.Metric object, a metric calculator
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,  
                     on_epoch=False,
                     logger=False,
                     on_step=True)
            
        return loss


    def on_train_epoch_end(self) -> None:
        '''
        Called at the end of the training epoch with the outputs of all training steps.
        '''
        self.log("train_loss",
                 self.train_loss.compute(),  # Calls the compute() method on self.train_loss to get the average loss over the entire epoch.
                 prog_bar=False,             # The metric will not be shown in the progress bar.
                 on_epoch=True,              # The metric will be aggregated on the epoch level.
                 on_step=False,              # This metric will not be logged at each training step.
                 logger=True)                # This metric will be sent to external logging services like TensorBoard.
        # in summary, calculate the metric, which is the AVERAGE loss value over the entire epoch (self.train_loss.compute())
        
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "   [Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f}  "
        print(str)

        # reset the metrics
        # The self.train_loss object is a torchmetrics.MeanMetric or similar, designed to compute a running mean of the loss values. 
        # It internally accumulates the losses for each batch during the epoch. If you don't reset this at the end of each epoch,
        # the running mean will continue to be influenced by the values from the previous epochs, leading to incorrect metrics.
        self.train_loss.reset()
        self.train_metrics.reset()


    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        Unlike the training_step, this function does not log the loss or any metrics. This is usually done in the validation_epoch_end method, 
        where the metrics for the entire epoch are logged.

        The update methods are used to update the metrics state. This is a batch-level operation, and the final metrics are usually computed and logged
        at the end of the validation epoch.
        '''
        x, y = batch
        y_hat = self(x)
        loss = self._ce_fn(y_hat, y)

        self.val_loss.update(loss)         # update the running mean of the loss values
        self.val_metrics.update(y_hat, y)  # update the metrics

        return loss


    def on_validation_epoch_end(self) -> None:

        self.log("val_loss",
                 self.val_loss.compute(),  # Calls the compute() method on self.val_loss to get the average loss over the entire epoch.
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True, 
                     on_step=False,
                     logger=True)

        # print the metrics
        if self.current_epoch == 0:
            print(f'\nEpoch {self.current_epoch}: ', end='')
        else:
            print(f'Epoch {self.current_epoch}: ', end='')

        str = "[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f}  "
        print(str, end='')

        self.val_loss.reset()
        self.val_metrics.reset()


    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        y_hat = self.model(x, save_attn_matrices=self.save_attention)
        # save_attn_matrices=True will be passed through LFP_Transformer.forward() to Transformer.forward() to Attention.forward()
        # And this flag informs the Attention.forward() to save the attention we want.
        # after running this line, the desired attention scores for the batch is saved in self.model.transformer.attn_matrices_each_batch

        if self.save_attention:
            # -----------------------------------
            # Initialize prediction storage attribute if it doesn't exist
            if not hasattr(self, "saved_predictions_all"):
                self.saved_predictions_all = []

            # Save the predictions of the current batch
            self.saved_predictions_all.append(y_hat.detach().cpu().numpy())

            # -----------------------------------
            # the first time in test_step, self.saved_attention does not exist, so define it here
            # self.saved_attention is an attribute of the ClassifierTrainer object
            if not hasattr(self, "saved_attention_all"):
                self.saved_attention_all = []  # Define it here if it doesn’t exist

            # accumlating the attention scores for all the batches
            self.saved_attention_all.append(self.model.transformer.attn_matrices_each_batch)
            # You can access self.saved_attention_matrices after the testing phase is completed, 
            # e.g., trainer.saved_attention_all
            # -----------------------------------

        loss = self._ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)

        return loss


    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        print(str)

        # for perturbation session, save the test f1 score
        if self.save_test_f1:
            self.test_f1 = self.trainer.logged_metrics['test_f1score']

        self.test_loss.reset()
        self.test_metrics.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer


    def predict_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        
        return y_hat