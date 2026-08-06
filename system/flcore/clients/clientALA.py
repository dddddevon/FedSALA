import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# MODIFIED: Added read_client_data_global for loading whole-label test shards
from utils.data_utils import read_client_data, read_client_data_global
from utils.ALA import ALA


class clientALA(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        
        self.loss = nn.CrossEntropyLoss()
        # explanation start: CIFAR-10 OPTIMIZER TWEAKS
        # The previous FedALA codebase just used plain SGD. 
        # To make ResNet18 perform well on CIFAR-10, we must add momentum, weight decay, and Cosine Annealing.
        # (Replacing the old vanilla optimizer here but keeping it in the comments for reference).
        
        # OLD DO NOT DELETE:
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # NEW FAIR TWEAK:
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=args.global_rounds * args.local_steps
        )
        # explanation end:

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.lower_layers_local = getattr(args, 'lower_layers_local', False)

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device,
                    lower_layers_local=self.lower_layers_local)


    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                
            # explanation start: CIFAR-10 TWEAKS
            # Step the Cosine Annealing learning rate schedule at the end of every local epoch
            # explanation end:
            self.scheduler.step()

    def local_initialization(self, received_global_model):
        if self.layer_idx == 0:
            for param, param_g in zip(self.model.parameters(), received_global_model.parameters()):
                param.data = param_g.data.clone()
        else:
            self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    # NEW: Load whole-label test shard for this client.
    # Unlike load_test_data() which loads the skewed (2-class) test set,
    # this loads from test_global/ which contains ALL 10 classes (~100 each).
    # Used to measure GENERALIZATION performance.
    def load_global_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data_global(self.dataset, self.id)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    # NEW: Evaluate this client's model on the whole-label test shard.
    # Identical logic to test_metrics() but runs on the balanced global test data.
    # This measures how well the model generalizes to ALL labels, not just the
    # 2 classes this client was trained on.
    # Uses try/except for AUC because early in training the model may predict
    # only a single class, causing roc_auc_score to fail.
    def test_metrics_global(self, model=None):
        testloader = self.load_global_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            auc = 0.0
        
        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num