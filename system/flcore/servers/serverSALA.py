"""
serverSALA.py — FedSALA Server
================================

WHAT THIS FILE DOES:
    Orchestrates the FedSALA training process:
    1. Creates all clients (using clientSALA)
    2. Each round: send global model → clients personalize + train → aggregate

HOW IT DIFFERS FROM serverALA.py:
    ONLY ONE CHANGE: Uses clientSALA instead of clientALA.
    Everything else is IDENTICAL — same FedAvg aggregation, same evaluation.

    All FedSALA-specific logic (Fisher, mask, weight learning) lives
    ENTIRELY on the client side in SALA.py. The server doesn't know or
    care about Fisher — it just averages whatever the clients send back.

CALLED BY: main.py
DEPENDS ON: flcore/clients/clientSALA.py
"""

import copy
import numpy as np
import torch
import time
from flcore.clients.clientSALA import *
from utils.data_utils import read_client_data
from threading import Thread


class FedSALA(object):

    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # MODIFIED: Added rs_local_acc and rs_global_acc for dual-perspective evaluation.
        # rs_test_acc is kept for backward compatibility with Centralized baseline.
        self.rs_test_acc = []
        self.rs_local_acc = []       # skewed local test accuracy per eval round
        self.rs_global_acc = []      # whole-label test accuracy per eval round
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        # THE ONLY DIFFERENCE: clientSALA instead of clientALA
        self.set_clients(args, clientSALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    # ---- Main Training Loop (identical to serverALA) ----

    def train(self):
        """
        Main FL training loop:
        Each round: select clients → send global → clients train → aggregate
        """
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Send global model to ALL clients (triggers SALA personalization)
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # Selected clients train locally
            for client in self.selected_clients:
                client.train()

            # Collect and aggregate client models
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    # ---- Helper Methods (all identical to serverALA) ----

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def send_models(self):
        """Send global model to all clients. This triggers local_initialization()."""
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.local_initialization(self.global_model)

    def receive_models(self):
        """Collect models from selected clients, weighted by training samples."""
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        """Standard FedAvg: weighted average of all client models."""
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    # ---- Evaluation (identical to serverALA) ----

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    # NEW: Aggregate whole-label test results across all clients.
    # This mirrors test_metrics() but calls client.test_metrics_global() instead,
    # evaluating each client on its non-overlapping shard of the full-label test set.
    def test_metrics_global(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_global()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    # MODIFIED: evaluate() now computes BOTH local (skewed) and global (whole-label)
    # test accuracy. Returns (local_acc, global_acc) so main.py can use them
    # for method-specific early stopping score computation.
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_global = self.test_metrics_global()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        global_test_acc = sum(stats_global[2])*1.0 / sum(stats_global[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_local_acc.append(test_acc)
            self.rs_global_acc.append(global_test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Local Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Global Test Accuracy: {:.4f}".format(global_test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        return test_acc, global_test_acc
