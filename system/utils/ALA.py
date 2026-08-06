import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple



#explanation start:
#In Python 3, class ALA: and class ALA(object): are exactly the same thing. 
#Every class built in standard Python 3 automatically inherits 
# from the base object class behind the scenes. 


class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple], 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10,
                lower_layers_local: bool = False) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10
            lower_layers_local: If True, lower layers are kept as local (not overwritten by global).
                                If False (default), lower layers are overwritten by global model (original FedALA). Default: False

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.lower_layers_local = lower_layers_local

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """

        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=False)


        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # explanation start:

        #What does parameters() do and return? 
        # It returns a Python Generator containing all the learnable weight and bias matrices (tensors) of the neural network. 
        # However, a generator can only be looped over once (e.g., for p in model.parameters():), 
        # and you cannot pick out specific layers directly (you can't do model.parameters()[0]).
                
        # Why is changing it to a list() necessary? 
        # By wrapping it in list(), PyTorch unpacks the generator and stores every single layer's tensor as individual elements in a standard Python list. 
        # This is critical for ALA because of the slicing that happens later. 
        # Once it's a list, ALA can do things like params_g[0] (grab the first layer) or params_g[-self.layer_idx:] (grab only the last few layers).

        #explanation end:

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # explanation start:
        #What does index 0 mean here? In PyTorch, a model's parameters are stored in order, 
        #from the very first layer (like the first Convolutional layer nearest to the image) 
        #to the last layer.

        #params_g[0] and params[0] extract the first tensor (the weight matrix of the very 
        #first layer) in the global model and the local model, respectively.
        #explanation end:





        # preserve all the updates in the lower layers   <--- how does this exactly work?

        # --- LOWER LAYERS HANDLING ---
        # Original FedALA: lower layers (below ALA zone) are overwritten with global model.
        #   lower layers = global, upper layers = ALA blend.
        #
        # When lower_layers_local=True: lower layers stay LOCAL (not overwritten).
        #   This mirrors FedSALA M3: High-Fisher params -> ALA, Low-Fisher params -> local.
        #   With layer_idx=17, the top 17 tensors = 75.16% of all parameters.
        #
        if not self.lower_layers_local:
            # ORIGINAL: overwrite lower layers with global
            for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
                param.data = param_g.data.clone()
        # else: lower layers stay as-is (local knowledge preserved)
        #
        # Explanation:
        #   params[:-self.layer_idx]  -> everything EXCEPT the last layer_idx tensors (lower layers)
        #   param.data = param_g.data.clone() -> force local lower layers to match global exactly
        #   Result when lower_layers_local=False: lower layers = global, upper layers = ALA blend
        #   Result when lower_layers_local=True:  lower layers = local,  upper layers = ALA blend
        # --- END LOWER LAYERS HANDLING ---

      
        # for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
        #     param.data = param_g.data.clone()
        #Explanation start:
        
        #params is a list of all your local model's parameters (layers).
        #params_g is a list of the global model's parameters.
        #The Slicing ([:-self.layer_idx]): In Python, [:-N] slices a list to include everything except the last N items. If your model has 10 layers and self.layer_idx is 2, it gets the first 8 layers ("lower layers").
        
        # 1. zip
        #zip(...): Pairs up the local layer and global layer side-by-side.
        #param.data = param_g.data.clone(): 
        # It takes the raw numbers (.data) of the global parameter, 
        # creates an exact copy (.clone()), 
        # and blindy overwrites the local parameter with it.
        
        #Result: The lower layers of the local model are forced to match the global model exactly.    
        
        #zip() : takes two or more lists and pair them of item by item -> output: "iterator" of tuple
        #Example: zip([1,2,3], ['a','b','c']) -> [(1,'a'), (2,'b'), (3,'c')]
        #This allows you not to use index based loop

        # 2. .clone()
        # why is it used?
        # If you just write param.data = param_g.data, 
        # they will both point to the exact same physical memory address.
        # So if you update param.data, param_g.data will also be updated.
        # .clone() creates a new copy of the tensor in memory.
        
        #Explanation end:


        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())
        # explanation start:

        # model.parameters() -> returns an iterator over all the parameters of the model
        # 1. only loop over a iterator once -> how?
        # 2. no slicing
        # list(...) -> converts the iterator to a list
        
        # copy vs deepcopy
        # A PyTorch model is basically a box, containing smaller boxes (layers), 
        # containing data (tensors). A standard copy.copy() is "shallow"—it copies 
        # the outer box, but the inner boxes still point to the original memory addresses. 
        # copy.deepcopy() recursively digs all the way down and .clone()s absolutely 
        # everything so it is a truly independent object.


        # explanation end:


        # only consider higher layers
        params_p = params[-self.layer_idx:] # The higher layers of the local model
        params_gp = params_g[-self.layer_idx:] # The higher layers of the global model
        params_tp = params_t[-self.layer_idx:] # The higher layers of the temp local model

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)
        # explanation start:
        # optimizer is the strict set of rules that dictates exactly how the AI is allowed to follow that compass
        # What this does: New_Weight = Old_Weight - (Learning_Rate * Gradient)
        # It creates a standard PyTorch Stochastic Gradient Descent optimizer
        #  hooked up to params_tp (the higher layers of the temporary testing model).
        # The trick: 
        # Notice lr=0. 
        # The authors do not want this optimizer to actually update the model's weights. 
        # They only created it so they can easily use optimizer.zero_grad() later to wipe clean the gradient math between loops.
        # explanation end:

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]
        # explanation start:
        # What this does: 
        # On the very first iteration (when self.weights is empty), 
        # it creates the actual mixing variables. 
        # For every single tensor matrix in the higher layers, 
        # it creates a perfectly matching matrix entirely filled with 1.0s.
        # Why: 
        # A weight of 1.0 represents "purely Global Model". 
        # ALA starts out biased, assuming the Global Model is 100% correct, 
        # and will slowly lower these 1.0s if the local data proves they should be lower.
        # explanation end:




        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight
        # explanation start:
        # What this does: 
        # This is the core hybridization formula: $Hybrid = Local + ((Global - Local) \times Weight)$. 
        # Because the weights are currently all 1.0, this essentially sets the temporary model (param_t) 
        # to be mathematically identical to the global model (param_g) right at the start.
        # explanation end:


        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                # explanation start:
                # What this does: Moves the data to the GPU (or CPU) so the model can read it. 
                # The if/else is just a safety check: depending on how the dataset was built, 
                # x might be a raw image tensor, or it might be a list containing an image tensor.
                # explanation end:

                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()
                # explanation start:
                # zero_grad() uses that dummy optimizer to perfectly clean the blackboard of any old math.
                # model_t(x) pushes the images through the temporary hybrid model to get predictions.
                # self.loss(...) checks how wrong the model was compared to the true labels.
                # .backward() does the heavy lifting: it calculates exactly how much the loss would decrease if we tweaked the parameters inside model_t.
                # explanation end:

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)
                

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone() # I accidently deleted this line and it reoccurred