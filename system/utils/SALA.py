"""
SALA.py — Selective Adaptive Local Aggregation (FedSALA Core Module)
=====================================================================

WHAT THIS FILE DOES:
    This is the brain of FedSALA. It decides WHICH parameters are important
    to each client (using Fisher Information) and only applies ALA's learned
    blending to those important parameters. The rest just take the global value.

HOW IT WORKS (5 Steps, every communication round):
    1. FISHER:  Measure how sensitive each parameter is to this client's data
    2. EMA:     Smooth Fisher scores across rounds (prevents flip-flopping)
    3. MASK:    Top P% Fisher → M=1 (personalize), Bottom → M=0 (use global)
    4. INIT:    M=0 params get global value. M=1 params get ALA blend.
    5. LEARN:   Learn the ALA blend weights, but ONLY for M=1 params (mask-gated)

KEY DIFFERENCE FROM ALA.py:
    ALA.py:  Picks parameters by LAYER POSITION (last N layers = personalize)
    SALA.py: Picks parameters by FISHER VALUE (data-driven, scattered everywhere)

    Result: More expensive per round, but smarter selection → better accuracy.

RELATIONSHIP TO OTHER FILES:
    - Ported Fisher computation from:  DynamicPFL/utils.py
    - Extends weight learning from:    FedALA/system/utils/ALA.py
    - Called by:                        FedALA/system/flcore/clients/clientSALA.py
"""

import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch import autograd
from torch.utils.data import DataLoader
from typing import List, Tuple


class SALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple],
                batch_size: int,
                rand_percent: int,
                fisher_threshold: float = 0.5,
                fisher_ema_alpha: float = 0.9,
                fisher_sample_percent: int = 10,
                fedsala_method: int = 3,
                eta: float = 1.0,
                device: str = 'cpu',
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize SALA module.

        Args:
            cid:                    Client ID (for logging)
            loss:                   Loss function (CrossEntropyLoss)
            train_data:             Client's local training data as list of (x, y) tuples
            batch_size:             Batch size for ALA weight learning
            rand_percent:           % of local data to sample for ALA weight learning (e.g., 80)
            fisher_threshold:       Fraction of params to personalize (0.5 = top 50% get M=1)
            fisher_ema_alpha:       EMA smoothing. Higher = more stable mask (0.9 recommended)
            fisher_sample_percent:  % of local data for Fisher computation (10 = use 10%)
            fedsala_method:         Which method variant to use (1, 2, 3, or 4):
                                      M1: High-Fisher=ALA, Low-Fisher=Global
                                      M2: High-Fisher=Freeze, Low-Fisher=ALA
                                      M3: High-Fisher=ALA, Low-Fisher=Freeze (default/hypothesis)
                                      M4: High-Fisher=Global, Low-Fisher=ALA
            eta:                    Learning rate for ALA weight updates (default: 1.0)
            device:                 'cuda' or 'cpu'
            threshold:              Convergence threshold for weight learning (std of losses < this)
            num_pre_loss:           Number of recent losses to check for convergence
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.fisher_threshold = fisher_threshold
        self.fisher_ema_alpha = fisher_ema_alpha
        self.fisher_sample_percent = fisher_sample_percent
        self.fedsala_method = fedsala_method
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        # --- State that persists across rounds ---

        # ALA weights: one scalar per parameter element, controls blend ratio.
        #   weight=1  → fully use global value
        #   weight=0  → fully keep local value
        # Initialized to all-ones (fully trust global) on first call.
        # IMPORTANT: Full-sized (covers ALL params), unlike ALA which only covers top layers.
        self.weights = None

        # start_phase=True means we haven't converged yet (will train until convergence).
        # After first convergence, switches to False → only 1 epoch per round.
        self.start_phase = True

        # Running average of Fisher values across rounds (prevents mask flip-flopping).
        # None until first round.
        self.fisher_ema = None

        # Current binary mask — stored for debugging. Access via self.current_mask after a round.
        self.current_mask = None

        # Tracks the current global round number for logging purposes
        self.round_idx = 0


    # =========================================================================
    # STEP 1: Compute Fisher Information
    # =========================================================================
    def compute_fisher_diag(self, model: nn.Module) -> list:
        """
        Measure how important each parameter is to THIS client's data.

        Fisher Information = E[gradient^2]. High Fisher = parameter matters a lot
        for this client's loss function. Low Fisher = parameter doesn't matter much.

        We only use a small sample (default 10%) to keep this fast.
        We normalize per-layer so conv layers don't dominate just because they're bigger.

        Returns: List of tensors (same shapes as model params), values in [0, 1].
        """

        # --- Sample a small subset for Fisher computation ---
        fisher_ratio = self.fisher_sample_percent / 100
        fisher_num = max(1, int(fisher_ratio * len(self.train_data)))
        fisher_idx = random.randint(0, max(0, len(self.train_data) - fisher_num))
        fisher_loader = DataLoader(
            self.train_data[fisher_idx:fisher_idx + fisher_num],
            self.batch_size, drop_last=False
        )

        # --- Compute diagonal Fisher: F_ii = E[(d log p / d theta_i)^2] ---
        model.eval()
        fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

        for data, labels in fisher_loader:
            if type(data) == type([]):
                data[0] = data[0].to(self.device)
            else:
                data = data.to(self.device)
            labels = labels.to(self.device)

            # Get log-probabilities from model output
            log_probs = torch.nn.functional.log_softmax(model(data), dim=1)

            # For each sample, compute gradient and square it
            for i, label in enumerate(labels):
                log_prob = log_probs[i, label]

                model.zero_grad()
                grad1 = autograd.grad(
                    log_prob, model.parameters(),
                    create_graph=False, retain_graph=True
                )

                # Fisher = average of gradient^2 across samples
                for fisher_val, grad_val in zip(fisher_diag, grad1):
                    fisher_val.add_(grad_val.detach() ** 2)

                del log_prob, grad1

        # --- Average over samples ---
        num_samples = max(1, len(fisher_loader.dataset))
        fisher_diag = [f / num_samples for f in fisher_diag]

        # =====================================================================
        # ELIMINATED: LAYER-WISE MIN-MAX NORMALIZATION
        # =====================================================================
        # Why this was removed:
        # The normalization process below was done layer-wise. This means the min
        # and max values were not extracted from the model as a whole, but separately
        # for each individual layer. Because there is a massive numerical difference 
        # in the raw Fisher values between layers (e.g., BN/FC layers naturally have 
        # much larger gradients than early Conv layers), normalizing layer-by-layer 
        # tampered with the values by forcing every layer to the same [0, 1] scale. 
        # As a result, the normalized values no longer reflected the critical magnitude 
        # differences *between* layers. Therefore, we eliminated this normalization 
        # step to allow the naturally high gradients of BN and FC layers to dictate 
        # the global mask threshold.
        # 
        # Code reference for what was bypassed:
        # normalized_fisher = []
        # for f in fisher_diag:
        #     f_min = torch.min(f)
        #     f_max = torch.max(f)
        #     if f_max - f_min > 1e-8:
        #         normalized_fisher.append((f - f_min) / (f_max - f_min))
        #     else:
        #         normalized_fisher.append(torch.zeros_like(f))
        # return normalized_fisher
        # =====================================================================

        return fisher_diag


    # =========================================================================
    # STEP 2: EMA Smooth Fisher Scores
    # =========================================================================
    def update_fisher_ema(self, current_fisher: list) -> list:
        """
        Smooth Fisher scores across rounds to prevent the mask from flip-flopping.

        Without EMA: a parameter might be M=1 in round 5, M=0 in round 6, M=1 in round 7...
        With EMA (alpha=0.9): the mask changes gradually, only switching when a trend is clear.

        Formula: fisher_ema = 0.9 * old_fisher_ema + 0.1 * new_fisher
        """

        if self.fisher_ema is None:
            # Round 1: no history to blend with, just use raw values
            self.fisher_ema = [f.clone() for f in current_fisher]
        else:
            # Round 2+: blend old (90%) with new (10%)
            alpha = self.fisher_ema_alpha
            self.fisher_ema = [
                alpha * ema + (1 - alpha) * curr
                for ema, curr in zip(self.fisher_ema, current_fisher)
            ]

        return self.fisher_ema


    # =========================================================================
    # STEP 3: Generate Binary Mask
    # =========================================================================
    def generate_mask(self, smoothed_fisher: list) -> list:
        """
        Convert smoothed Fisher scores into a binary mask.

        Example with fisher_threshold=0.5 (top 50%):
            - All Fisher values across ALL layers are pooled together
            - Find the value at the 50th percentile
            - Everything above → M=1 (this param gets ALA personalization)
            - Everything below → M=0 (this param just takes the global value)

        Returns: List of binary tensors (0.0 or 1.0), same shapes as model params.
        """

        # Pool all Fisher values into one big vector
        all_fisher = torch.cat([f.flatten() for f in smoothed_fisher])

        # Find the cutoff value at the (100-P)th percentile
        # e.g., fisher_threshold=0.5 → keep top 50% → cutoff at 50th percentile
        k = int((1.0 - self.fisher_threshold) * all_fisher.numel())
        if k <= 0:
            threshold_val = all_fisher.min() - 1  # edge case: all M=1
        elif k >= all_fisher.numel():
            threshold_val = all_fisher.max() + 1  # edge case: all M=0
        else:
            threshold_val = torch.kthvalue(all_fisher, k).values

        # Apply threshold: M=1 where Fisher >= cutoff
        mask = [
            (f >= threshold_val).float().to(self.device)
            for f in smoothed_fisher
        ]

        self.current_mask = mask
        return mask


    # =========================================================================
    # MAIN ENTRY POINT: Called every communication round
    # =========================================================================
    def adaptive_local_aggregation(self,
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        The main method. Called by clientSALA.local_initialization() every round.

        What happens:
            1. Compute Fisher → which params matter to this client?
            2. EMA smooth → don't flip-flop the mask
            3. Generate mask → M=1 (personalize) vs M=0 (take global)
            4. Initialize:
                 M=0 params → overwrite with global (this client doesn't need these)
                 M=1 params → blend local and global using learned weights
            5. Learn weights → train the blend ratio using local data (mask-gated)

        After this method returns, local_model contains the personalized parameters.
        """

        # --- Sample local data for weight learning ---
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(
            self.train_data[rand_idx:rand_idx + rand_num],
            self.batch_size, drop_last=False
        )

        # --- Get parameter references ---
        params_g = list(global_model.parameters())  # global (from server)
        params = list(local_model.parameters())      # local (trained on this client's data)

        # Increment round tracker globally for the client
        self.round_idx += 1

        # --- Skip round 1 (Bootstrapping phase) ---
        # In Round 1, the local model hasn't been structurally personalized yet.
        # Computing Fisher and ALA blending here is useless. We forcefully bootstrap
        # exactly from Round 2 onwards to guarantee valid Fisher mapping.
        if self.round_idx == 1:
            for param, param_g in zip(params, params_g):
                param.data = param_g.data.clone()
            return

        # =====================================================================
        # STEP 1-3: Fisher → EMA → Mask
        # =====================================================================
        current_fisher = self.compute_fisher_diag(local_model)
        smoothed_fisher = self.update_fisher_ema(current_fisher)
        mask = self.generate_mask(smoothed_fisher)

        # Debugging: Dump layer-wise Fisher distribution for Client 0
        if str(self.cid) == "0":
            
            # Calculate totals to visually prove global threshold percentage
            total_params = sum(m.numel() for m in mask)
            total_m1 = sum(m.sum().item() for m in mask)
            global_pct = (total_m1 / total_params) * 100
            
            # Using append mode 'a' to build a history across all rounds
            with open("fisher_distribution_client0.txt", "a") as fw:
                fw.write(f"\n=================== ROUND {self.round_idx} ===================\n")
                fw.write(f"Layer Name{'':<25} | Mean Fisher  | Max Fisher   | Selected (%)\n")
                fw.write("-" * 75 + "\n")
                layer_names = [name for name, _ in local_model.named_parameters()]
                for name, f_tensor, m_tensor in zip(layer_names, smoothed_fisher, mask):
                    mean_val = f_tensor.mean().item()
                    max_val = f_tensor.max().item()
                    selected = (m_tensor.sum().item() / m_tensor.numel()) * 100
                    fw.write(f"{name:<35} | {mean_val:<12.2e} | {max_val:<12.2e} | {selected:>5.1f}%\n")
                
                # Append Global Model Summary at the bottom
                fw.write("-" * 75 + "\n")
                fw.write(f"{'GLOBAL PARAMETER POOL':<35} | {'':<12} | {'':<12} | {global_pct:>5.1f}%\n")
                fw.write(f" -> Total Parameters:   {int(total_params):,}\n")
                fw.write(f" -> Protected M=1:      {int(total_m1):,}\n\n")

        # Log: how many parameters are in each zone?
        total_params = sum(m.numel() for m in mask)
        total_m1 = sum(m.sum().item() for m in mask)
        print(f'Client {self.cid}: M=1 params: {int(total_m1)}/{total_params} '
              f'({100*total_m1/total_params:.1f}%)')

        # =====================================================================
        # STEP 4 + 5: Method-Dependent Init & Weight Learning
        # =====================================================================
        # The method variant determines:
        #   - How param_t is initialized (which zone gets ALA blend vs freeze/global)
        #   - Which zone's weights get gradient updates (gate mask)
        #
        # Method 1: M=1(High Fisher)=ALA blend,  M=0(Low Fisher)=Global overwrite
        # Method 2: M=1(High Fisher)=Freeze local, M=0(Low Fisher)=ALA blend
        # Method 3: M=1(High Fisher)=ALA blend,  M=0(Low Fisher)=Freeze local  ★ default
        # Method 4: M=1(High Fisher)=Global overwrite, M=0(Low Fisher)=ALA blend
        # =====================================================================

        # Create a temporary model for weight learning (don't modify local_model directly)
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # NOTE: Unlike ALA.py, we CANNOT freeze any layers here.
        # M=1 params are scattered across ALL layers, so every layer needs gradients.
        # We use SGD(lr=0) just as a gradient computation tool (same trick as ALA.py).
        optimizer = torch.optim.SGD(params_t, lr=0)

        # Initialize blend weights to all-ones (= fully trust global) on first ever call
        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device)
                           for param in params]

        # --- Helper: compute blended init and weight gate based on method ---
        def apply_method(param_t, param, param_g, weight, m, method):
            """Set param_t.data based on method variant."""
            ala_blend = param.data + (param_g.data - param.data) * weight
            if method == 1:    # High=ALA, Low=Global
                param_t.data = m * ala_blend + (1 - m) * param_g.data.clone()
            elif method == 2:  # High=Freeze, Low=ALA
                param_t.data = m * param.data.clone() + (1 - m) * ala_blend
            elif method == 3:  # High=ALA, Low=Freeze  ★
                param_t.data = m * ala_blend + (1 - m) * param.data.clone()
            elif method == 4:  # High=Global, Low=ALA
                param_t.data = m * param_g.data.clone() + (1 - m) * ala_blend

        def get_weight_gate(m, method):
            """Return the mask that gates gradient flow to the ALA zone."""
            if method in (1, 3):  # ALA is on high-Fisher (M=1)
                return m
            else:                 # ALA is on low-Fisher (M=0)
                return (1 - m)

        # Set up temp model with the blended parameters (Step 4)
        for param_t, param, param_g, weight, m in zip(params_t, params, params_g,
                                                       self.weights, mask):
            apply_method(param_t, param, param_g, weight, m, self.fedsala_method)

        # =====================================================================
        # STEP 5: Weight Learning (train the blend ratio)
        # =====================================================================
        # Goal: find the optimal weight for each parameter in the ALA zone
        #       that minimizes the loss on local data.
        #       Non-ALA zone weights stay frozen (gate masks them out).

        losses = []  # track loss to detect convergence
        cnt = 0      # epoch counter
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)
                loss_value.backward()

                # Update blend weights using METHOD-GATED gradients
                for param_t, param, param_g, weight, m in zip(
                        params_t, params, params_g, self.weights, mask):
                    gate = get_weight_gate(m, self.fedsala_method)
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g.data - param.data) * gate),
                        0, 1)

                # Rebuild temp model with updated weights
                for param_t, param, param_g, weight, m in zip(
                        params_t, params, params_g, self.weights, mask):
                    apply_method(param_t, param, param_g, weight, m, self.fedsala_method)

            losses.append(loss_value.item())
            cnt += 1

            # After first convergence, only do 1 epoch per round (speed optimization)
            if not self.start_phase:
                break

            # Check convergence: if recent losses are stable (low std), stop
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client: {self.cid}\tStd: {np.std(losses[-self.num_pre_loss:]):.6f}'
                      f'\tSALA epochs: {cnt}')
                break

        self.start_phase = False

        # Copy the blended parameters back to the actual local model
        for param, param_t in zip(params, params_t):
            param.data = param_t.data.clone()
