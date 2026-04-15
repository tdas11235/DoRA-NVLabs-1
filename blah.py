import json
import os
from transformers import TrainerCallback
import torch


class FrobDoraCosineCallback(TrainerCallback):
    def __init__(self, log_every=10, save_path="frob_dora_logs.jsonl", eps=1e-8):
        self.log_every = log_every
        self.save_path = save_path
        self.eps = eps

        self.initial_M = {}
        self.initial_D = {}

        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def _get_direction(self, module):
        if module.Wdecompose:
            W = module.weight
            return W / (torch.linalg.norm(W) + self.eps)

        elif module.r > 0:
            W = module.weight
            BA = module.lora_B.weight @ module.lora_A.weight
            new_W = W + BA * module.scaling
            return new_W / (torch.linalg.norm(new_W) + self.eps)

        return None

    def _flatten_normalize(self, D):
        d = D.reshape(-1)
        return d / (torch.norm(d) + self.eps)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        for name, module in model.named_modules():
            if hasattr(module, "m_scalar"):
                self.initial_M[name] = module.m_scalar.detach().clone()

                D = self._get_direction(module)
                if D is not None:
                    self.initial_D[name] = self._flatten_normalize(D.detach())

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0:
            return

        layer_logs = {}

        for name, module in model.named_modules():
            if name in self.initial_M:
                layer_entry = {}

                # ΔM
                delta_m = torch.norm(
                    module.m_scalar - self.initial_M[name]
                ).item()
                layer_entry["delta_M"] = float(delta_m)

                # ΔD (cosine drift)
                if name in self.initial_D:
                    current_D = self._get_direction(module)
                    if current_D is not None:
                        d_flat = self._flatten_normalize(current_D.detach())
                        cos = torch.dot(self.initial_D[name], d_flat)
                        delta_d = (1.0 - cos).item()
                        layer_entry["delta_D_cos"] = float(delta_d)

                layer_logs[name] = layer_entry

        log_entry = {
            "step": int(state.global_step),
            "layers": layer_logs,
        }

        with open(self.save_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Logged step {state.global_step}")