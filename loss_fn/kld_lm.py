
import torch
import torch.nn as nn


def softmax_with_T(logits: torch.Tensor, t: int = 1):
    """
    softmax with temperature T for soft targets
    """
    log_soft = torch.log_softmax(logits/t, dim=1)
    return log_soft


class KLDiv(nn.KLDivLoss):
    def __init__(self):
        # reduction is applied later
        super().__init__(reduction='none', log_target=True)

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor):
        """
        Loss between inp (logits/action from AVT Student) and tgt (logits from LM Teacher)
        Note: both converted to soft targets here.
        Args:
            inp: (B, C)
            tgt: (B, C)
        """
        assert inp.ndim == tgt.ndim
        assert inp.shape == tgt.shape
        T = 30
        res = super().forward(softmax_with_T(inp, T), softmax_with_T(tgt, T))
        return res * (T**2)
