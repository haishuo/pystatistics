"""Private research consumer for the qualified DVEB exact-ARMA ABI."""

from .evaluator import DVEBCPUExactArma, DVEBCudaTransferExactArma

__all__ = ["DVEBCPUExactArma", "DVEBCudaTransferExactArma"]
