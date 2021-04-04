from .nonadaptivea3c_train import nonadaptivea3c_train
from .nonadaptivea3c_val import nonadaptivea3c_val
from .eotp_train import eotp_train
from .eotp_val import eotp_val

trainers = [ 
    'vanilla_train',
    'learned_train',
]

testers = [
    'vanilla_val',
    'learned_val',
]

variables = locals()