# Adapted from Facebook by Roberto Pellerito
"""
Excav sequence dataset.
"""

from .mot17_sequence import ModifiedSequence


class ExcavSequence(ModifiedSequence):
    """excavator Dataset from RSL ETH Zurich.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    data_folder = 'EXCAV'
