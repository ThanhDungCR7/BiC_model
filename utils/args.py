from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models

def add_management_args() -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """

    return 