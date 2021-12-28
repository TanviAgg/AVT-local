import os
import logging
import random
import subprocess
import pickle

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import func


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    func.train.main(cfg)


if __name__ == "__main__":
    with open('outputs/label_input_seq.pickle', 'rb') as handle:
        loaded_val = pickle.load(handle)
    logging.basicConfig(format=('%(asctime)s %(levelname)-8s'
                                ' {%(module)s:%(lineno)d} %(message)s'),
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    torch.multiprocessing.set_start_method('spawn')
    main()  # pylint: disable=no-value-for-parameter  # Uses hydra
