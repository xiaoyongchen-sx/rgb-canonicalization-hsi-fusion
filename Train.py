from omegaconf import OmegaConf
from model.ResShift_model import ResShiftTrainer
import torch
import sys

if __name__ == "__main__":
    print(torch.cuda.is_available())
    path = sys.argv[1] if len(sys.argv) > 1 else r"config/harvard_baseline.yaml"
    configs = OmegaConf.load(path)
    Trainer = ResShiftTrainer(configs=configs)
    Trainer.train(configs.train.epochs,configs.train.test_frequency)  #verboser : test frequency

