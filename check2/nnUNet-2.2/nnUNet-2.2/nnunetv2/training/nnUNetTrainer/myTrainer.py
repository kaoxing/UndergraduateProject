# author: Kaoxing
# date: 2024-3-4
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# todo 训练器
class myTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """
        Trainer for my model
        """
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        print("Inited myTrainer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    myTrainer = myTrainer()
