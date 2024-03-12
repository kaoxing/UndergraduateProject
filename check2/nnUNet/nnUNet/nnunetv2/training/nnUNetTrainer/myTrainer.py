# author: Kaoxing
# date: 2024-3-4
import torch
import copy
from torch import nn
from nnUNet.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnUNet.nnUNet.nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnUNet.nnUNet.nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


# todo 可以尝试将刚出现标签的几张图像删除，然后再训练
# todo 训练器
class myTrainer(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """
        Trainer for my model
        """
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        print("Inited myTrainer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                         num_input_channels, deep_supervision=enable_deep_supervision)
        # do some changes to the network
        print("network------------------\n", network)
        # 将UNet的编码器复制一份，然后将其连接到解码器
        encoder = copy.deepcopy(network.encoder)



if __name__ == '__main__':
    myTrainer = myTrainer()
