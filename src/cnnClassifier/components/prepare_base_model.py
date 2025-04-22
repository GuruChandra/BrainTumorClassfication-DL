import os
import urllib.request as request
import zipfile
from cnnClassifier.logging import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity import PrepareBaseModelConfig

import torchvision.models as models
import torch
from torch import nn
import torchinfo
from torchinfo import summary

torch.manual_seed(42)
torch.cuda.manual_seed(42)


class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config =config
    
    def get_base_model(self):
        try:
            self.model = models.resnet18(pretrained=True)
            torch.save(self.model,self.config.base_model_path)
        except Exception  as e:
            raise e

    @staticmethod
    def _prepare_full_model(model,classes:int,device:str):
        model.fc = nn.Linear(model.fc.in_features, classes)
        model = model.to(device)
        return model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model= self.model,
            classes=self.config.params_classes,
            device=self.config.params_device
        )
        logger.info(f"Model: \n {summary(self.full_model, input_size=[1,3, 224,224])}")

        torch.save(self.full_model,self.config.updated_base_model_path)