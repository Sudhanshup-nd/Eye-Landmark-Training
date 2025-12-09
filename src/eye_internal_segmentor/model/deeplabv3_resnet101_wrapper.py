import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from .blink_classifier import blinker

class deeplabv3_resnet101_segmentor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = deeplabv3_resnet101(weights=None, num_classes=4)
        self.backbone.train()

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---deeplabv3_resnet101_segmentor---')
        print('NUM PARAMS =', self.num_params)
        print()
        #print(self.backbone)


    def set_train(self):
        if self.backbone is not None:
            self.backbone.train()

    def set_eval(self):
        if self.backbone is not None:
            self.backbone.eval()

    def forward(self, x):
        x = self.backbone(x)
        return x

# class deeplabv3_resnet101_segmentor_w_blink(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.backbone = deeplabv3_resnet101(weights=None, num_classes=4)
#         self.backbone.train()
#         self.blink_classifier = blinker(num_inputs=131072, dropout=True)

#         self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print('---deeplabv3_resnet101_segmentor  with blink head---')
#         print('NUM PARAMS =', self.num_params)
#         print()
#         #print(self.backbone)


#     def set_train(self):
#         if self.backbone is not None:
#             self.backbone.train()
#             self.blink_classifier.train()

#     def set_eval(self): 
#         if self.backbone is not None:
#             self.backbone.eval()
#             self.blink_classifier.eval()

#     def forward(self, x):
#         x = self.backbone(x)
#         #print(x["features"].shape)
#         x_b = self.blink_classifier(x["features"])
#         return x["out"], x_b
    
class deeplabv3_resnet101_segmentor_w_blink(torch.nn.Module):  
    def __init__(self):  
        super().__init__()  
  
        self.backbone = deeplabv3_resnet101(weights=None, num_classes=5) 
        #self.backbone.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        self.backbone.train()  
        self.blink_classifier = blinker(num_inputs=20480, dropout=True)  
  
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  
        print('---deeplabv3_resnet101_segmentor  with blink head---')  
        print('NUM PARAMS =', self.num_params)  
        print()  
  
    def set_train(self):  
        if self.backbone is not None:  
            self.backbone.train()  
            self.blink_classifier.train()  
  
    def set_eval(self):   
        if self.backbone is not None:  
            self.backbone.eval()  
            self.blink_classifier.eval()  
  
    def forward(self, x):  
        x = self.backbone(x)  
        x_b = self.blink_classifier(x["out"])  
        return x["out"][:,:4,:,:], x["out"][:,4,:,:], x_b 
    
