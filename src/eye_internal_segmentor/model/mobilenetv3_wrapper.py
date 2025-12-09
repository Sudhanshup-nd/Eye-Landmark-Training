import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, lraspp_mobilenet_v3_small
from .blink_classifier import blinker

class lraspp_mobilenet_v3_large_segmentor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_large(weights=None, num_classes=4)
        self.backbone.train()

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_large segmentor---')
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

class lraspp_mobilenet_v3_small_segmentor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_small(weights=None, num_classes=4)
        self.backbone.train()

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_small segmentor---')
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
        return x["out"]

class lraspp_mobilenet_v3_small_segmentor_64CH(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_small(weights=None, num_classes=4, inter_channels=64)
        self.backbone.train()

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_small segmentor (64 CH)---')
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
        return x["out"]

class lraspp_mobilenet_v3_small_segmentor_32CH(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_small(weights_small=None, num_classes=4, inter_channels=32)
        self.backbone.train()

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_small segmentor (32 CH)---')
        print('NUM PARAMS =', self.num_params)
        print()


    def set_train(self):
        if self.backbone is not None:
            self.backbone.train()

    def set_eval(self):
        if self.backbone is not None:
            self.backbone.eval()

    def forward(self, x):
        x = self.backbone(x)
        return x["out"]

class lraspp_mobilenet_v3_small_segmentor_16CH(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_small(weights_small=None, num_classes=4, inter_channels=16)
        self.backbone.train()

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_small segmentor (16 CH)---')
        print('NUM PARAMS =', self.num_params)
        print()


    def set_train(self):
        if self.backbone is not None:
            self.backbone.train()

    def set_eval(self):
        if self.backbone is not None:
            self.backbone.eval()

    def forward(self, x):
        x = self.backbone(x)
        return x["out"]

class lraspp_mobilenet_v3_large_segmentor_w_blink(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_large(weights=None, num_classes=4)
        self.backbone.train()

        self.blink_classifier = blinker(num_inputs=15360, dropout=True)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_large segmentor + blink---')
        print('NUM PARAMS =', self.num_params)
        print()
        #print(self.backbone)


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
        x_b = self.blink_classifier(x["features"])
        return x["out"], x_b

class lraspp_mobilenet_v3_small_segmentor_w_blink(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = lraspp_mobilenet_v3_small(weights=None, num_classes=4)
        self.backbone.train()

        self.blink_classifier = blinker(num_inputs=10240, dropout=True)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---lraspp_mobilenet_v3_small segmentor + blink---')
        print('NUM PARAMS =', self.num_params)
        print()
        #print(self.backbone)


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
        x_b = self.blink_classifier(x["processed_high"])
        return x["out"], x_b


if __name__=="__main__":
    m = lraspp_mobilenet_v3_large_segmentor_w_blink()
    x = torch.randn((1,1,64,64))
    y, yb = m(x)