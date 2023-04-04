# +
from torchsummary import summary

from nets.UNet_Nested import NestedUNet

if __name__ == "__main__":
    model = NestedUNet(num_classes=2).train().cuda()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    summary(model,(3,512,512))
# -




