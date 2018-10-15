from .faster_rcnn import FasterRCNN

class VGG16(FasterRCNN):
    def __init__(self, classes, pretrained=True, class_agnostic=False):
        super(VGG16, self).__init__(classes, class_agnostic)
        self.pretrained = pretrained


    def init_param(self):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512


