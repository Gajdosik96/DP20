import dlib
from practical_part.objectDetection.holder.dlibHolder import DlibHolder
from practical_part.objectDetection.holder.retinaHolder import RetinaHolder


class MatchHolder:
    def __init__(self,
                 gt_rect: dlib.rectangle,
                 gt_index: int,
                 dlib_bw: DlibHolder = None,
                 dlib_rgb: DlibHolder = None,
                 retina_bw: RetinaHolder = None,
                 retina_rgb: RetinaHolder = None):

        self.gt_rect = gt_rect
        self.gt_index = gt_index
        self.dlib_bw = dlib_bw
        self.dlib_rgb = dlib_rgb
        self.retina_bw = retina_bw
        self.retina_rgb = retina_rgb
