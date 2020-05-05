import dlib


class DlibHolder:
    def __init__(self,
                 det_rect: dlib.rectangle = None,
                 det_score: float = None,
                 iou: float = None,
                 det_index: int = None):

        self.det_rect = det_rect
        self.det_score = det_score
        self.iou = iou
        self.det_index = det_index
