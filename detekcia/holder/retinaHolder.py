import dlib


class RetinaHolder:
    def __init__(self,
                 color_type: str = None,
                 det_rect: dlib.rectangle = None,
                 det_score: float = None,
                 iou: float = None,
                 det_index: int = None):

        self.color_type = color_type
        self.det_rect = det_rect
        self.det_score = det_score
        self.iou = iou
        self.det_index = det_index
