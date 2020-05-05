import dlib


def intersect_over_union(box1: dlib.rectangle, box2: dlib.rectangle):
    if box1.contains(box2):
        overall_area = box1.area()
    elif box2.contains(box1):
        overall_area = box2.area()
    elif box1.intersect(box2).area():
        overall_area = (box1.area() - box1.intersect(box2).area()) + (box2.area() - box1.intersect(box2).area()) + box1.intersect(box2).area()
    else:
        return 0.0

    return box1.intersect(box2).area() / overall_area


def fill_matrix_with_iou(corr_matrix, bbox_gt, bbox_dt):
    for g_i, b_gt in enumerate(bbox_gt):
        for d_i, b_dt in enumerate(bbox_dt):
            corr_matrix[g_i][d_i] = intersect_over_union(b_gt, b_dt)