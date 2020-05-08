from pathlib import Path
from helpers.pickleHelper import PickleHelper

from practical_part.objectDetection.utils.utils import *
import numpy as np

INPUT_DETECTION_DLIB = Path("data/final_data/practical_part/pkl/dlib/DLIB_250.PKL")
INPUT_DETECTION_RETINA = Path("data/final_data/practical_part/pkl/retina/RETINANET_250_001.PKL")

INPUT_GROUND_TRUTH = Path("data/final_data/practical_part/pkl/face_ann_in_pkl/gt_250_faces.pkl")

OUTPUT_DIR_DLIB_BW_GT = Path("data/final_data/practical_part/corr_matrix/dlib/bw_gt")
OUTPUT_DIR_DLIB_RGB_GT = Path("data/final_data/practical_part/corr_matrix/dlib/rgb_gt")

OUTPUT_DIR_RETINA_BW_GT = Path("data/final_data/practical_part/corr_matrix/retinanet/bw_gt")
OUTPUT_DIR_RETINA_RGB_GT = Path("data/final_data/practical_part/corr_matrix/retinanet/rgb_gt")


def make_corr_matrix(detection, ground_truth, output_dir_bw, output_dir_rgb):
    gt_data = PickleHelper.load(path=ground_truth)

    dt_data = PickleHelper.load(path=detection)

    for image in gt_data:
        gt_detection = gt_data[image]
        detection = dt_data[image]

        # shape=(y, x)
        corr_matrix_rgb = np.zeros(shape=(len(gt_detection), len(detection['rgb'][0])))
        corr_matrix_bw = np.zeros(shape=(len(gt_detection), len(detection['bw'][0])))

        fill_matrix_with_iou(corr_matrix_rgb, bbox_gt=gt_detection, bbox_dt=detection['rgb'][0])
        fill_matrix_with_iou(corr_matrix_bw, bbox_gt=gt_detection, bbox_dt=detection['bw'][0])

        PickleHelper.save(path=(output_dir_bw/Path(f"{Path(image).stem}_001.pkl")).__str__(), data=corr_matrix_bw)
        PickleHelper.save(path=(output_dir_rgb/Path(f"{Path(image).stem}_001.pkl")).__str__(), data=corr_matrix_rgb)


if __name__ == '__main__':
    make_corr_matrix(detection=INPUT_DETECTION_DLIB, ground_truth=INPUT_GROUND_TRUTH, output_dir_bw=OUTPUT_DIR_DLIB_BW_GT, output_dir_rgb=OUTPUT_DIR_DLIB_RGB_GT)
    make_corr_matrix(detection=INPUT_DETECTION_RETINA, ground_truth=INPUT_GROUND_TRUTH, output_dir_bw=OUTPUT_DIR_RETINA_BW_GT, output_dir_rgb=OUTPUT_DIR_RETINA_RGB_GT)

