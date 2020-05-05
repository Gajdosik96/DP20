from pathlib import Path
from helpers.pickleHelper import PickleHelper
from practical_part.objectDetection.holder.matchHolder import MatchHolder
from practical_part.objectDetection.holder.dlibHolder import DlibHolder
from practical_part.objectDetection.holder.retinaHolder import RetinaHolder

from scipy.optimize import linear_sum_assignment

PATH_COR_MAT_DLIB_BW = Path("data/final_data/practical_part/corr_matrix/dlib/bw_gt")
PATH_COR_MAT_DLIB_RGB = Path("data/final_data/practical_part/corr_matrix/dlib/rgb_gt")

PATH_COR_MAT_RETINANET_BW = Path("data/final_data/practical_part/corr_matrix/retinanet/bw_gt")
PATH_COR_MAT_RETINANET_RGB = Path("data/final_data/practical_part/corr_matrix/retinanet/rgb_gt")

PATH_GT_DATA = Path("data/final_data/practical_part/pkl/face_ann_in_pkl/gt_250_faces.pkl")

DLIB_DATA = PickleHelper.load(Path("data/final_data/practical_part/pkl/dlib/DLIB_250.PKL"))
RETINA_DATA = PickleHelper.load(Path("data/final_data/practical_part/pkl/retina/RETINANET_250_001.PKL"))

OUTPUT_MATCHED_PKL = Path("data/final_data/practical_part/matched/matched_pkl_det001_m001.pkl")

get_matrix_path = lambda x, y: x/f"{y.stem}.pkl"
get_matrix_path_thr = lambda x, y: x/f"{y.stem}_001.pkl"

GT_DATA = PickleHelper.load(path=PATH_GT_DATA.__str__())


def process_matrix(corr_matrix, new_image_data, threshold_match: float, color: str, model: str, img_name: str):
    row_ind, col_ind = linear_sum_assignment(corr_matrix, maximize=True)

    number_gt_det, number_dt_det = corr_matrix.shape[0], corr_matrix.shape[1]
    detections_idx = [x for x in range(number_dt_det)]

    if model == "dlib":
        det_data = DLIB_DATA
    else:
        det_data = RETINA_DATA

    for gt_id, det_id in zip(row_ind, col_ind):
        max_iou_row = corr_matrix[gt_id][det_id]
        if max_iou_row >= threshold_match:
            if model == "dlib" and color == "bw":
                new_image_data["matched"][gt_id].dlib_bw = DlibHolder()
                new_image_data["matched"][gt_id].dlib_bw.det_index = det_id
                new_image_data["matched"][gt_id].dlib_bw.det_rect = det_data[img_name][color][0][det_id]
                new_image_data["matched"][gt_id].dlib_bw.iou = max_iou_row
                new_image_data["matched"][gt_id].dlib_bw.det_score = det_data[img_name][color][1][det_id]

                detections_idx.pop(detections_idx.index(det_id))
            elif model == "dlib" and color == "rgb":
                new_image_data["matched"][gt_id].dlib_rgb = DlibHolder()
                new_image_data["matched"][gt_id].dlib_rgb.det_index = det_id
                new_image_data["matched"][gt_id].dlib_rgb.det_rect = det_data[img_name][color][0][det_id]
                new_image_data["matched"][gt_id].dlib_rgb.iou = max_iou_row
                new_image_data["matched"][gt_id].dlib_rgb.det_score = det_data[img_name][color][1][det_id]

                detections_idx.pop(detections_idx.index(det_id))

            elif model == "retina" and color == "bw":
                new_image_data["matched"][gt_id].retina_bw = RetinaHolder()
                new_image_data["matched"][gt_id].retina_bw.det_index = det_id
                new_image_data["matched"][gt_id].retina_bw.det_rect = det_data[img_name][color][0][det_id]
                new_image_data["matched"][gt_id].retina_bw.iou = max_iou_row
                new_image_data["matched"][gt_id].retina_bw.det_score = det_data[img_name][color][1][det_id]

                detections_idx.pop(detections_idx.index(det_id))

            elif model == "retina" and color == "rgb":
                new_image_data["matched"][gt_id].retina_rgb = RetinaHolder()
                new_image_data["matched"][gt_id].retina_rgb.det_index = det_id
                new_image_data["matched"][gt_id].retina_rgb.det_rect = det_data[img_name][color][0][det_id]
                new_image_data["matched"][gt_id].retina_rgb.iou = max_iou_row
                new_image_data["matched"][gt_id].retina_rgb.det_score = det_data[img_name][color][1][det_id]

                detections_idx.pop(detections_idx.index(det_id))
        else:
            if color == "bw":
                if model == "dlib":
                    new_image_data["unmatched_bw_dlib"].append(det_id)
                    detections_idx.pop(detections_idx.index(det_id))

                else:
                    new_image_data["unmatched_bw_retina"].append(det_id)
                    detections_idx.pop(detections_idx.index(det_id))

            else:
                if model == "dlib":
                    new_image_data["unmatched_rgb_dlib"].append(det_id)
                    detections_idx.pop(detections_idx.index(det_id))

                else:
                    new_image_data["unmatched_rgb_retina"].append(det_id)
                    detections_idx.pop(detections_idx.index(det_id))

    if detections_idx:
        for unamtched_det_id in detections_idx:
            if color == "bw":
                if model == "dlib":
                    new_image_data["unmatched_bw_dlib"].append(unamtched_det_id)
                else:
                    new_image_data["unmatched_bw_retina"].append(unamtched_det_id)

            else:
                if model == "dlib":
                    new_image_data["unmatched_rgb_dlib"].append(unamtched_det_id)
                else:
                    new_image_data["unmatched_rgb_retina"].append(unamtched_det_id)


def create_empty_matched_object(ddict, gt_data):
    for idx, gt_det in enumerate(gt_data):
        ddict["matched"][idx] = MatchHolder(gt_index=idx, gt_rect=gt_det)


def run_match2pklsave():
    result = {}
    for img_name in GT_DATA:
        new_image_data = {"matched": {},
                          "unmatched_bw_dlib": [],
                          "unmatched_bw_retina": [],
                          "unmatched_rgb_dlib": [],
                          "unmatched_rgb_retina": []}

        create_empty_matched_object(new_image_data, GT_DATA[img_name])

        img_dlib_bw_path = PickleHelper.load(path=get_matrix_path(PATH_COR_MAT_DLIB_BW, Path(img_name)))
        process_matrix(corr_matrix=img_dlib_bw_path, new_image_data=new_image_data, threshold_match=0.5, color="bw", model="dlib", img_name=img_name)

        img_dlib_rgb_path = PickleHelper.load(path=get_matrix_path(PATH_COR_MAT_DLIB_RGB, Path(img_name)))
        process_matrix(corr_matrix=img_dlib_rgb_path, new_image_data=new_image_data, threshold_match=0.5, color="rgb", model="dlib", img_name=img_name)

        img_retina_bw_path = PickleHelper.load(path=get_matrix_path_thr(PATH_COR_MAT_RETINANET_BW, Path(img_name)))
        process_matrix(corr_matrix=img_retina_bw_path, new_image_data=new_image_data, threshold_match=0.5, color="bw", model="retina", img_name=img_name)

        img_retina_rgb_path = PickleHelper.load(path=get_matrix_path_thr(PATH_COR_MAT_RETINANET_RGB, Path(img_name)))
        process_matrix(corr_matrix=img_retina_rgb_path, new_image_data=new_image_data, threshold_match=0.5, color="rgb", model="retina", img_name=img_name)

        result[img_name] = new_image_data

    PickleHelper.save(path=OUTPUT_MATCHED_PKL.__str__(), data=result)


def run_match2pkl(iou_threshold):
    result = {}
    for img_name in GT_DATA:
        new_image_data = {"matched": {},
                          "unmatched_bw_dlib": [],
                          "unmatched_bw_retina": [],
                          "unmatched_rgb_dlib": [],
                          "unmatched_rgb_retina": []}

        create_empty_matched_object(new_image_data, GT_DATA[img_name])

        img_dlib_bw_path = PickleHelper.load(path=get_matrix_path(PATH_COR_MAT_DLIB_BW, Path(img_name)))
        process_matrix(corr_matrix=img_dlib_bw_path, new_image_data=new_image_data, threshold_match=iou_threshold, color="bw", model="dlib", img_name=img_name)

        img_dlib_rgb_path = PickleHelper.load(path=get_matrix_path(PATH_COR_MAT_DLIB_RGB, Path(img_name)))
        process_matrix(corr_matrix=img_dlib_rgb_path, new_image_data=new_image_data, threshold_match=iou_threshold, color="rgb", model="dlib", img_name=img_name)

        img_retina_bw_path = PickleHelper.load(path=get_matrix_path_thr(PATH_COR_MAT_RETINANET_BW, Path(img_name)))
        process_matrix(corr_matrix=img_retina_bw_path, new_image_data=new_image_data, threshold_match=iou_threshold, color="bw", model="retina", img_name=img_name)

        img_retina_rgb_path = PickleHelper.load(path=get_matrix_path_thr(PATH_COR_MAT_RETINANET_RGB, Path(img_name)))
        process_matrix(corr_matrix=img_retina_rgb_path, new_image_data=new_image_data, threshold_match=iou_threshold, color="rgb", model="retina", img_name=img_name)

        result[img_name] = new_image_data

    return result


if __name__ == '__main__':
    run_match2pklsave()

