import os
import json

import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm
# %matplotlib inline

import random
import math
import cv2
import numpy as np
import math
from collections import defaultdict
# from slowfast.datasets.cv2_transform import scale

# Choose to evaluate the results using commands in (1) or (2):

# ------------------------------------------------------------------------------------ #

### (1) Evaluating methods with GT bboxes: load validation results

### Baseline (Image-based)
# result_dir = './output/BASELINE_32x2_R50_SHORT_SCRATCH_EVAL_GT'
result_dir = './slowfast/datasets/vidor/vidhoi/output/BASELINE_32x2_R50_SHORT_SCRATCH_EVAL_GT'

### (1-1) Evaluating above methods with 22,808 examples in total
result_json_name = 'all_results_vidor_checkpoint_epoch_00020.pyth_proposal_less-168-examples.json'

# ------------------------------------------------------------------------------------ #

### (2) OR evaluating methods WITH DETECTED BBOXES loaded from vidvrd-mff: load validation results

# ------------------------------------------------------------------------------------ #

with open(os.path.join(result_dir, result_json_name), 'r') as f:
    all_results = json.load(f)

# Settings

delete_less_than_25_instances = False  # non-rare setting
delete_more_than_25_instances = True  # rare setting
assert not (delete_less_than_25_instances and delete_more_than_25_instances)

# For computing Temporal-mAP
only_evaluate_temporal_predicates = False  # should be True when visualizing correct prediction!

### Settings for visualizing demo images purpose only
is_demo = False
is_demo_incorrect_hois = False
is_demo_save_imgs = False
is_demo_show_imgs = False
is_demo_top_10 = False  # may be used with incorrect hois

only_demo_specific_videos = False
only_demo_specific_frames = False
to_demo_video_names = []
to_demo_frame_names = []


with open('./slowfast/datasets/vidor/idx_to_pred.pkl', 'rb') as f:
    idx_to_pred = pickle.load(f)  # number 50
with open('./slowfast/datasets/vidor/idx_to_obj.pkl', 'rb') as f:
    idx_to_obj = pickle.load(f)  # number 78
with open('./slowfast/datasets/vidor/pred_to_idx.pkl', 'rb') as f:
    pred_to_idx = pickle.load(f)  # number 50
# pred_to_idx
temporal_predicates = [  # number 25
    'towards',
    'away',
    'pull',
    'caress',
    'push',
    'press',
    'wave',
    'hit',
    'lift',
    'pat',
    'grab',
    'chase',
    'release',
    'wave_hand_to',
    'squeeze',
    'kick',
    'shout_at',
    'throw',
    'smell',
    'knock',
    'lick',
    'open',
    'close',
    'get_on',
    'get_off',
]

if only_evaluate_temporal_predicates:
    temporal_predicates_idx = [pred_to_idx[pred] for pred in temporal_predicates]

print(all_results[0].keys())

# Visualization

# json_file = './output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-spa_conf/all_results_vidor_checkpoint_epoch_00020.pyth_proposal_less-168-examples_demo-all.json'
# json_file = './slowfast/datasets/vidor/vidhoi/output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool-spa_conf/all_results_vidor_checkpoint_epoch_00020.pyth_proposal_less-168-examples.json'
json_file = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_merge/0010/3359075894.json'
with open(json_file, 'r') as f:
    res = json.load(f)
print(len(res))

idx = random.randint(0, len(res))
# idx = 5
idx = 308
print('idx:', idx)
print(res[idx].keys())
# print(res[idx]['orig_video_idx'][0]) # defaults no reference
# res[idx]['orig_video_idx'][0] = '1027/5042598042/5042598042_000061'

# print(res[idx]['gt_boxes'])

img_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-video/frames/0010/3359075894'
# img_path = os.path.join(img_dir, res[idx]['file_name'])  # default
img_path = os.path.join(img_dir, res[idx]['file_name'])

print(img_path)

img = plt.imread(img_path)
# plt.imshow(img)
# plt.show()


def scale(size, image):
    """
    Scale the short side of the image to size.
    Args:
        size (int): size to scale the image.
        image (array): image to perform short side scale. Dimension is
            `height` x `width` x `channel`.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
    """
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (
            height <= width and height == size
    ):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return img


def vis_detections_allclss(im, dets, dets_clss, vidor_clss, thresh=0.5,
                           proposal_scores=None, sub_obj_pair=False, pred_cls=-1):
    """Visual debugging of detections."""
    for i in range(len(dets)):
        bbox = tuple(int(np.round(x)) for x in dets[i])  # yi ge si she wu ru de qu zheng
        class_name = vidor_clss[int(dets_clss[i])]
        if sub_obj_pair:  # if evaluating on HOI pair
            class_name = class_name + '_s' if i == 0 else class_name + '_o'

        if proposal_scores is not None:
            print('proposal_scores', proposal_scores)
            score = proposal_scores[i]
            if score > thresh:
                #                 print(bbox)
                cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                #                 print(class_name, bbox, score)
                cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=2)
        else:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            #             print(class_name, bbox)
            cv2.putText(im, '%s' % (class_name), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=2)

    if pred_cls != -1:  # different input form?
        pred_name = idx_to_pred[pred_cls]
        box1_x1y1 = list(int(np.round(x)) for x in dets[0])[:2]
        box2_x1y1 = list(int(np.round(x)) for x in dets[1])[:2]
        box1_box2_mid = (np.array(box1_x1y1) + np.array(box2_x1y1)) / 2
        box1_box2_mid = tuple(int(np.round(x)) for x in box1_box2_mid)

        cv2.line(im, tuple(box1_x1y1), tuple(box2_x1y1), (255, 0, 0), 2)
        cv2.putText(im, pred_name, box1_box2_mid, cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=2)
    #         cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

    return im


vidor_classes = list(idx_to_obj.values())

# img = cv2.resize(img, (224,224))
# img_vis = scale(224, img)  # zoom image
img_vis = img  # zoom image

# visualize gt boxes
# img_vis = vis_detections_allclss(img, res[idx]['gt_boxes'], res[idx]['gt_obj_classes'], vidor_classes)

# visualize proposal boxes (same as gt boxes when in ORACLE model)
pred_bbox = []
gt_bbox = []
gt_classes = []
gt_scores = []

# len_gt = len(res[idx]['gt_bbxo'])
# for i, x in enumerate(res[idx]['bbxo']):
#     pred_bboxo = res[idx]['bbxo'][i]
#     pred_bboxh = res[idx]['bbxh'][i]
#     if i < len_gt:
#         gt_bboxo = res[idx]['gt_bbxo'][i]
#         gt_bboxh = res[idx]['gt_bbxh'][i]
#         gt_class = res[idx]['gt_object'][i]
#         gt_bbox.append(gt_bboxh)
#         gt_bbox.append(gt_bboxo)
#         gt_classes.append(gt_class[0])
#         gt_classes.append(gt_class[1])
#         gt_scores.append(1)

for i, x in enumerate(res[idx]['gt_bbxo']):
    gt_bboxo = res[idx]['gt_bbxo'][i]
    gt_bboxh = res[idx]['gt_bbxh'][i]
    gt_class = res[idx]['gt_object'][i]
    gt_bbox.append(gt_bboxh)
    gt_bbox.append(gt_bboxo)
    gt_classes.append(gt_class[0])
    gt_classes.append(gt_class[1])
    gt_scores.append(1)


img_vis = vis_detections_allclss(img_vis, gt_bbox,
                                 gt_classes, vidor_classes, 0.2)

# img_vis = vis_detections_allclss(img_vis, [x[1:] for x in res[idx]['proposal_boxes']],
#                                  [x[1] for x in res[idx]['proposal_classes']], vidor_classes, 0.2,
#                                  proposal_scores=[x[1] for x in res[idx]['proposal_scores']])

# Can use all other result file as idx remains in the same order
# img_vis = vis_detections_allclss(img_vis, [x[1:] for x in all_results[idx]['proposal_boxes']], [x[1] for x in all_results[idx]['proposal_classes']], vidor_classes, 0.2, proposal_scores=[x[1] for x in all_results[idx]['proposal_scores']])

# plt.imshow(img_vis)
# plt.show()


def vis_hoi(img_idx, sub_cls, pred_cls, obj_cls, gt_sub_box, gt_obj_box):
    # img_path = 'slowfast/datasets/vidor/frames/' + res[img_idx]['orig_video_idx'][0] + '.jpg'
    #     new_idx = f"{int(res[img_idx]['orig_video_idx'][0].split('_')[-1])-15:06d}"
    #     frame_name = f"{res[img_idx]['orig_video_idx'][0].split('_')[0] + '_' + new_idx}"
    new_idx = f"{int('1027/5042598042/5042598042_000061'.split('_')[-1]) - 15:06d}"
    frame_name = f"{'1027/5042598042/5042598042_000061'.split('_')[0] + '_' + new_idx}"
    img_path = f"slowfast/datasets/vidor/frames/{frame_name}.jpg"

    img = plt.imread(img_path)
    img = scale(224, img)
    img = vis_detections_allclss(img, [gt_sub_box, gt_obj_box], [sub_cls, obj_cls], vidor_classes, sub_obj_pair=True,
                                 pred_cls=pred_cls)
    return img, '/'.join(
        [frame_name.split('/')[0], frame_name.split('/')[2]])  # frame_name # '/'.join(frame_name.split('/')[:-1])


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


tp, fp, scores, sum_gt = {}, {}, {}, {}

"""change start here 0613"""
with open('/home/student-pc/pycharmProject/upt-38/VidHOI/slowfast/datasets/vidor/pred_to_idx.pkl', 'rb') as f:
    pred_to_idx = pickle.load(f)  # number 50
with open('./slowfast/datasets/vidor/obj_to_idx.pkl', 'rb') as f:
    obj_to_idx = pickle.load(f)  # number 78

with open('./slowfast/datasets/vidor/idx_to_pred.pkl', 'rb') as f:
    idx_to_pred = pickle.load(f)  # number 50
with open('./slowfast/datasets/vidor/idx_to_obj.pkl', 'rb') as f:
    idx_to_obj = pickle.load(f)  # number 78

# Construct dictionaries of triplet class
for result in tqdm(res):
    # bbox_pair_ids = result['gt_bbox_pair_ids']

    for idx, action_id in enumerate(result['gt_verb']):

        subject_label_idx = int(result['gt_object'][idx][0])
        object_label_idx = int(result['gt_object'][idx][1])
        triplet_class = (subject_label_idx, action_id, object_label_idx)
        sub_name = idx_to_obj[subject_label_idx]
        obj_name = idx_to_obj[object_label_idx]
        action_name = idx_to_pred[action_id]
        triplet_name = (sub_name, action_name, obj_name)
        if triplet_class not in tp:  # should also not exist in fp, scores & sum_gt
            print("triplet_name: ", triplet_name)
            tp[triplet_class] = []
            fp[triplet_class] = []
            scores[triplet_class] = []
            sum_gt[triplet_class] = 0
        sum_gt[triplet_class] += 1
        # start to combine all posible triplet

# delete triplet classes that have less than 25 instances
# if delete_less_than_25_instances or delete_more_than_25_instances:
#     triplet_classes_to_delete = []
#     for triplet_class, count in sum_gt.items():
#         if delete_less_than_25_instances and count < 25 or delete_more_than_25_instances and count >= 25:
#             triplet_classes_to_delete.append(triplet_class)
#     for triplet_class in triplet_classes_to_delete:
#         del tp[triplet_class], fp[triplet_class], scores[triplet_class], sum_gt[triplet_class]

# Collect true positive, false positive & scores


correct_det_count = correct_hoi_count = total_det_count = 0
at_least_one_pair_bbox_detected_count = total_count = 0

curr_test = {}

for img_idx, result in enumerate(tqdm(res)):  # for each keyframe
    gt_bbox_pair_matched = set()
    # preds_obj_names = result['hoi_object']
    # preds_obj_ids = obj_to_idx['{}'.format(preds_obj_names)]
    # preds_sub_ids = [0]*len(preds_obj_ids)
    # print("preds_obj_names: ", preds_obj_names)
    # print("preds_obj_ids: ", preds_obj_ids)
    gt_bbox_pair_ids = result['gt_object']

    # take only top 100 confident triplets

    #     preds_scores = [
    #         (math.log(result['preds_score'][i][j]) + \
    #          math.log(result['proposal_scores'][preds_bbox_pair_ids[i][0]][1]) + \
    #          math.log(result['proposal_scores'][preds_bbox_pair_ids[i][1]][1]), i, j)
    #         for i in range(len(result['preds_score'])) for j in range(len(result['preds_score'][i]))
    #     ]
    preds_scores = [
        (math.log(result['score'][i] if result['score'][i] > 0 else 1e-300) + \
         math.log(1) + \
         math.log(1), i)
        for i in range(len(result['score']))
    ]
    preds_scores.sort(reverse=True)
    preds_scores = preds_scores[:10] if is_demo_top_10 else preds_scores[:100]

    at_least_one_pair_bbox_detected = False

    for score, i in preds_scores:  # for each HOI prediction, i-th pair and j-th action
        pred_sub_cls = 0
        pred_obj_name = result['hoi_object'][i]
        if pred_obj_name == 'baseball bat':
            pred_obj_name = 'bat'
        if pred_obj_name in obj_to_idx:
            pred_obj_cls = obj_to_idx['{}'.format(pred_obj_name)]
        else:
            continue
        pred_rel_name = result['pred'][i]
        if pred_rel_name == 'wield':
            pred_rel_name = 'wave'
        if pred_rel_name in pred_to_idx:
            pred_rel_cls = pred_to_idx['{}'.format(pred_rel_name)]
        else:
            continue
        triplet_class = (pred_sub_cls, pred_rel_cls, pred_obj_cls)
        triplet_name = ('person', pred_rel_name, pred_obj_name)
        if triplet_name not in curr_test:
            curr_test[triplet_name] = []
            print("pred_triplet_name: ", triplet_name)
            # print("pred_triplet_class: ", triplet_class)
        if triplet_class not in tp:
            continue

        pred_sub_box = result['bbxh'][i]
        pred_obj_box = result['bbxo'][i]
        is_match = False
        max_ov = max_gt_id = 0
        for k, gt_bbox_pair_id in enumerate(gt_bbox_pair_ids):  # for each ground truth HOI
            gt_sub_cls = gt_bbox_pair_id[0]
            gt_obj_cls = gt_bbox_pair_id[1]
            gt_rel_cls = result['gt_verb'][k]

            gt_sub_box = result['gt_bbxh'][k]
            gt_obj_box = result['gt_bbxo'][k]
            sub_ov = bbox_iou(gt_sub_box, pred_sub_box)
            obj_ov = bbox_iou(gt_obj_box, pred_obj_box)

            # import pdb; pdb.set_trace()

            # kaishi bijiao yuzhi yiji max iou
            if gt_sub_cls == pred_sub_cls and gt_obj_cls == pred_obj_cls and sub_ov >= 0.5 and obj_ov >= 0.5:
                # 1. ruguo manzu leibie douneng duishang bingqie iou dou dayu 0.5 zhengque de det shuliang +1
                correct_det_count += 1
                if not at_least_one_pair_bbox_detected:
                    at_least_one_pair_bbox_detected = True
                    at_least_one_pair_bbox_detected_count += 1

                if gt_rel_cls == pred_rel_cls:
                    is_match = True
                    correct_hoi_count += 1
                    min_ov_cur = min(sub_ov, obj_ov)
                    if min_ov_cur > max_ov:
                        max_ov = min_ov_cur
                        max_gt_id = k

            total_det_count += 1
            # print(f'Current detection/total detection ratio: {correct_det_count/total_det_count:5f}')

        if is_match and max_gt_id not in gt_bbox_pair_matched:
            tp[triplet_class].append(1)
            fp[triplet_class].append(0)
            gt_bbox_pair_matched.add(max_gt_id)
        else:
            tp[triplet_class].append(0)
            fp[triplet_class].append(1)
        scores[triplet_class].append(score)
    total_count += 1
#     print(f'Current at_least_one_pair_bbox_detected ratio: {at_least_one_pair_bbox_detected_count} / {total_count} = {at_least_one_pair_bbox_detected_count/total_count:5f}')

# print(f'[Final] correct HOI/correct detection ratio: {correct_hoi_count / correct_det_count:5f}')
# print(f'[Final] correct detection/total detection ratio: {correct_det_count / total_det_count:5f}')
# print(
#     f'[Final] at_least_one_pair_bbox_detected ratio: {at_least_one_pair_bbox_detected_count} / {total_count} = {at_least_one_pair_bbox_detected_count / total_count:5f}')

# for result in all_results:
#     bbox_pair_ids = result['bbox_pair_ids']
#     import pdb; pdb.set_trace()
# #     result['preds_score']

#     for idx, score in enumerate(result['preds_score']):
#         pass

# print('tp:', tp)
# print('fp:', fp)
# print('scores:', scores)

def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
        # dengjianju de qu 11 ge dian /voc07 de biaozhun
    return ap

# initialize ap
ap = np.zeros(len(tp))
max_recall = np.zeros(len(tp))

# print("tp.keys(): ", tp.keys())
# print("len of tp.keys(): ", len(tp.keys()))

for i, triplet_class in tqdm(enumerate(tp.keys())):
    sum_gt_temp = sum_gt[triplet_class]

    if sum_gt_temp == 0:
        continue
    tp_temp = np.asarray((tp[triplet_class]).copy())
    fp_temp = np.asarray((fp[triplet_class]).copy())
    res_num = len(tp_temp)
    if res_num == 0:
        continue
    scores_temp = np.asarray(scores[triplet_class].copy())
    sort_inds = np.argsort(-scores_temp) # sort decreasingly  fanhui shuzu paixu de suoyin e.g. paizai diyiwei de shu yunlaide xiabiao shiduoshao

    fp_temp = fp_temp[sort_inds]  # anzhao score jueduizhi congdi daogao paixu
    tp_temp = tp_temp[sort_inds]

    fp_temp = np.cumsum(fp_temp)  # fanhui leijihe s_n
    tp_temp = np.cumsum(tp_temp)

    rec = tp_temp / sum_gt_temp # recall
    prec = tp_temp / (fp_temp + tp_temp) # precision

    ap[i] = voc_ap(rec, prec)
    max_recall[i] = np.max(rec)

mAP = np.mean(ap[:])
m_rec = np.mean(max_recall[:])

print('------------------------------------------')
# print('mAP: {:.5f} max recall: {:.5f}'.format(mAP, m_rec))
print('mAP: {:.5f} '.format(mAP))
print('------------------------------------------')

"""the whole sorted by map thing part 1"""
# # tp_counts = [len(value) for value in tp.values()]
# tp_counts = []
# total_count = 0
# for value in tp.values():
#     total_count += len(value)
#     tp_counts.append(len(value))
# tp_count_ratio = [count/total_count for count in tp_counts]
#
# tp_keys = list(tp.keys())
# unsorted_maps = []
# for idx, classAP in enumerate(ap):
#     s, p, o = tp_keys[idx]
#     count = tp_counts[idx]
#     count_ratio = tp_count_ratio[idx]
#     unsorted_maps.append((classAP, (s, p, o), count, count_ratio))
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP}, count = {count} ({count_ratio*100:.3f}%)')

"""the whole sorted by map thing"""
# # Sorted by mAP
# sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx == 30:
#         print('--')
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "hug"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'hug':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "lean_on"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'lean_on':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "ride"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'ride':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "towards"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'towards':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "hug"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'hug':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "away"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'away':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "play(instrument)"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'play(instrument)':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "push"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'push':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by mAP and view only "pull"
# # sorted_maps = sorted(unsorted_maps, reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] != 'pull':
#         continue
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
#
# # Sorted by nb of occurrence
# sorted_maps_by_count = sorted(unsorted_maps, key=lambda val:val[3], reverse=True)
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps_by_count):
#     if idx == 30:
#         print('--')
#     print(f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio*100:.3f}%)')
"""end here"""

# pred_APs = defaultdict(list)
# pred_count = defaultdict(int)
# total_count = 0
# for classAP, (s, p, o), count, count_ratio in unsorted_maps:
#     pred_APs[p].append(classAP)
#     pred_count[p] += count
#     total_count += count
#
# print('-- Predicate-wise mean AP --')
# pred_mean_APs = {}
# for idx, (p, APs) in enumerate(pred_APs.items()):
#     #     if idx == 17:
#     #         print('--')
#     print(f'{idx_to_pred[p]}: {sum(APs) / len(APs):5f} ({pred_count[p] * 100 / total_count:5f}%)')
#     pred_mean_APs[idx_to_pred[p]] = sum(APs) / len(APs)
#
# print(len(pred_mean_APs))
#
# # (person, away, person)
# # (person, towards, person)
# # (person, lift, ball/sports_ball)
# # (person, push, baby_walker)
# # (person, pull, baby_walker)
# # (person, hug, person)
# # (person, lean_on, person)

# example_mAPs = []
#
# for idx, (classAP, (s, p, o), count, count_ratio) in enumerate(sorted_maps):
#     if idx_to_pred[p] == 'away' and idx_to_obj[o] == 'person':
#         person_away_person = classAP
#         example_mAPs.append(person_away_person)
#
#     elif idx_to_pred[p] == 'towards' and idx_to_obj[o] == 'person':
#         person_towards_person = classAP
#         example_mAPs.append(person_towards_person)
#
#     elif idx_to_pred[p] == 'lift' and idx_to_obj[o] == 'ball/sports_ball':
#         person_lift_ball_sports_ball = classAP
#         example_mAPs.append(person_lift_ball_sports_ball)
#
#     elif idx_to_pred[p] == 'push' and idx_to_obj[o] == 'baby_walker':
#         person_push_baby_walker = classAP
#         example_mAPs.append(person_push_baby_walker)
#
#     elif idx_to_pred[p] == 'pull' and idx_to_obj[o] == 'baby_walker':
#         person_pull_baby_walker = classAP
#         example_mAPs.append(person_pull_baby_walker)
#
#     elif idx_to_pred[p] == 'hug' and idx_to_obj[o] == 'person':
#         person_hug_person = classAP
#         example_mAPs.append(person_hug_person)
#
#     elif idx_to_pred[p] == 'lean_on' and idx_to_obj[o] == 'person':
#         person_lean_on_person = classAP
#         example_mAPs.append(person_lean_on_person)
#
#     else:
#         continue
#     print(
#         f'({idx_to_obj[s]}, {idx_to_pred[p]}, {idx_to_obj[o]}): mAP = {classAP:.5f}, count = {count} ({count_ratio * 100:.3f}%)')
"""end here"""

"""draw things down there"""
# TODO
# example_mAPs = [
#     person_away_person,
#     person_towards_person,
#     person_lift_ball_sports_ball,
#     person_push_baby_walker,
#     person_pull_baby_walker,
#     person_hug_person,
#     person_lean_on_person,
# ]


