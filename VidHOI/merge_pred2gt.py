from cv2 import imwrite
import os
import json
import pickle


# zhuyao haishi dui json file jinxing caozuo
# ruhe duqu dict bing jinxing zuhe shi zuizhongyao de
#

if __name__ == '__main__':


    # # result_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-annotation/validation'
    # pred_dir = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_4inference/0010'
    # pred_json = '3359075894.json'
    # gt_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-video/frames/0010'
    # gt_json = '3359075894.json'
    #
    # # output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_v1'
    # output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_merge'
    # # output_path = '/home/student-pc/pycharmProject/upt-38/upt/vidhoi/validation-video/frames'
    # res = []

    pred_dir = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_4inference'
    # pred_json = '3359075894.json'
    gt_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-video/frames'
    # gt_json = '3359075894.json'
    output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_merge'
    # res = []


    objects_name = [
        "person", "car", "guitar", "chair", "handbag", "toy", "baby_seat", "cat", "bottle", "backpack",
        "motorcycle", "ball/sports_ball", "laptop", "table", "surfboard", "camera", "sofa",
        "screen/monitor", "bicycle", "vegetables", "dog", "fruits", "cake", "cellphone", "cup",
        "bench", "snowboard", "skateboard", "bread", "bus/truck", "ski", "suitcase", "stool", "bat",
        "elephant", "fish", "baby_walker", "dish", "watercraft", "scooter", "pig", "refrigerator",
        "horse", "crab", "bird", "piano", "cattle/cow", "lion", "chicken", "camel", "electric_fan",
        "toilet", "sheep/goat", "rabbit", "train", "penguin", "hamster/rat", "snake", "frisbee",
        "aircraft", "oven", "racket", "faucet", "antelope", "duck", "stop_sign", "sink", "kangaroo",
        "stingray", "turtle", "tiger", "crocodile", "bear", "microwave", "traffic_light", "panda",
        "leopard", "squirrel"
    ]
    verbs_name = [
        "lean_on", "watch", "above", "next_to", "behind", "away", "towards", "in_front_of", "hit",
        "hold", "wave", "pat", "carry", "point_to", "touch", "play(instrument)", "release", "ride",
        "grab", "lift", "use", "press", "inside", "caress", "pull", "get_on", "cut", "hug", "bite",
        "open", "close", "throw", "kick", "drive", "get_off", "push", "wave_hand_to", "feed", "chase",
        "kiss", "speak_to", "beneath", "smell", "clean", "lick", "squeeze", "shake_hand_with", "knock",
        "hold_hand_of", "shout_at"
    ]
    # haiyou yixie map
    with open('./slowfast/datasets/vidor/idx_to_pred.pkl', 'rb') as f:
        idx_to_pred = pickle.load(f)  # number 50
    with open('./slowfast/datasets/vidor/idx_to_obj.pkl', 'rb') as f:
        idx_to_obj = pickle.load(f)  # number 78
    with open('./slowfast/datasets/vidor/pred_to_idx.pkl', 'rb') as f:
        pred_to_idx = pickle.load(f)  # number 50
    with open('./slowfast/datasets/vidor/obj_to_idx.pkl', 'rb') as f:
        obj_to_idx = pickle.load(f)  # number 78
    # pred_to_idx

    pred_root = os.walk(pred_dir)
    gt_root = os.walk(gt_dir)
    for path, dir_list, file_list in pred_root:
        for dir_name in dir_list:  # 0010
            root_2 = os.walk(os.path.join(path, dir_name))
            output_dir = os.path.join(output_path, dir_name)
            pred_path = os.path.join(pred_dir, dir_name)
            gt_path = os.path.join(gt_dir, dir_name)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            for path_2, dir_list_2, file_list_2 in root_2:  # 3359075894
                for file_name_2 in file_list_2:
                    pred_file = os.path.join(pred_path, file_name_2)
                    gt_file = os.path.join(gt_path, file_name_2)
                    # print("pred_path:", pred_file)
                    # print("gt_path:", gt_file)
                    output_json = os.path.join(output_dir, file_name_2)
                    res = []
                    with open(pred_file, 'r') as f:
                        all_preds = json.load(f)
                    with open(gt_file, 'r') as f:
                        all_gts = json.load(f)

                    pred_frame_id = all_preds['file_name']
                    pred = all_preds['pred']
                    hoi_objects = all_preds['hoi_objects']
                    scores = all_preds['scores']
                    bbox_o = all_preds['bbox_o']
                    bbox_h = all_preds['bbox_h']

                    gt_frame_id = all_gts['frame_name']
                    gt_anno = all_gts['annotation']
                    video_path = all_gts['video_path']

                    total_frames = 0
                    for i, curr_gt_file_name in enumerate(gt_frame_id):
                        cuur_pred_file_name = pred_frame_id[i]
                        if cuur_pred_file_name == curr_gt_file_name:
                            curr_gt_anno = gt_anno[i]
                            # gt for each frame
                            curr_gt_bbxh = curr_gt_anno['boxes_h']
                            curr_gt_bbxo = curr_gt_anno['boxes_o']
                            curr_gt_obj = curr_gt_anno['object']
                            curr_gt_verb = curr_gt_anno['verb']
                            # pred for each frame
                            curr_pred = pred[i]
                            curr_hoi_objects = hoi_objects[i]
                            curr_score = scores[i]
                            curr_bbxh = bbox_h[i]
                            curr_bbxo = bbox_o[i]
                            datameta = {
                                "file_name": curr_gt_file_name,
                                "gt_bbxh": curr_gt_bbxh,
                                "gt_bbxo": curr_gt_bbxo,
                                "gt_object": curr_gt_obj,
                                "gt_verb": curr_gt_verb,
                                "pred": curr_pred,
                                "hoi_object": curr_hoi_objects,
                                "score": curr_score,
                                "bbxh": curr_bbxh,
                                "bbxo": curr_bbxo
                            }
                            res.append(datameta)
                            total_frames += 1
                        else:
                            continue

                    # output_dir = os.path.join(output_path, video_path[:4])
                    # if not os.path.isdir(output_dir):
                    #     os.makedirs(output_dir)

                    print("total_frames: {} of video {}".format(total_frames, os.path.join(dir_name, file_name_2)))
                    # xianzai wo xuyao gaoyixie jiegou ba zhexie xinxi cunchu qilai
                    # output_json = os.path.join(output_dir, f'{video_id}.json')
                    # output_json = os.path.join(output_dir, f'{video_path[5:]}.json')
                    with open(output_json, 'w') as f:
                        json.dump(res, f)

    print('end')




