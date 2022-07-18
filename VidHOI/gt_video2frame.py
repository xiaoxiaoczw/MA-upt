from cv2 import imwrite
import os
import json
import pickle


# zhuyao haishi dui json file jinxing caozuo
# ruhe duqu dict bing jinxing zuhe shi zuizhongyao de
#

if __name__ == '__main__':
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

    total_frames = 0
    result_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-annotation/validation'
    root = os.walk(r'/home/student-pc/MA/dataset/Vidhoi/validation-annotation/validation')
    for path, dir_list, file_list in root:
        for dir_name in dir_list:
            root_3 = os.walk(os.path.join(path, dir_name))
            for path_3, dir_list_3, file_list_3 in root_3:
                for file_name_3 in file_list_3:
                    print("file_name:", os.path.join(path_3, file_name_3))

                    with open(os.path.join(path_3, file_name_3), 'r') as f:
                        all_results = json.load(f)

                    # ranhou kaishi dui duqu daode json jieguo all_res zuo chuli
                    """ TODO """
                    # frame name example 3598080384_000001.jpg

                    video_path = all_results['video_path']
                    video_path = os.path.splitext(video_path)[0]
                    video_id = all_results['video_id']
                    frame_count = all_results['frame_count']
                    width = all_results['width']
                    height = all_results['height']
                    sub_obj = all_results['subject/objects']
                    trajectories = all_results['trajectories']
                    relation_instances = all_results['relation_instances']

                    # print("all_results:\n", all_results)
                    # print("video_path:\n", video_path)
                    # print("frame_count:\n", frame_count)
                    # print("width:\n", width)
                    # print("height:\n", height)
                    # print("sub_obj:\n", sub_obj)
                    # print("trajectories:\n", trajectories)
                    # print("relation_instances:\n", relation_instances)

                    # output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_v1'
                    output_path = '/home/student-pc/pycharmProject/upt-38/upt/vidhoi/validation-video/frames'
                    # output_dir = os.path.join(output_path, os.path.splitext(video_path)[0])
                    output_dir = os.path.join(output_path, video_path[:4])
                    print("output_dir: ", output_dir)
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)

                    total_frames += frame_count
                    print("total_frames:", total_frames)

                    # xianzai wo xuyao gaoyixie jiegou ba zhexie xinxi cunchu qilai

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


                    # chuliwei meiyou shuomingwenzi zhiyou list de yizhongdongxi
                    sub_obj_list = []
                    trajectories_list = []
                    relation_instances_list = []
                    action_pair_list = []
                    frame_name_list = []
                    size_list = []
                    anno_list = []
                    anno = []
                    sub_obj_dict = {}

                    for idx, relation_instances_label in enumerate(relation_instances):
                        sub_id = relation_instances_label['subject_tid']
                        obj_id = relation_instances_label['object_tid']
                        action_name = relation_instances_label['predicate']
                        begin_fid = relation_instances_label['begin_fid']
                        end_fid = relation_instances_label['end_fid']
                        relation_instances_list.append([sub_id, obj_id, action_name, begin_fid, end_fid])

                    # print("relation_instances_list:\n", relation_instances_list)

                    # shuju yijing quanbu chulicheng womenxuyaode list le
                    # xianzai xuyao anzhenlaifenpei action for trajectory

                    # action_gt_list1 = []
                    action_gt_list = [[]] * frame_count
                    unique = {}
                    for i in range(frame_count):
                        unique[i] = []
                    for idx, relation in enumerate(relation_instances_list):
                        sub_id = relation[0]
                        obj_id = relation[1]
                        action_name = relation[2]
                        begin = relation[3]
                        end = relation[4]
                        for i in range(begin, end):
                            assert isinstance(action_name, object)
                            """v1"""
                            # test = action_gt_list[i]
                            # test.append([sub_id, obj_id, action_name])
                            # action_gt_list[i].append([sub_id, obj_id, action_name])
                            """v2"""
                            # action_gt_list[i] = [sub_id, obj_id, action_name]
                            """v3"""
                            unique[i].append([sub_id, obj_id, action_name])
                    # print("action_gt_list:\n", action_gt_list)
                    # print("unique:\n", unique)

                    for idx, sub_obj_label in enumerate(sub_obj):
                        tid = sub_obj_label['tid']
                        category = sub_obj_label['category']
                        if category == 'adult' or category == 'baby' or category == 'child':
                            category = 'person'
                        sub_obj_list.append([tid, category])
                        sub_obj_dict[tid] = category

                    # print("sub_obj_list:\n", sub_obj_list)

                    for idx, curr_trajectories in enumerate(trajectories):
                        frame_name = f'{video_id}_' + str(idx + 1).zfill(6) + '.jpg'
                        frame_name_list.append(frame_name)
                        mid = []
                        boxex_h = []
                        boxex_o = []
                        size_list.append([width, height])
                        # if not unique[idx]:
                        #     continue
                        for m, curr_act_dict in enumerate(unique[idx]):
                            curr_triplet = curr_act_dict
                            sub_tid = curr_triplet[0]
                            obj_tid = curr_triplet[1]
                            act_name = curr_triplet[2]

                            for n, trajectories_label in enumerate(curr_trajectories):
                                tid = trajectories_label['tid']
                                bbox = trajectories_label['bbox']
                                xmin = bbox['xmin']
                                ymin = bbox['ymin']
                                xmax = bbox['xmax']
                                ymax = bbox['ymax']
                                bbox4 = [xmin, ymin, xmax, ymax]
                                mid.append([tid, bbox4])
                                if sub_tid == tid:
                                    boxex_h.append(bbox4)
                                elif obj_tid == tid:
                                    boxex_o.append(bbox4)

                            # # xuyao chazhao tid duiying de category name lai queren shi human haishi object
                            # # nawoznegjiayige map queren tid shifouwei person
                            # # haoxiang jiushi yong dict jiuxing
                            # for j, curr_sub_obj in enumerate(sub_obj_list):
                            #     if curr_sub_obj[0] == tid and curr_action_gt0 != []:
                            #         # queren dangqian de tid
                            #         # 1. human or not
                            #         # 2. sub or obj
                            #
                            #         # 1
                            #         # if curr_sub_obj[1] == 'person':
                            #         #     boxex_h.append(bbox4)
                            #         # else:
                            #         #     boxex_o.append(bbox4)
                            #
                            #         # 2
                            #         if curr_action_gt0[0] == tid:
                            #             boxex_h.append(bbox4)
                            #         elif curr_action_gt0[1] == tid:
                            #             boxex_o.append(bbox4)
                            # # v2
                            # # for j, curr_sub_obj in enumerate(sub_obj):
                            # #     if curr_sub_obj['tid'] == tid:
                            # #         curr_name = curr_sub_obj['category']
                            # #         if curr_name == 'person':
                            # #             boxex_h.append(bbox4)
                            # #         else:
                            # #             boxex_o.append(bbox4)

                        num_h = len(boxex_h)
                        hoi = []
                        obj = []
                        verb = []

                        if not unique[idx]:
                            # obj = [[0], [0]]*num_h
                            obj = [0]*num_h
                            verb = [0]*num_h
                            hoi = [0]*num_h
                        else:
                            curr_action_gt = unique[idx]
                            """new version for action_gt have multi action"""
                            # 2. for action_gt have multi action
                            for i, action_gt in enumerate(curr_action_gt):
                                action_name = action_gt[2]
                                action_id = pred_to_idx['{}'.format(action_name)]
                                verb.append(action_id)
                                sub_tid = action_gt[0]
                                obj_tid = action_gt[1]
                                sub_name = sub_obj_dict[sub_tid]
                                obj_name = sub_obj_dict[obj_tid]
                                sub_id = obj_to_idx['{}'.format(sub_name)]
                                obj_id = obj_to_idx['{}'.format(obj_name)]
                                obj.append([sub_id, obj_id])
                                hoi_id = int((obj_id + 1) * (action_id + 1) / (1.0 * (78 * 50)) * 557)
                                hoi.append(hoi_id)

                            """old version"""
                            # # 1. for action_gt only one act
                            # action_name = curr_action_gt[2]
                            # action_id = pred_to_idx['{}'.format(action_name)]
                            # verb.append(action_id)
                            # sub_tid = curr_action_gt[0]
                            # obj_tid = curr_action_gt[1]
                            # sub_name = sub_obj_dict[sub_tid]
                            # obj_name = sub_obj_dict[obj_tid]
                            # sub_id = obj_to_idx['{}'.format(sub_name)]
                            # obj_id = obj_to_idx['{}'.format(obj_name)]
                            # obj.append([sub_id, obj_id])  # 1. zhe shi sub obj doujiajinqu   for vidhoi version
                            # # obj.append(sub_id)
                            # # obj.append(obj_id)
                            # # obj.append(obj_id)  # 2. zhe shi moren sub wei human de qingkuang  for hicodet version
                            # hoi_id = int((obj_id + 1) * (action_id + 1) / (1.0 * (78 * 50)) * 557)
                            # hoi.append(hoi_id)

                        res = {
                            "boxes_h": boxex_h,
                            "boxes_o": boxex_o,
                            "hoi": hoi,
                            "object": obj,
                            "verb": verb
                        }
                        anno.append(res)
                        trajectories_list.append(mid)

                    # print("trajectories_list:\n", trajectories_list)
                    # print("frame_name_list:\n", frame_name_list)



                    # jiexialai wo xuyao ba tamen cunchu cheng json wenjian
                    # write json
                    datameta = {
                        "annotation": anno,
                        "video_path": video_path,
                        "frame_name": frame_name_list,
                        "frame_count": frame_count,
                        # "bbox_gt": trajectories_list,
                        # "action_gt": action_gt_list,
                        "size": size_list,
                        "objects_name": objects_name,
                        "verbs_name": verbs_name
                    }
                    output_json = os.path.join(output_dir, f'{video_id}.json')
                    with open(output_json, 'w') as f:
                        json.dump(datameta, f)

    print("total_frames:", total_frames)

