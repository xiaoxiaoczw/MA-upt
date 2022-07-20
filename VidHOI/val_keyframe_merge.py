from cv2 import imwrite, imread
import os
import json
import pickle
from tqdm import tqdm

# main idea is to use this keyframe json file to extract the corresponding image
# and merge them to one dir
# and rewrite the keyframes json file to satisfy the upt ground truth form

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

    input_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-video'
    input_json = 'val_frame_annots.json'
    image_path = os.path.join(input_dir, 'frames')
    output_dir = 'val_keyframes'
    output_path = os.path.join(input_dir, output_dir)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    with open(os.path.join(input_dir, input_json), 'r') as f:
        keyframes = json.load(f)

    anno_list = []
    # if #obj > 1
    bbox_h_list = []
    bbox_o_list = []
    hoi_list = []
    object_list = []
    verb_list = []

    frame_name_list = []
    size_list = []

    for idx, curr_frame in enumerate(tqdm(keyframes)):
        video_folder = curr_frame['video_folder']  # 0010
        video_id = curr_frame['video_id']  # 3359075894
        frame_id = curr_frame['frame_id']  # 000001 but we need 3359075894_000001
        height = curr_frame['height']
        width = curr_frame['width']
        bbox_h = list(curr_frame['person_box'].values())
        bbox_o = list(curr_frame['object_box'].values())
        h_id = curr_frame['person_id']
        o_id = curr_frame['object_id']  # what are these two obj number mean?
        obj_cls = curr_frame['object_class']
        verb_cls = curr_frame['action_class']
        obj_pair = [0, obj_cls]

        """get correspond image path"""
        # form like /home/student-pc/MA/dataset/Vidhoi/validation-video/frames/0010/3359075894/3359075894_000001.jpg
        frame_name = video_id + '_' + frame_id + '.jpg'
        image_file = os.path.join(image_path, video_folder, video_id, frame_name)
        if not os.path.exists(image_file):
            continue
        img = imread(image_file)
        img_out_file = os.path.join(output_path, frame_name)
        imwrite(img_out_file, img)

        """rewrite the keyframe json"""
        anno = {
            "boxes_h": [bbox_h],
            "boxes_o": [bbox_o],
            "hoi": [0],
            "object": [obj_pair],
            "verb": verb_cls
        }
        anno_list.append(anno)
        frame_name_list.append(frame_name)
        size_list.append([width, height])

    datameta = {
        "annotation": anno_list,
        "frame_name": frame_name_list,
        "size": size_list,
        "objects_name": objects_name,
        "verbs_name": verbs_name
    }
    output_json = os.path.join(input_dir, 'val_keyframe_merge.json')
    with open(output_json, 'w') as f:
        json.dump(datameta, f)