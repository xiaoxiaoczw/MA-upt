from cv2 import imwrite, imread
import os
import json
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# main idea is to split the whole dataset into train and test dataset

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
    input_json = 'val_keyframe_merge_new.json'
    image_path = os.path.join(input_dir, 'val_keyframes')
    output_dir = 'val_keyframes_split'
    output_path = os.path.join(input_dir, output_dir)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    with open(os.path.join(input_dir, input_json), 'r') as f:
        keyframes = json.load(f)
    anno_set = keyframes['annotation']
    frame_name_set = keyframes['frame_name']
    size_set = keyframes['size']

    """add train-test split"""
    val_split = 0.25
    train_idx, val_idx = train_test_split(list(range(len(anno_set))), test_size=val_split)
    output_path_train = os.path.join(output_path, 'train')
    output_path_test = os.path.join(output_path, 'test')
    if not os.path.isdir(output_path_train):
        os.makedirs(output_path_train)
    if not os.path.isdir(output_path_test):
        os.makedirs(output_path_test)

    anno_list = []
    train_anno_list, test_anno_list = [], []
    # if #obj > 1
    bbox_h_list = []
    bbox_o_list = []
    hoi_list = []
    object_list = []
    verb_list = []

    frame_name_list = []
    size_list = []
    train_frame_name_list, test_frame_name_list = [], []
    train_size_list, test_size_list = [], []

    for idx, anno in enumerate(tqdm(anno_set)):
        frame_name = frame_name_set[idx]
        size = size_set[idx]
        """get correspond image path"""
        # form like /home/student-pc/MA/dataset/Vidhoi/validation-video/val_keyframes/3359075894_000001.jpg

        image_file = os.path.join(image_path, frame_name)
        if not os.path.exists(image_file):
            continue
        img = imread(image_file)
        img_out_file = os.path.join(output_path, frame_name)
        # store images for all
        # imwrite(img_out_file, img)

        if idx in train_idx:
            train_img_out_file = os.path.join(output_path_train, frame_name)
            imwrite(train_img_out_file, img)
            train_anno_list.append(anno)
            train_frame_name_list.append(frame_name)
            train_size_list.append(size)
        elif idx in val_idx:
            test_img_out_file = os.path.join(output_path_test, frame_name)
            imwrite(test_img_out_file, img)
            test_anno_list.append(anno)
            test_frame_name_list.append(frame_name)
            test_size_list.append(size)

    datameta = {
        "annotation": anno_list,
        "frame_name": frame_name_list,
        "size": size_list,
        "objects_name": objects_name,
        "verbs_name": verbs_name
    }
    train_datameta = {
        "annotation": train_anno_list,
        "frame_name": train_frame_name_list,
        "size": train_size_list,
        "objects_name": objects_name,
        "verbs_name": verbs_name
    }
    test_datameta = {
        "annotation": test_anno_list,
        "frame_name": test_frame_name_list,
        "size": test_size_list,
        "objects_name": objects_name,
        "verbs_name": verbs_name
    }
    # store json file for all
    # output_json = os.path.join(input_dir, 'val_keyframe_merge_new.json')
    # with open(output_json, 'w') as f:
    #     json.dump(datameta, f)
    # save train-test split json result
    train_output_json = os.path.join(input_dir, 'val_keyframe_train.json')
    with open(train_output_json, 'w') as f:
        json.dump(train_datameta, f)
    test_output_json = os.path.join(input_dir, 'val_keyframe_test.json')
    with open(test_output_json, 'w') as f:
        json.dump(test_datameta, f)