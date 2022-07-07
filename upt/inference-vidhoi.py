"""
Visualise detected human-object interactions in an image
Fred Zhang <frederic.zhang@anu.edu.au>
The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pocket
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import DataFactory
from upt import build_detector

import json
from tqdm import tqdm

"""for test to import thop, ptflops and pytorch_model_summary"""
from thop import profile
from ptflops import get_model_complexity_info
from pytorch_model_summary import summary

warnings.filterwarnings("ignore")

OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def draw_hoi(ax, boxes, unique_actions, scores, thresh, pred, image, pairing, actions, objects):
    for verb in unique_actions:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == verb)).squeeze(1)
        bx_h, bx_o = boxes[pairing].unbind(0)
        curr_bxh = bx_h[keep]
        curr_bxo = bx_o[keep]
        # curr_score = scores[keep]
        # curr_act = actions[pred[keep]]
        # curr_obj = OBJECTS[objects[keep]]
        # pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        pocket.utils.draw_box_pairs(image, curr_bxh, curr_bxo, width=5)
        plt.imshow(image)
        plt.axis('off')
        for i in range(len(keep)):
            testpp = bx_h[keep[i], :2]
            testpp[1] += i * 20

            txt = plt.text(*testpp, f"{scores[keep[i]]:.4f}, {actions[pred[keep[i]]]} {OBJECTS[objects[keep[i]]]}",
                           fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()


def draw_objects(ax, boxes, pairing, objects):
    bx_h, bx_o = boxes[pairing].unbind(0)
    # 这一条我可以得知每一个oh pair 分别对应的bh和bo的位置，然后我们也可以知道这个位置对应的o和h的名称，接下来要做的就是
    # 找出unique的名称，然后遍历所有的名称，输出相应的bounding box 但是这个意味着如果有多个person 或许只能输出一个？
    # keep = torch.nonzero(pred == action).squeeze(1)
    unique_objects = torch.unique(objects)
    for obj in unique_objects:
        sample_idx = torch.nonzero(objects == obj).squeeze(1)
        # 找出当前obj的所有位置，都画上bx，这样重复也无所谓了 只要有bx就行 只是可能compute cost 比较大
        for idx in sample_idx:
            # curr_bx1 = bx_h[idx]
            curr_bx2 = bx_o[idx]
            # xy1 = curr_bx1[:2]
            xy2 = curr_bx2[:2]
            # h1, w1 = (curr_bx1[2:] - curr_bx1[:2]).tolist()
            h2, w2 = (curr_bx2[2:] - curr_bx2[:2]).tolist()
            # patch1 = patches.Rectangle(xy1.tolist(), h1, w1, facecolor='none', edgecolor='w')
            patch2 = patches.Rectangle(xy2.tolist(), h2, w2, facecolor='none', edgecolor='w')
            # ax.add_patch(patch1)
            ax.add_patch(patch2)
            # txt = plt.text(*xy1.tolist(), f"{OBJECTS[objects[idx]]}", fontsize=20, fontweight='semibold', color='w')
            txt = plt.text(*xy2.tolist(), f"{OBJECTS[objects[idx]]}", fontsize=20, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
            # break # not work yet


def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i + 1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()


def visualise_entire_image(image, output, actions, action=None, thresh=0.2):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique())
    no = len(boxes)
    # pairing 是2维的 上面这句表明[0]记录human的idx 而obj的数量是由box个数来表示的，难道这其中不会有human的box吗？

    scores = output['scores']
    # scores 是表示每一个human object pair的分数吗？
    objects = output['objects']
    # objects 维度同上 表示对应位置上的object idx 但是这个对应位置表示啥？
    pred = output['labels']
    # pred 维度同上，表示对应位置上的verb 从此处猜测对应位置上应该是human object pair

    # Visualise detected human-object pairs with attached scores
    # test --action 36
    if action is not None:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        # 此时bx-h 和 bx-o 的dim 和pairing 的 dim 相同 ，相当于是把box扩展到每个pair了 然后再根据idx选择
        # 那我可不可以认为他就是这样选pair的 首先得到所有可能的pair 然后根据 thres 来选
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')

        for i in range(len(keep)):
            # default
            # txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')

            # v-1
            # txt = plt.text(*bx_h[keep[i], :2], f"score: {scores[keep[i]]:.4f}, object: {OBJECTS[objects[keep[i]]]}.",
            #                fontsize=15, fontweight='semibold', color='w')

            # v-2 注意这些奇怪八绕的索引 {actions[pred[keep[i]]]} {OBJECTS[objects[keep[i]]]}
            txt = plt.text(*bx_h[keep[i], :2],
                           f"{scores[keep[i]]:.4f}, {actions[pred[keep[i]]]} {OBJECTS[objects[keep[i]]]}",
                           fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        plt.savefig("./cooktest/results/curr_output.jpg")
        plt.show()
        return

    pairing = output['pairing']
    coop_attn = output['attn_maps'][0]
    comp_attn = output['attn_maps'][1]

    # Visualise attention from the cooperative layer
    for i, attn_1 in enumerate(coop_attn):
        fig, axe = plt.subplots(2, 4)
        fig.suptitle(f"Attention in coop. layer {i}")
        axe = np.concatenate(axe)
        ticks = list(range(attn_1[0].shape[0]))
        labels = [v + 1 for v in ticks]
        for ax, attn in zip(axe, attn_1):
            im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)

    x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    x, y = torch.nonzero(x != y).unbind(1)
    pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]

    # Visualise attention from the competitive layer
    fig, axe = plt.subplots(2, 4)
    fig.suptitle("Attention in comp. layer")
    axe = np.concatenate(axe)
    ticks = list(range(len(pairs)))
    for ax, attn in zip(axe, comp_attn):
        im = ax.imshow(attn, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(pairs, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(pairs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    # Print predicted actions and corresponding scores
    unique_actions = torch.unique(pred)  # 找到不重复的动作
    for verb in unique_actions:
        print(f"\n=> Action: {actions[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)  # torch.nonzero 找到tensor中所有不为0的值的索引/下标
        # 也就是找到pred中属于当前verb的所有位置
        # squeeze 从数组的形状中删除单维度条目，即把shape中为1的维度去掉，一个reshape操作
        # 遍历所有下标 找到对应的pair，score以及object idx是1维的 注意
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx] + 1
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                # f"score: {scores[idx]:.4f}"
                f"score: {scores[idx]:.4f}, object: {OBJECTS[objects[idx]]}."
            )

    # Draw the bounding boxes
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    # draw_boxes(ax, boxes) # default
    # draw_objects(ax, boxes, pairing, objects)
    draw_hoi(ax, boxes, unique_actions, scores, thresh, pred, image, pairing, actions, objects)
    plt.savefig("./cooktest/results/curr_output.jpg")
    plt.show()


def output_res(image, output, actions, dev, action=None, thresh=0.2):
    """output result parameters by json file in the whole image"""
    res = []
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    scale_fct = scale_fct.to(dev)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique())
    no = len(boxes)

    scores = output['scores']
    objects = output['objects']
    pred = output['labels']

    # Visualise detected human-object pairs with attached scores
    # test --action 36

    pairing = output['pairing']
    coop_attn = output['attn_maps'][0]
    comp_attn = output['attn_maps'][1]

    x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    x, y = torch.nonzero(x != y).unbind(1)
    pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]

    new_scores = []
    new_hoi_objects = []
    new_det_objects = []
    new_det_bbox = []
    new_hoi_bbox_o = []
    new_hoi_bbox_h = []
    new_action = []
    # Print predicted actions and corresponding scores
    unique_actions = torch.unique(pred)
    """draw object"""
    # about unique need more concern
    # TODO
    bx_h, bx_o = boxes[pairing].unbind(0)
    unique_objects = torch.unique(objects)
    for obj in unique_objects:
        sample_idx = torch.nonzero(objects == obj).squeeze(1)
        for idx in sample_idx:
            # curr_bx1 = bx_h[idx]
            curr_bx2 = bx_o[idx]
            new_det_bbox.append(curr_bx2.tolist())
            new_det_objects.append(OBJECTS[objects[idx]])
            break  # not work so good
        # new_det_bbox.append(mid1)
        # new_det_objects.append(mid2)

    for verb in unique_actions:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == verb)).squeeze(1)
        bx_h, bx_o = boxes[pairing].unbind(0)
        curr_bxh = bx_h[keep]
        curr_bxo = bx_o[keep]
        # mid1 = []
        # mid2 = []
        # mid3 = []
        # mid4 = []
        # mid5 = []
        # for i in range(len(keep)):  # keep kenengyouduige
        #     testpp = bx_h[keep[i], :2]
        #     testpp[1] += i * 20
        #     mid1.append(scores[keep[i]].tolist())
        #     mid2.append(actions[pred[keep[i]]])
        #     mid3.append(OBJECTS[objects[keep[i]]])
        #     mid4.append(curr_bxh[i].tolist())
        #     mid5.append(curr_bxo[i].tolist())
        # if mid1:
        #     new_scores.append(mid1)
        # if mid2:
        #     new_action.append(mid2)
        # if mid3:
        #     new_hoi_objects.append(mid3)
        # if mid4:
        #     new_hoi_bbox_h.append(mid4)
        # if mid5:
        #     new_hoi_bbox_o.append(mid5)
        for i in range(len(keep)):  # keep kenengyouduige
            testpp = bx_h[keep[i], :2]
            testpp[1] += i * 20
            if scores[keep[i]].tolist():
                new_scores.append(scores[keep[i]].tolist())
            if actions[pred[keep[i]]]:
                new_action.append(actions[pred[keep[i]]])
            if OBJECTS[objects[keep[i]]]:
                new_hoi_objects.append(OBJECTS[objects[keep[i]]])
            if curr_bxh[i].tolist():
                new_hoi_bbox_h.append(curr_bxh[i].tolist())
            if curr_bxo[i].tolist():
                new_hoi_bbox_o.append(curr_bxo[i].tolist())

    return new_hoi_bbox_h, new_hoi_bbox_o, new_scores, new_hoi_objects, new_action


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root)
    # dataset =  dataset.to(device)
    # hicodet exist a relate between object to verb ? conversion dim = 80 means each object has some related actions ?
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' or args.dataset == 'vidhoi'\
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' or args.dataset == 'vidhoi' else 24
    actions = dataset.dataset.verbs if args.dataset == 'hicodet' or args.dataset == 'vidhoi' else \
        dataset.dataset.actions

    upt = build_detector(args, conversion)
    if torch.cuda.is_available():
        upt.to(device)  # model try to use gpu
    upt.eval()

    """test about model size and flops"""
    # print(upt)  # my test here
    # dummy_input = torch.randn(1, 3, 128, 128)
    dummy_input = torch.randn(1, 3, 720, 1280)

    "thop"
    # flops, params = profile(upt, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    "ptflops report error"
    # flops, params = get_model_complexity_info(upt, (3, 128, 128), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    "pytorch_model_summary"
    # print(summary(upt, dummy_input, show_input=False, show_hierarchical=False))

    "direct"
    Trainable_params = sum(p.numel() for p in upt.parameters() if p.requires_grad)
    print('Trainable_params: ', Trainable_params)
    print('Trainable_params: %.2f M' % (Trainable_params / 1000000.0))

    """test end here"""

    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        # checkpoint = torch.load(args.resume, map_location='cpu')  # defaults
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
        # upt.to(device)
    else:
        print(f"=> Start from a randomly initialised model")

    if args.image_path is None:
        image, _ = dataset[args.index]
        if torch.cuda.is_available():
            image = image.to(device)
        output = upt([image])
        image = dataset.dataset.load_image(
            os.path.join(dataset.dataset._root,
                         dataset.dataset.filename(args.index)
                         ))
        visualise_entire_image(image, output[0], actions, args.action, args.action_score_thresh)

    else:
        """change below for one video"""
        if os.path.isdir(args.image_path):
            output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_4inference'
            # input_dir = '/home/student-pc/MA/dataset/Vidhoi/validation-video/frames'
            root = os.walk(args.image_path)
            """1. for each video"""
            # for path, dir_list, file_list in root:
            #     for dir_name in dir_list:  # 0010
            #         # dir_name.sort()
            #         root_2 = os.walk(os.path.join(path, dir_name))
            #         output_dir = os.path.join(output_path, dir_name)
            #         if not os.path.isdir(output_dir):
            #             os.makedirs(output_dir)
            #         bx_h_list = []
            #         bx_o_list = []
            #         # bbx_list = []
            #         # pairing_list = []
            #         scores_list = []
            #         det_objects_list = []
            #         hoi_objects_list = []
            #         pred_list = []
            #         file_name_list = []
            #         for path_2, dir_list_2, file_list_2 in root_2:  # 3359075894
            #             for dir_name_2 in dir_list_2:
            #                 root_3 = os.walk(os.path.join(path_2, dir_name_2))
            #                 video_path = os.path.join(path_2, dir_name_2)
            #                 print("video_path:", video_path)
            #                 output_json = os.path.join(output_dir, f'{dir_name_2}.json')
            #                 print("output_json:", output_json)
            #                 if os.path.exists(output_json):
            #                     if output_json != '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_4inference/0010/3359075894.json':
            #                         print("this video has already done ")
            #                         continue
            #                 for path_3, dir_list_3, file_list_3 in root_3:  # 3359075894_000001
            #                     file_list_3.sort()
            #                     for file_name_3 in tqdm(file_list_3, colour='CYAN'):
            #                         # output_dir = os.path.join(output_dir, file_name_3)
            #                         # if not os.path.isdir(output_dir):
            #                         #     os.makedirs(output_dir)
            #                         # print("file_name:", os.path.join(path_3, file_name_3))
            #                         image_path = os.path.join(path_3, file_name_3)
            #                         image = dataset.dataset.load_image(image_path)
            #                         image_tensor, _ = dataset.transforms(image, None)
            #                         if torch.cuda.is_available():
            #                             image_tensor = image_tensor.to(device)  # data try to use gpu
            #                         output = upt([image_tensor])
            #                         # if torch.cuda.is_available():
            #                         #     output = output.to(device)
            #                         if not output:
            #                             bbx_h, bbx_o, scores, hoi_objects, pred = [], [], [], [], []
            #                         else:
            #                             bbx_h, bbx_o, scores, hoi_objects, pred = output_res(image, output[0], actions, device, args.action,
            #                                                                                 args.action_score_thresh)
            #                         bx_h_list.append(bbx_h)
            #                         bx_o_list.append(bbx_o)
            #                         scores_list.append(scores)
            #                         # det_objects_list.append(det_objects)
            #                         hoi_objects_list.append(hoi_objects)
            #                         pred_list.append(pred)
            #                         file_name_list.append(file_name_3)
            #                         # break
            #
            #                     datameta = {
            #                         "bbox_h": bx_h_list,
            #                         "bbox_o": bx_o_list,
            #                         # "det_objects": det_objects_list,
            #                         "scores": scores_list,
            #                         "hoi_objects": hoi_objects_list,
            #                         "pred": pred_list,
            #                         "file_name": file_name_list
            #                     }
            #                     # output_json = os.path.join(output_dir, f'{file_name_3}.json')
            #                     with open(output_json, 'w') as f:
            #                         json.dump(datameta, f)

            """2. only for one video"""
            # output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_4inference_testgt'
            #
            # video_path = args.image_path[-15:]  # 0010/3359075894
            # output_dir = os.path.join(output_path, video_path[:4])
            # video_id = video_path[5:]
            # if not os.path.isdir(output_dir):
            #     os.makedirs(output_dir)
            # root = os.walk(args.image_path)
            # bx_h_list = []
            # bx_o_list = []
            # # bbx_list = []
            # # pairing_list = []
            # scores_list = []
            # det_objects_list = []
            # hoi_objects_list = []
            # pred_list = []
            # file_name_list = []
            #
            # for path, dir_list, file_list in root:
            #     file_list.sort()
            #     for file_name in tqdm(file_list, colour='CYAN'):
            #         image_path = os.path.join(path, file_name)
            #         image = dataset.dataset.load_image(image_path)
            #         image_tensor, _ = dataset.transforms(image, None)
            #         if torch.cuda.is_available():
            #             image_tensor = image_tensor.to(device)  # data try to use gpu
            #         output = upt([image_tensor])
            #         bbx_h, bbx_o, scores, hoi_objects, pred = output_res(image, output[0], actions, device, args.action,
            #                                                              args.action_score_thresh)
            #         bx_h_list.append(bbx_h)
            #         bx_o_list.append(bbx_o)
            #         scores_list.append(scores)
            #         # det_objects_list.append(det_objects)
            #         hoi_objects_list.append(hoi_objects)
            #         pred_list.append(pred)
            #         file_name_list.append(file_name)
            #         # break
            #
            # datameta = {
            #     "bbox_h": bx_h_list,
            #     "bbox_o": bx_o_list,
            #     # "det_objects": det_objects_list,
            #     "scores": scores_list,
            #     "hoi_objects": hoi_objects_list,
            #     "pred": pred_list,
            #     "file_name": file_name_list
            # }
            # output_json = os.path.join(output_dir, f'{video_id}.json')
            # with open(output_json, 'w') as f:
            #     json.dump(datameta, f)

            """3. test for replace gt"""
            video_path = args.image_path[-15:]  # 0010/3359075894

            output_path = '/home/student-pc/MA/dataset/Vidhoi/validation_frame_anno_4inference_testgt'
            gt_path = args.image_path + '.json'
            with open(gt_path, 'r') as f:
                all_frames_target = json.load(f)

            output_dir = os.path.join(output_path, video_path[:4])
            video_id = video_path[5:]
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            root = os.walk(args.image_path)
            bx_h_list = []
            bx_o_list = []
            # bbx_list = []
            # pairing_list = []
            scores_list = []
            det_objects_list = []
            hoi_objects_list = []
            pred_list = []
            file_name_list = []

            for path, dir_list, file_list in root:
                file_list.sort()
                for idx, file_name in enumerate(tqdm(file_list, colour='CYAN')):
                    if idx > 408:  # cause in 0010/3359075894 video only 408 valid gt
                        break
                    image_path = os.path.join(path, file_name)
                    image = dataset.dataset.load_image(image_path) #default
                    # image_tensor, _ = dataset.transforms(image, None)
                    image_tensor, target_tensor = dataset[idx]
                    # if target_tensor == torch.Size([]):

                    # image_tensor, target_tensor = dataset.transforms(image, target)
                    if torch.cuda.is_available():
                        image_tensor = image_tensor.to(device)  # data try to use gpu
                        # target = target.to(device)  # data try to use gpu
                    output = upt([image_tensor], [target_tensor])

                    """add target /gt here"""
                    # _, target = dataset[idx]
                    # target = all_frames_target['annotation'][idx]
                    # target = torch.Tensor(target)
                    # target = pocket.ops.relocate_to_cuda(target)
                    # output = upt([image_tensor], [target])
                    """end here"""

                    bbx_h, bbx_o, scores, hoi_objects, pred = output_res(image, output[0], actions, device, args.action,
                                                   args.action_score_thresh)
                    bx_h_list.append(bbx_h)
                    bx_o_list.append(bbx_o)
                    scores_list.append(scores)
                    # det_objects_list.append(det_objects)
                    hoi_objects_list.append(hoi_objects)
                    pred_list.append(pred)
                    file_name_list.append(file_name)
                    # break

            datameta = {
                "bbox_h": bx_h_list,
                "bbox_o": bx_o_list,
                # "det_objects": det_objects_list,
                "scores": scores_list,
                "hoi_objects": hoi_objects_list,
                "pred": pred_list,
                "file_name": file_name_list
            }
            output_json = os.path.join(output_dir, f'{video_id}.json')
            with open(output_json, 'w') as f:
                json.dump(datameta, f)
        else:
            image = dataset.dataset.load_image(args.image_path)
            image_tensor, _ = dataset.transforms(image, None)
            output = upt([image_tensor])
            visualise_entire_image(image, output[0], actions, args.action, args.action_score_thresh)

    # visualise_entire_image(image, output[0], actions, args.action, args.action_score_thresh)  # default


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--human-idx', type=int, default=0)

    # parser.add_argument('--device', default='cpu')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
                        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
                        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
                        help="Path to an image file.")

    args = parser.parse_args()

    main(args)

    # print("this code end here")
