"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pickle
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
# import vidhoi
from vidhoi.vidhoi import vidhoi

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys

sys.path.append('detr')
import datasets.transforms as T

import json

# from inference_vidhoi import output_res
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


def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets


class DataFactory(Dataset):
    def __init__(self, name, partition, data_root):
        # defaults here
        # if name not in ['hicodet', 'vcoco']:
        #     raise ValueError("Unknown dataset ", name)
        if name not in ['hicodet', 'vcoco', 'vidhoi']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        #  add in 20220524 for vidhoi eval
        elif name == 'vidhoi':
            self.dataset = vidhoi(
                # root=os.path.join(data_root, 'vidhoi_test_2'),
                # root=os.path.join(data_root, 'validation-video/frames/0001/2793806282'), # 3598080384 2793806282
                # root=os.path.join(data_root, 'validation-video/frames/0005/4564478328'), # 0005/4564478328 work
                root=os.path.join(data_root, 'validation-video/frames/0010/3359075894'),  # 0010/3359075894
                # anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                anno_file=os.path.join(data_root, 'validation-video/frames/0010/3359075894.json'),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                                       ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
            ])
        # if it is not train here
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        # if self.name == 'hicodet' or 'vidhoi':
        if self.name == 'hicodet':  # default
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
            image, target = self.transforms(image, target)  # defaults

        elif self.name == 'vidhoi':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            # target['boxes_h'][:, :2] -= 1
            # target['boxes_o'][:, :2] -= 1

            image, target = self.transforms(image, target)  # defaults

        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            image, target = self.transforms(image, target)  # defaults

        # image, target = self.transforms(image, target)  #defaults

        return image, target


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def test_hico(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # output = net(inputs)  # defaults

            """version for gt replace detector output 0706 """
            # target = pocket.ops.relocate_to_cuda(batch[-1])
            target = batch[-1]
            output = net(inputs, targets=target)
            """end here"""
            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)  # chansheng yige yu scores xiangtong size de all zero tensor
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                         gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                         boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)

        return meter.eval()

    """additional add test vidhoi method 0602"""

    @torch.no_grad()
    def test_vidhoi(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        meter = DetectionAPMeter(
            557, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # output = net(inputs)
            """version for gt replace detector output 0706 """
            # target = pocket.ops.relocate_to_cuda(batch[-1])
            target = batch[-1]
            output = net(inputs, targets=target)
            """end here"""

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)  # chansheng yige yu scores xiangtong size de all zero tensor
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                         gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                         boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)

        return meter.eval()

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num

        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vidhoi(self, dataloader, args, cache_dir='matlab-vidhoi'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction
        actions = dataset.verbs if args.dataset == 'hicodet' or args.dataset == 'vidhoi' else \
            dataset.actions

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((557, nimages), dtype=object)

        boxes_h_list = []
        boxes_o_list = []
        interactions_list = []
        scores_list = []

        bx_h_list = []
        bx_o_list = []
        # bbx_list = []
        # pairing_list = []
        scor_list = []
        det_objects_list = []
        hoi_objects_list = []
        pred_list = []
        file_name_list = []

        action_score_thresh = 0.2

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # output = net(inputs)

            """version for gt replace detector output 0706 """
            # target = pocket.ops.relocate_to_cuda(batch[-1])
            target = batch[-1]
            output = net(inputs, targets=target)

            device = torch.device(args.device)

            image = batch[0][0]
            output_0 = output[0]

            """output_res function """
            # Rescale the boxes to original image size
            rgb, ow, oh = image.size()
            h, w = output_0['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            scale_fct = scale_fct.to(device)
            boxes = output_0['boxes'] * scale_fct
            # Find the number of human and object instances
            nh = len(output_0['pairing'][0].unique())
            no = len(boxes)

            scor = output_0['scores']
            objects = output_0['objects']
            pred = output_0['labels']

            pairing = output_0['pairing']

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
                keep = torch.nonzero(torch.logical_and(scor >= action_score_thresh, pred == verb)).squeeze(1)
                bx_h, bx_o = boxes[pairing].unbind(0)
                curr_bxh = bx_h[keep]
                curr_bxo = bx_o[keep]
                for i in range(len(keep)):  # keep kenengyouduige
                    testpp = bx_h[keep[i], :2]
                    testpp[1] += i * 20
                    if scor[keep[i]].tolist():
                        new_scores.append(scor[keep[i]].tolist())
                    if actions[pred[keep[i]]]:
                        new_action.append(actions[pred[keep[i]]])
                    if OBJECTS[objects[keep[i]]]:
                        new_hoi_objects.append(OBJECTS[objects[keep[i]]])
                    if curr_bxh[i].tolist():
                        new_hoi_bbox_h.append(curr_bxh[i].tolist())
                    if curr_bxo[i].tolist():
                        new_hoi_bbox_o.append(curr_bxo[i].tolist())

            bbx_h, bbx_o, scor, hoi_objects, pred = new_hoi_bbox_h, new_hoi_bbox_o, new_scores, new_hoi_objects, new_action
            # bbx_h, bbx_o, scores, hoi_objects, pred =
            #     inference_vidhoi.output_res(
            #     image, output[0], actions, device, args.action,
            #     args.action_score_thresh)

            bx_h_list.append(bbx_h)
            bx_o_list.append(bbx_o)
            scor_list.append(scor)
            # det_objects_list.append(det_objects)
            hoi_objects_list.append(hoi_objects)
            pred_list.append(pred)
            # file_name_list.append(file_name)

            """end here"""

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            """store result as json"""
            boxes_h_list.append(boxes_h.tolist())
            boxes_o_list.append(boxes_o.tolist())
            interactions_list.append(interactions.tolist())
            scores_list.append(scores.tolist())
            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num

        """v1 save as json"""
        datameta = {
            "boxes_h": bx_h_list,
            "boxes_o": bx_o_list,
            "scores": scor_list,
            "hoi_objects": hoi_objects_list,
            "pred": pred
        }
        output_json = os.path.join(cache_dir, f'{cache_dir}.json')
        with open(output_json, 'w') as f:
            json.dump(datameta, f)

        """v2 save as json"""
        detameta_2 = {
            "boxes_h": boxes_h_list,
            "boxes_o": boxes_o_list,
            "interactions": interactions_list,
            "scores": scores_list
        }
        output_json_2 = os.path.join(cache_dir, f'{cache_dir}_2.json')
        with open(output_json_2, 'w') as f:
            json.dump(detameta_2, f)

        # Replace None with size (0,0) arrays
        for i in range(557):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        # print("all_result:\n", all_results)
        for object_idx in range(78):
            interaction_idx = object2int[object_idx]
            """change below"""
            curr_result = all_results[interaction_idx]
            curr_result_list = []
            for i, res in enumerate(curr_result):
                mid_list = []
                for j, mid in enumerate(res):
                    mid_list.append(mid.tolist())
                # print("res:\n", res)
                curr_result_list.append(mid_list)
            # print("curr_result_list:\n", curr_result_list)
            print(object_idx + 1)
            # print("curr_result:\n", all_results[interaction_idx])
            output_json = os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.json')
            with open(output_json, 'w') as f:
                json.dump(curr_result_list, f)

            """end here"""
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
