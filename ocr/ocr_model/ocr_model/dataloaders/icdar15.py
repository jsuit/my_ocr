from collections import defaultdict
import enum
from typing import Dict, List
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from cv2 import (
    cvtColor,
    imread as cv2_imread,
    fillPoly as cv2_fillPoly,
    IMREAD_COLOR as cv2_IMREAD_COLOR,
    boundingRect as cv2_bbox,
    minAreaRect as cv2_min_bbox,
    boxPoints as cv2_box_points,
    COLOR_BGR2RGB,
    INTER_CUBIC,
)
from shapely.geometry import Polygon, MultiPoint, LineString
import pyclipper
import hydra
from tokenizers import Tokenizer, Encoding as SentenceEncoding


def batch_boxes(batch_size: int, torch_bboxes: list[torch.Tensor], max_len: int):
    size = [batch_size, max_len]
    for dim in range(1, torch_bboxes[0].ndim):
        size.append(torch_bboxes[0].shape[dim])
    boxes = torch.zeros(*size, dtype=torch.float32)
    if not boxes.numel():
        return torch.zeros(batch_size, 1, 4)

    for tensor_box, torch_bbox in zip(boxes, torch_bboxes):
        tensor_box[: torch_bbox.shape[0], :] = torch_bbox

    return boxes


def batch_text(text_encoding: List[List[SentenceEncoding]]):

    bs = len(text_encoding)
    max_num_sents = max([len(sentence) for sentence in text_encoding])
    max_len = max([len(sentences[0]) for sentences in text_encoding])
    text_ids = torch.zeros(bs, max_num_sents, max_len, dtype=torch.long)
    attention_mask = torch.zeros_like(text_ids)
    for i, sentences in enumerate(text_encoding):
        batch_ids = torch.Tensor([sentence.ids for sentence in sentences])

        text_ids[i, : batch_ids.shape[0], : batch_ids.shape[1]] = batch_ids
        batch_attn_mask = torch.Tensor(
            [sentence.attention_mask for sentence in sentences]
        )
        attention_mask[
            i, : batch_attn_mask.shape[0], : batch_attn_mask.shape[1]
        ] = batch_attn_mask
    return text_ids, attention_mask


class ICDARDataset(Dataset):
    def __init__(self, config, train: bool = True) -> None:
        super().__init__()
        stage_str = "train" if train else "test"
        self._images = os.path.join(
            config.data_dir, config.data.dataset_name, f"{stage_str}_images"
        )
        self._num_polygons_per_image = 4
        self._gts = os.path.join(
            config.data_dir, config.data.dataset_name, f"{stage_str}_gts"
        )
        self._shrink_ratio = config.data.shrink_ratio
        self._thresh_min, self._thresh_max = (
            config.data.min_thresh,
            config.data.max_thresh,
        )
        self._config = config
        self._train = train

        self._transforms = (
            self.train_transforms() if train else self.val_transforms(viz=False)
        )
        self._do_viz = config.viz
        if not isinstance(config.viz, (bool, int)):
            raise TypeError(
                f"visualization flag {config.viz} type {type(config.viz).__name__} is not an int or bool"
            )

        if self._do_viz:
            self._viz_transforms = self.val_transforms(viz=True)

        self._use_text = self._config
        self._tokenizer = (
            Tokenizer.from_pretrained(self._config.tokenization)
            if self._config.data.with_text
            else None
        )
        if self._tokenizer:
            self._tokenizer.enable_padding(pad_id=0)

        def dir_exists(dir_):
            if not os.path.exists(dir_) or not os.path.isdir(dir_):
                raise ValueError(
                    f"tried to find directory {dir_} but it either does not exist or is not a directory"
                )

        [
            dir_exists(d)
            for d in [
                self._images,
                self._gts,
            ]
        ]
        self.setup()
        self._labels = self.load_anns()

    def setup(self):
        self._image_paths = [
            os.path.join(self._images, image_file)
            for image_file in os.listdir(self._images)
        ]
        self._gts_pth = [
            os.path.join(self._gts, train_gt) for train_gt in os.listdir(self._gts)
        ]

    def load_anns(self):
        gt_paths = self._gts_pth
        labels = {}

        for gt in gt_paths:
            with open(gt, "r") as f:
                readlines = f.readlines()
            lines = []
            for line in readlines:
                parts = line.strip().split(",")
                line = [i.strip("\ufeff").strip("\xef\xbb\xbf") for i in parts]
                label = line[-1]
                if not label:
                    label = "###"
                polygons = np.array(
                    list(map(float, line[:8])), dtype=np.float32
                ).reshape((-1, 2))

                lines.append({"poly": polygons, "text": label.lower()})
            img_name = gt.split("/")[-1][:-4]
            if img_name in labels:
                raise KeyError(f"name of image {img_name} is already been seen")
            labels[img_name] = lines
        return labels

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index: int):
        img_path = self._image_paths[index]

        img_name = img_path.split("/")[-1]

        if not self._train:
            img_name = f"gt_{img_name[:-4]}"

        labels = self._labels[img_name]
        img = (
            cvtColor(cv2_imread(img_path, cv2_IMREAD_COLOR), COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255.0
        )  #
        cv_h, cv_w = img.shape[0], img.shape[1]
        all_polys_list = [
            self._fit_polygon_to_image(label["poly"], img_h=cv_h, img_w=cv_w)
            for label in labels
        ]

        text_list = [text["text"] for text in labels]
        expanded_text = []

        bboxs, to_delete = self._get_bboxs(all_polys_list)

        if to_delete:
            text_list, bboxs, all_polys_list = self._filter_data(
                delete_set=to_delete,
                text=text_list,
                bboxs=bboxs,
                polygons=all_polys_list,
            )
        n_poly_coords = [len(poly) for poly in all_polys_list]
        for i, text in enumerate(text_list):
            expanded_text.extend([text] * n_poly_coords[i])
        all_polygons_np = np.concatenate(all_polys_list, axis=0)
        bboxs_np = np.stack(bboxs).astype(np.float32)
        transformed = self._transforms(
            image=img,
            keypoints=all_polygons_np,
            text=expanded_text,
            bboxes=bboxs_np,
            bbox_text=text_list,
        )

        transformed_img = transformed["image"]
        polygons = transformed["keypoints"]
        transformed_text = transformed["text"]

        text = []
        all_polys_list = []
        i = 0

        polygons = np.array(polygons, dtype=np.float32)

        for i in range(0, len(polygons), self._num_polygons_per_image):
            # instead of one big vector, break it up into lists, where len of list are
            # the number of polygons in image
            all_polys_list.append(polygons[i : i + self._num_polygons_per_image])
            text.append(transformed_text[i])
        # gt_image : has polygons, like a mask, to predict
        # mask: bad polygons
        # min_box_points: min bounding box points
        # bboxs: list of all bboxs (not min) (xywh)
        h, w = transformed_img.shape[-2:]
        (gt_img, mask, polygons, min_box_points, bboxs, text_list,) = self._seg_mask(
            h=h,
            w=w,
            textlist=text,
            polygonlist=all_polys_list,
        )

        thresh_map, thresh_mask = self.threshold_map(
            polygons=polygons, h=h, w=w, shrink=False
        )
        bboxs[:, 2] = bboxs[:, 0] + bboxs[:, 2]
        bboxs[:, 3] = bboxs[:, 1] + bboxs[:, 3]
        deltas, center_scores = self.lrtb_map(bboxes=bboxs, img_h=h, img_w=w)

        data = dict(
            image=transformed_img,
            gt_img=torch.from_numpy(gt_img),
            gt_mask=torch.from_numpy(mask),
            rotated_bboxes_points=torch.from_numpy(min_box_points),
            bboxes=torch.from_numpy(bboxs),
            thresh_map=torch.from_numpy(thresh_map),
            thresh_mask=torch.from_numpy(thresh_mask),
            # text=text_list,
            distance_deltas=torch.from_numpy(
                np.stack(deltas, axis=0).astype(np.float32)
            ),
            centerness=torch.from_numpy(center_scores.astype(np.float32)),
        )
        if self._do_viz:
            data.update({"viz_image": self._viz_transforms(img)})
        return data

    def _filter_data(self, delete_set, text, bboxs, polygons):

        data = [
            (t, b, p)
            for i, (t, b, p) in enumerate(zip(text, bboxs, polygons))
            if i not in delete_set
        ]
        return list(zip(*data))

    def _fit_polygon_to_image(self, polygons, img_h, img_w):
        if not len(polygons):
            return polygons
        polygons[:, 0] = np.clip(polygons[:, 0], 0, img_w - 1)
        polygons[:, 1] = np.clip(polygons[:, 1], 0, img_h - 1)
        return polygons

    def _get_bboxs(self, polygons: List[np.ndarray]):
        min_text_size = self._config.data.min_text_size
        bboxs = []
        to_be_deleted = set()
        for i, polygon in enumerate(polygons):
            assert (
                polygon.ndim == 2 and polygon.shape[-1] == 2
            ), f"a polygon should be 2 dimensional with shape Nx2. Got {polygon.shape}"
            x, y, w, h = cv2_bbox(polygon)
            if w < min_text_size or h < min_text_size:
                to_be_deleted.add(i)
                continue
            bboxs.append([x, y, w, h])

        return bboxs, to_be_deleted

    def train_transforms(self):
        hflip = A.HorizontalFlip(p=0.5)
        rotate = A.Rotate(limit=(-4, 4))
        rs = A.RandomScale(scale_limit=[0.5, 3.0], p=1.0, interpolation=INTER_CUBIC)
        resize = A.Resize(
            height=self._config.data.resize.height,
            width=self._config.data.resize.width,
            interpolation=INTER_CUBIC,
        )

        normalize = A.Normalize(max_pixel_value=1.0)
        color = A.ColorJitter(
            p=0.16, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
        )
        grey = A.ToGray(p=0.004)
        return A.Compose(
            [
                hflip,
                rs,
                resize,
                color,
                grey,
                rotate,
                normalize,
                color,
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", label_fields=["text"]),
            bbox_params=A.BboxParams(
                format="coco", min_visibility=1e-4, label_fields=["bbox_text"]
            ),
        )

    def val_transforms(self, viz=False):

        rc = A.Resize(
            height=self._config.data.resize.height,
            width=self._config.data.resize.width,
            interpolation=INTER_CUBIC,
        )
        if not viz:
            normalize = A.Normalize(max_pixel_value=1.0)
            return A.Compose(
                [rc, normalize, ToTensorV2()],
                keypoint_params=A.KeypointParams(format="xy", label_fields=["text"]),
                bbox_params=A.BboxParams(format="coco", label_fields=["bbox_text"]),
            )
        else:
            return A.Compose([rc, ToTensorV2()])

    def area(self, polygon):
        area = 0.0
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            area += (polygon[i][0] * polygon[j][1]) - (polygon[i][1] * polygon[j][0])

        return area

    def _seg_mask(self, h: int, w: int, textlist, polygonlist):
        mask = np.ones((1, h, w), dtype=np.float32)
        min_text_size = self._config.data.min_text_size
        pyclip = pyclipper.PyclipperOffset
        gt_img = np.zeros((1, h, w), dtype=np.float32)
        min_box_points = []
        bboxs = []
        text_list = []
        kept_polys = []
        for text, polygon in zip(textlist, polygonlist):
            # for each polygon in image
            points: np.ndarray = polygon  # 4 x 2
            if points.shape[0] != 4:
                continue
            height = np.max(points[:, 1]) - np.min(points[:, 1])
            width = np.max(points[:, 0]) - np.min(points[:, 0])
            if text == "###" or min(height, width) < min_text_size:

                cv2_fillPoly(mask[0], [points.astype(np.int32)], 0)  # fill out mask

            else:
                area = self.area(polygon=polygon)
                if area < 1:
                    if abs(area) < 1:
                        cv2_fillPoly(mask[0], [points.astype(np.int32)], 0)
                        continue
                    else:
                        polygon = polygon[(0, 3, 2, 1), :]

                padding = pyclip()
                padding.AddPath(
                    [tuple(p) for p in points],
                    pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON,
                )
                shrinked = padding.Execute(0)
                if not shrinked:
                    cv2_fillPoly(mask[0], [points.astype(np.int32)], 0)
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    cv2_fillPoly(gt_img[0], [shrinked.astype(np.int32)], 1)
                    min_box_points.append(
                        cv2_box_points(cv2_min_bbox(shrinked)).astype(np.float32)
                    )
                    bboxs.append(cv2_bbox(shrinked))
                    text_list.append(text)
                    kept_polys.append(points)

        if self._tokenizer:
            if not text_list:
                text_list = ["[PAD]"]
            text_list = self._tokenizer.encode_batch(text_list)

        # num_bboxs, 4, 2
        min_box_points = (
            np.zeros((1, 4, 2), dtype=np.float32)
            if not min_box_points
            else np.stack(min_box_points)
        )

        if not bboxs:
            bboxs = [[0.0, 0.0, 0.0, 0.0]]
        return (
            gt_img,
            mask,
            kept_polys,
            min_box_points,
            np.array(bboxs, dtype=np.float32),
            text_list,
        )

    @staticmethod
    def _normalize_polygons_(polygon: np.ndarray, expanded_polygon: np.ndarray):
        expanded = expanded_polygon.astype(np.float32)

        x = expanded[:, 0]
        y = expanded[:, 1]
        xmin, xmax = int(x.min()), int(x.max())
        ymin, ymax = int(y.min()), int(y.max())

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        return polygon, xmax - xmin + 1, ymax - ymin + 1, xmin, xmax, ymin, ymax

    def threshold_map(self, polygons: List[Polygon], h: int, w: int, shrink=False):
        mask = np.zeros((1, h, w), dtype=np.float32)
        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        canvas = np.zeros((1, h, w), dtype=np.float32)

        for polygon in polygons:
            shape_polygon = Polygon(polygon)
            distance = (
                shape_polygon.area
                * (1 - np.power(self._shrink_ratio, 2))
                / shape_polygon.length
            )
            distance = -1 * distance if shrink else distance
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(
                shape_polygon.exterior.coords,
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON,
            )

            padded_polygon = np.array(padding.Execute(distance)[0], dtype=np.int32)
            cv2_fillPoly(mask[0], [padded_polygon], 1.0)
            polygon, width, height, xmin, xmax, ymin, ymax = self._normalize_polygons_(
                polygon=polygon, expanded_polygon=padded_polygon
            )
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            points = MultiPoint(points=list(zip(x.flatten(), y.flatten())))
            pg = Polygon(polygon)
            buffer_points = pg.buffer(distance=distance)
            ls = LineString(pg.exterior.coords)
            contains_func = buffer_points.contains
            filtered_points = list(filter(contains_func, points.geoms))
            distances = map(ls.distance, filtered_points)
            distances: np.ndarray = np.clip(
                np.array(list(distances), dtype=np.float32) / distance, 0.0, 1.0
            )
            dist_map = np.ones((height, width), dtype=np.float32)
            xmin_valid = min(max(xmin, 0), w - 1)
            xmax_valid = max(min(xmax, w - 1), 0)
            ymin_valid = min(max(ymin, 0), h - 1)
            ymax_valid = max(min(ymax, h - 1), 0)

            for dist, point in zip(distances, filtered_points):
                dist_map[int(point.y), int(point.x)] = dist
                # everything more than one unit of distance away from original
                # exterior edge of polygon is a zero

            canvas[
                0, ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1
            ] = np.fmax(
                1.0
                - dist_map[
                    ymin_valid - ymin : ymax_valid - ymax + height,
                    xmin_valid - xmin : xmax_valid - xmax + width,
                ],
                canvas[0, ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1],
            )
            polygon[:, 0] += xmin
            polygon[:, 1] += ymin

        canvas = canvas * (self._thresh_max - self._thresh_min) + self._thresh_min

        return canvas, mask

    def lrtb_map(self, bboxes: np.ndarray, img_h: int, img_w: int):
        "left right top bottom distance map"
        """
        bboxes: x,y, w,h
        """

        left = bboxes[:, 0:1]
        right = bboxes[:, 2:3]
        top = bboxes[:, 1:2]
        bottom = bboxes[:, 3:4]  # N, 1
        left = left[:, np.newaxis, :]
        right = right[:, np.newaxis, :]
        top = top[:, np.newaxis, :]
        bottom = bottom[:, np.newaxis, :]  # N, 1, 1
        xs, ys = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))
        xs, ys = xs[np.newaxis, :, :], ys[np.newaxis, :, :]
        left_delta = xs - left  # N, h, w
        right_delta = right - xs
        top_delta = ys - top
        b_delta = bottom - ys

        deltas = np.stack(
            (left_delta, right_delta, top_delta, b_delta), axis=-1
        )  # n, h,w, 4

        li, ri, ti, bi = (
            left_delta >= -1e-5,
            right_delta >= -1e-5,
            top_delta >= -1e-5,
            b_delta >= -1e-5,
        )

        mask_indices = ~(li * ri * ti * bi)
        deltas[mask_indices] = float("inf")

        sum_dists = deltas.sum(axis=-1)  # n, h, w, 4 -> n, h, w
        sum_dists = sum_dists.reshape(bboxes.shape[0], -1)  # n, (h*w)
        bbox_idxs = sum_dists.argmin(axis=0)  # h*w

        bbox_idxs = bbox_idxs.reshape(img_h, img_w)

        deltas = np.take_along_axis(
            deltas, bbox_idxs[np.newaxis, :, :, np.newaxis], axis=0
        )
        deltas = deltas[0].reshape(img_h, img_w, 4)
        mask_indices = deltas.sum(axis=-1) == float("inf")
        deltas[mask_indices] = 0.0
        dl = deltas[:, :, 0]
        dr = deltas[:, :, 1]
        width_min = np.where(dl < dr, dl, dr)
        width_max = np.where(dl >= dr, dl, dr)
        width_ratio = width_min / (width_max + 1e-5)

        dt, db = deltas[:, :, 2], deltas[:, :, 3]

        height_min = np.where(dt < db, dt, db)
        height_max = np.where(dt >= db, dt, db)
        height_ratio = height_min / (height_max + 1e-5)

        centerness = np.sqrt(width_ratio * height_ratio)
        centerness[mask_indices] = 0.0
        deltas[..., 0:2] /= self._config.data.resize.width
        deltas[..., 2:4] /= self._config.data.resize.height
        return deltas, centerness[..., np.newaxis]  # H, W, 4; H,W, 1

    @staticmethod
    def batch(batches: List[Dict]) -> Dict[str, torch.Tensor]:

        data_dict = defaultdict(list)
        max_len = -1
        batch_size = len(batches)

        for batch in batches:
            for key, values in batch.items():
                data_dict[key].append(values)
                if "bboxes" == key:
                    max_len = max(max_len, len(values))
        if max_len == -1:
            raise ValueError(f"Expected to see key bboxes, but didn't see it")
        tensor_dict = {}

        for key, list_tensors in data_dict.items():
            if "bbox" not in key:
                if "text" != key:

                    tensor_dict[key] = torch.stack(list_tensors)
                else:
                    tensor_dict[key] = batch_text(list_tensors)
            else:

                tensor_dict[key] = batch_boxes(
                    batch_size=batch_size, torch_bboxes=data_dict[key], max_len=max_len
                )
        return tensor_dict


if __name__ == "__main__":

    @hydra.main(config_path="../../configs", config_name="default.yaml")
    def main(config):

        ds = ICDARDataset(config=config, train=True)
        dl = DataLoader(
            batch_size=1,
            dataset=ds,
            num_workers=0,
            collate_fn=ICDARDataset.batch,
            shuffle=True,
        )

        import matplotlib.pyplot as plt
        import time
        import cv2

        t0 = time.time()
        for data in dl:

            """img = data["image"]
            img = np.ascontiguousarray(img[0].permute(1, 2, 0).numpy())
            mask = data["thresh_mask"]
            thresh_map = data["thresh_map"]
            bboxes = data["bboxes"][0]
            # text = data["text"]
            # print(text)
            for bbox in bboxes:
                x, y, w, h = bbox

                cv2.rectangle(
                    img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2
                )
            plt.imshow(img)
            plt.show()
            # temp_mask[temp_mask == 2] = 0

            plt.imshow(mask[0].permute(1, 2, 0).numpy())
            plt.show()
            plt.imshow(thresh_map[0].permute(1, 2, 0).numpy())
            plt.show()
            plt.imshow(img * data["gt_img"][0].permute(1, 2, 0).numpy())
            plt.show()
            """
        print(time.time() - t0)

    main()
