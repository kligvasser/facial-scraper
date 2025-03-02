import torch
import cv2
import pandas as pd
import numpy as np

from facenet_pytorch import MTCNN


class FaceExtractor:
    def __init__(
        self,
        prob_threshold=0.975,
        change_var_threshold=0.06,
        box_var_threshold=0.06,
        min_box_size=0.1,
        min_block_size=16,
        max_block_size=16 * 8,
        max_resized_dim=150,
        num_frame_skip=2,
        min_frame_skip=2,
        chunk_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.prob_threshold = prob_threshold
        self.change_var_threshold = change_var_threshold
        self.box_var_threshold = box_var_threshold
        self.min_box_size = min_box_size
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.max_resized_dim = max_resized_dim
        self.num_frame_skip = num_frame_skip
        self.min_frame_skip = min_frame_skip
        self.chunk_size = chunk_size
        self.device = device
        self.mtcnn = MTCNN(select_largest=True, device=device)

    def extract(self, frames):
        resized = self.resize_and_skip_frames(frames)
        height, width, _ = resized[0].shape

        boxes, probs = list(), list()
        for i in range(0, len(resized), self.chunk_size):
            box, prob = self.mtcnn.detect(
                resized[i : i + self.chunk_size], landmarks=False
            )
            boxes += box.tolist()
            probs += [x[0] for x in prob]

        blocks = self.slice_blocks(boxes, probs, [height, width])
        faces, indexes = self.extract_faces_from_blocks(frames, blocks)
        return faces, indexes

    def resize_and_skip_frames(self, frames, interpolation=cv2.INTER_LINEAR):
        h, w = frames[0].shape[:2]
        scale = float(self.max_resized_dim) / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        frames_ = frames[:: self.num_frame_skip]

        return [
            cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
            for frame in frames_
        ]

    def slice_blocks(self, boxes, probs, size):
        def _extract_box_values(row):
            b0 = row["box"][0][0] / size[1]
            b1 = row["box"][0][1] / size[0]
            b2 = row["box"][0][2] / size[1]
            b3 = row["box"][0][3] / size[0]
            s0 = b2 - b0
            s1 = b3 - b1

            return pd.Series(
                [b0, b1, b2, b3, s0, s1], index=["b0", "b1", "b2", "b3", "s0", "s1"]
            )

        df = pd.DataFrame({"prob": probs, "box": boxes})

        df["index"] = df.index
        df = df[df["prob"] >= self.prob_threshold]

        if len(df) == 0:
            return []

        df_ = df.apply(_extract_box_values, axis=1)
        df = pd.concat([df, df_], axis=1)

        df = df[(df["s0"] > self.min_box_size) & (df["s1"] > self.min_box_size)]

        for b in ["b0", "b1", "b2", "b3", "index"]:
            df["{}d".format(b)] = df[b].diff().abs()

        change_points = df[
            (df["b0d"] > self.change_var_threshold)
            | (df["b1d"] > self.change_var_threshold)
            | (df["b2d"] > self.change_var_threshold)
            | (df["b3d"] > self.change_var_threshold)
            | (df["indexd"] > self.min_frame_skip)
        ].index
        split_points = change_points.tolist()

        blocks = list()
        if len(split_points) == 0:
            blocks.append(df)
        elif len(split_points) == 1:
            blocks.append(df.iloc[: split_points[0]])
            blocks.append(df.iloc[split_points[0] :])
        else:
            for n in range(len(split_points) - 1):
                block = df.iloc[split_points[n] : split_points[n + 1]]
                if len(block) > self.min_block_size:
                    blocks.append(block)

        self.df = df

        return blocks

    def extract_faces_from_blocks(self, frames, blocks):
        faces, indexes = list(), list()
        for block in blocks:
            if (
                (len(block["b0"].values) == 0)
                or (block["b0"].std() > self.box_var_threshold)
                or (block["b1"].std() > self.box_var_threshold)
                or (block["b2"].std() > self.box_var_threshold)
                or (block["b3"].std() > self.box_var_threshold)
            ):
                continue

            bbox = [
                block["b0"].values.min(),
                block["b1"].values.min(),
                block["b2"].values.max(),
                block["b3"].values.max(),
            ]
            block_indexes = block["index"].values.tolist()
            start_index = block_indexes[0]
            end_index = min(block_indexes[-1], start_index + self.max_block_size)
            start_index *= self.num_frame_skip
            end_index *= self.num_frame_skip

            fframes = frames[start_index:end_index]
            faces.append(self.extract_face_square(fframes, bbox))
            indexes.append([start_index, end_index])
        return faces, indexes

    def extract_face_square(self, frames, bbox, margin=0.4):
        min_x, min_y, max_x, max_y = bbox
        height, width, num_channel = frames[0].shape

        center_x = (min_x + max_x) * width // 2
        center_y = (min_y + max_y) * height // 2
        size = max(int((max_x - min_x) * width), int((max_y - min_y) * height))
        size += int(margin * size)

        crop_x1 = int(max(center_x - size / 2, 0))
        crop_y1 = int(max(center_y - size / 2, 0))
        crop_x2 = int(min(center_x + size / 2, width))
        crop_y2 = int(min(center_y + size / 2, height))

        faces = list()
        for frame in frames:
            cropped_image = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
            squared_image = np.zeros((size, size, num_channel), dtype=frame.dtype)
            squared_image[: cropped_image.shape[0], : cropped_image.shape[1], :] = (
                cropped_image
            )
            faces.append(squared_image)

        return faces
