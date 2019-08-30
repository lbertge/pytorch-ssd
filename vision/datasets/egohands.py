import numpy as np
import pathlib
import cv2
import pandas as pd

class EgoHandsDataset:

    def __init__(self, root, transform=None, target_transform=None, dataset_type="train", balance_data=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        # data not unbalanced
        self.balance_data = balance_data
        self.min_image_num = -1
        
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])

    def _read_data(self):
        annotation_file = f"{self.root}/{self.dataset_type}/{self.dataset_type}_labels.csv"
        annotations = pd.read_csv(annotation_file)

        # TODO not sure if safe to assume all pics are class "hand"
        class_names = ['BACKGROUND'] + ["hand"]
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        filesize = [1280, 720, 1280, 720]
        for image_id, group in annotations.groupby("filename"):
            boxes = group.loc[:, ["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)
            boxes = np.divide(boxes, filesize)
            assert np.all(boxes >= 0) and np.all(boxes <= 1)

            labels = np.array([class_dict[name] for name in group["class"]])
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_state is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in examples['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                f"Number of Images: {len(self)}",
                f"Min Number of Images: {self.min_image_num}",
                "Label Distribution:"]

        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / image_id
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

