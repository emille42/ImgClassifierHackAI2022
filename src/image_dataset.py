from torch.utils.data import Dataset
import os
import cv2


class ImgDataset(Dataset):
    def __init__(self, img_df, img_path, transform=None, with_label=True):
        self.img_df = img_df
        self.img_path = img_path
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, index):
        path = os.path.join(self.img_path, self.img_df.iloc[index, 0])
        if self.with_label:
            label = int(self.img_df.iloc[index, 1])

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.with_label:
            return image, label
        else:
            return image, self.img_df.iloc[index, 0]
