import albumentations as A
from albumentations.pytorch import ToTensorV2


geometric_transforms = A.OneOf(
    [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.30, rotate_limit=15),
     A.GridDistortion(num_steps=5),
     A.Perspective()
     ], p=0.5)

color_transforms = A.OneOf([
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.Equalize(),
    A.HueSaturationValue(hue_shift_limit=7),
], p=0.33)

noise_transforms = A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0)),
                            A.GaussianBlur(blur_limit=(1, 5)),
                            A.Downscale(0.5, 0.5),
                            A.ISONoise()
                            ], p=0.33)

before_input_transforms = A.Compose([A.Resize(128, 128),
                                     A.Normalize(), ToTensorV2()
                                     ], p=1)

train_transforms = A.Compose([A.HorizontalFlip(p=0.5),
                              geometric_transforms,
                              color_transforms,
                              noise_transforms,
                              before_input_transforms
                              ])

test_transforms = A.Compose([A.Resize(128, 128), A.Normalize(), ToTensorV2()])
