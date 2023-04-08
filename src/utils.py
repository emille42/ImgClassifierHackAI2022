from sklearn.model_selection import KFold
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from image_dataset import ImgDataset
from torch.utils.data import DataLoader
from train_model import train_model
import os


def predict(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    pred_labels = []
    img_names = []

    for inputs, paths in tqdm(dataloader):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        pred_labels.append(nn.functional.softmax(
            preds, dim=1).argmax(-1).data.cpu().numpy())
        img_names.extend(paths)

    pred_labels = np.concatenate(pred_labels)
    return pred_labels, img_names


def show_images(dataloader, n=10):
    img_tensors, labels = next(iter(dataloader))
    for idx in range(n):
        image = img_tensors[idx].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        plt.imshow(image.clip(0, 1))
        plt.title("Класс: " + str(int(labels[idx])))
        plt.show()


def cross_validation(models_list, train_df, n_splits, batch_size, grad,
                     transforms, train_path, n_epochs, metric, n_classes=1, device='cpu', SEED=42):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    losses = []
    metrics = []
    for idx, get_model_f in enumerate(models_list):
        model_losses = []
        model_metrics = []

        fold_idx = 1
        for train_indices, test_indices in kfold.split(train_df):

            print(f"Start train model {idx+1}, fold {fold_idx}")
            model = get_model_f(n_classes=n_classes, grad=grad)
            loss_f = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

            train = train_df.iloc[train_indices]
            train_dataset = ImgDataset(
                train, train_path, transform=transforms[0], with_label=True)
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            test = train_df.iloc[test_indices]
            test_dataset = ImgDataset(
                test, train_path, transform=transforms[1], with_label=True)
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            _, best_loss, best_metric = train_model(model, loss_func=loss_f,
                                                    train_dataloader=train_dataloader,
                                                    val_dataloader=test_dataloader,
                                                    optimizer=optimizer, n_epochs=n_epochs,
                                                    device=device, metric=metric, logs=False)
            model_losses.append(best_loss)
            model_metrics.append(best_metric)
            print(f"End train model {idx+1}, fold {fold_idx}\n")
            fold_idx += 1

        losses.append(np.mean(model_losses))
        metrics.append(np.mean(model_metrics))
        print(f"Model {idx+1} mean loss {np.mean(model_losses)}")
        print(f"Model {idx+1} mean metric {np.mean(model_metrics)}")

    losses = {idx+1: val for idx, val in enumerate(losses)}
    metrics = {idx+1: val for idx, val in enumerate(metrics)}
    print(f"Mean losses: {losses}")
    print(f"Mean metrics: {metrics}")


def augment_and_save(df, max_count, data_path, save_folder, transforms):
    aug_df = pd.DataFrame(columns=df.columns)
    written = 0

    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        img_class = row.iloc[1]

        class_n = df[df['class'] == img_class].shape[0]
        if class_n > max_count:
            continue

        img_path = os.path.join(data_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aug_img = transforms(image=image)['image']

        aug_name = f"aug_{idx}_" + img_name
        save_path = os.path.join(save_folder, aug_name)

        if cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)):
            written += 1
            aug_df.loc[len(aug_df)] = [aug_name, img_class]

    print(f"Written {written} augmented images total. Path: {save_folder}")

    return aug_df


def show_images(dataloader, n=10):
    img_tensors, labels = next(iter(dataloader))
    for idx in range(n):
        image = img_tensors[idx].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        plt.imshow(image.clip(0, 1))
        plt.title("Класс: " + str(int(labels[idx])))
        plt.show()


def predict(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    pred_labels = []
    img_names = []

    for inputs, paths in tqdm(dataloader):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        pred_labels.append(nn.functional.softmax(
            preds, dim=1).argmax(-1).data.cpu().numpy())
        img_names.extend(paths)

    pred_labels = np.concatenate(pred_labels)
    return pred_labels, img_names
