import torchvision.models as models
import torch.nn as nn


def get_swin_s(n_classes, grad=False):
    model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.head = nn.Linear(model.head.in_features, n_classes)
    return model


def get_resnet152(n_classes, grad=False):
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad

    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_wide_resnet101(n_classes, grad=False):
    model = models.wide_resnet101_2(
        weights=models.Wide_ResNet101_2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_efficientnet_b7(n_classes, grad=False):
    model = models.efficientnet_b7(
        weights=models.EfficientNet_B7_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_efficientnet_v2_m(n_classes, grad=False):
    model = models.efficientnet_v2_m(
        weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_densenet161(n_classes, grad=False):
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)

    return model


def get_efficientnet_b7_mod(n_classes, grad=False):
    model = models.efficientnet_b7(
        weights=models.EfficientNet_B7_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 64), nn.ReLU(),
                                        nn.BatchNorm1d(num_features=64), nn.Dropout(0.3), nn.Linear(64, n_classes))
    return model


def get_efficientnet_v2_s_mod(n_classes, grad=False):
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 64), nn.ReLU(),
                                        nn.BatchNorm1d(num_features=64), nn.Dropout(0.3), nn.Linear(64, n_classes))

    return model


def get_densenet161_mod(n_classes, grad=False):
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier.in_features, 64), nn.ReLU(),
                                     nn.BatchNorm1d(num_features=64), nn.Dropout(0.3), nn.Linear(64, n_classes))
    return model


def get_wide_resnet101_mod(n_classes, grad=False):
    model = models.wide_resnet101_2(
        weights=models.Wide_ResNet101_2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 64), nn.ReLU(),
                             nn.BatchNorm1d(num_features=64), nn.Dropout(0.3), nn.Linear(64, n_classes))

    return model


def get_swin_s_mod(n_classes, grad=False):
    model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad
    model.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.head.in_features, 64), nn.ReLU(),
                               nn.BatchNorm1d(num_features=64), nn.Dropout(0.3), nn.Linear(64, n_classes))
    return model


def get_resnet18_mod(n_classes, grad=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = grad

    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 64),
                             nn.ReLU(), nn.BatchNorm1d(num_features=64),
                             nn.Dropout(0.3), nn.Linear(64, n_classes))
    return model


def get_resnet152_mod(n_classes, grad=False):
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = grad

    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 64),
                             nn.ReLU(), nn.BatchNorm1d(num_features=64),
                             nn.Dropout(0.3), nn.Linear(64, n_classes))

    return model
