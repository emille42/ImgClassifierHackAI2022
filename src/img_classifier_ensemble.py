
import torch.nn as nn
import torch


class ImgClassifiersEnsemble(nn.Module):
    def __init__(self, list_of_models, n_classes):
        super().__init__()
        self.models = nn.ModuleList(list_of_models)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(n_classes*len(list_of_models))
        self.classifier = nn.Linear(n_classes*len(list_of_models), n_classes)

    def forward(self, x):
        outs = []
        for model in self.models:
            out = model(x)
            outs.append(out)

        stacked = torch.cat(outs, dim=-1)
        stacked = self.relu(stacked)
        stacked = self.bn(stacked)
        out = self.classifier(stacked)
        return out
