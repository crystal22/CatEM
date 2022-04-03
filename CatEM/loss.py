import torch.nn as nn
import torch


# 二元交叉熵损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):  # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)


class MeanSquareMultiplyWeightLoss(nn.Module):
    def __init__(self):
        super(MeanSquareMultiplyWeightLoss, self).__init__()

    def forward(self, pred, target, G):
        part = torch.square(pred - target)
        result = part.mul(G)
        return result.mean()


class MeanSquareWithManifoldItem(nn.Module):
    def __init__(self):
        super(MeanSquareWithManifoldItem, self).__init__()

    def forward(self, net, M, W, alpha):
        U = net[0].weight
        V = net[1].weight
        part1 = torch.square(U.mm(V.T) - M)
        total_loss = torch.sum(part1)
        for i in range(M.shape[0]):
            temp_u = U[i].view(-1, U[i].shape[0])
            temp_u = temp_u.repeat(M.shape[0], 1)
            # yummy
            part2 = alpha * torch.sum(torch.sum(torch.square(temp_u - V), dim=1) * W[i])

            total_loss += part2
        return total_loss / (M.shape[0] * M.shape[1])


class MeanSquareWithLaplacianRegularizer(nn.Module):
    def __init__(self):
        super(MeanSquareWithLaplacianRegularizer, self).__init__()

    def forward(self, net, M, A, alpha):
        U = net[0].weight
        V = net[1].weight
        # part1 = (1 - alpha) * torch.square(U.mm(V.T) - M)

        part1 = torch.square(U.mm(V.T) - M)
        part1_loss = torch.sum(part1)

        A = (A + torch.transpose(A, 0, 1)) / 2
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        part2 = torch.mm(torch.mm(U.T, L), U)
        part2_loss = part2.trace()

        total_loss = part1_loss + alpha * part2_loss

        return total_loss / (M.shape[0] * M.shape[1])


class MeanSquareWithAdaptiveConstraintAndSpatialEnhanced(nn.Module):
    def __init__(self):
        super(MeanSquareWithAdaptiveConstraintAndSpatialEnhanced, self).__init__()

    def forward(self, net, M, W1, W2, lambda1, lambda2):
        U = net[0].weight
        V = net[1].weight
        part1 = torch.square(U.mm(V.T) - M)
        total_loss = torch.sum(part1)
        for i in range(M.shape[0]):
            temp_u = U[i].view(-1, U[i].shape[0])
            temp_u = temp_u.repeat(M.shape[0], 1)
            # yummy
            part2 = lambda2 * torch.sum(torch.sum(torch.square(temp_u - V), dim=1) * W1[i])
            part3 = lambda1 * torch.sum(torch.sum(torch.square(temp_u - V), dim=1) * W2[i])

            total_loss += (part2 + part3)

        total_loss += torch.sqrt(torch.sum(torch.square(W1)))
        return total_loss / (M.shape[0] * M.shape[1])
