import torch
import torch.nn
from models.model_utils import fps_subsample
from metrics.CD.chamfer3D import dist_chamfer_3D
from metrics.CD.fscore import fscore
chamfer_dist = dist_chamfer_3D.chamfer_3DDist()



def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1

def get_loss(pcds_pred,   gt, sqrt=True,alpha1=1,alpha2=1):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred

    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt)
    # partial_matching = PM(partial, P2)


    loss_all = cdc + alpha1*cd1 + alpha2*cd2
    losses = [cdc, cd1, cd2]
    return loss_all, losses

def get_loss_hyper(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """

    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred
    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = calc_cd_like_hyperV2(Pc, gt_c)
    # cd1 = CD(P1, gt_1)
    cd1 = calc_cd_like_hyperV2(P1, gt_1)
    # cd2 = CD(P2, gt_2)
    cd2 = calc_cd_like_hyperV2(P2, gt)

    partial_matching = calc_cd_one_side_like_hyperV2(partial, P2)

    loss_all = (cdc + cd1 + cd2 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, partial_matching]   
    return loss_all, losses

def get_loss_hyper_pcn(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """

    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred
    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = calc_cd_like_hyperV2(Pc, gt_c)
    # cd1 = CD(P1, gt_1)
    cd1 = calc_cd_like_hyperV2(P1, gt_1)
    # cd2 = CD(P2, gt_2)
    cd2 = calc_cd_like_hyperV2(P2, gt)

    # partial_matching = calc_cd_one_side_like_hyperV2(partial, P2)

    loss_all = cdc + cd1 + cd2 
    losses = [cdc, cd1, cd2]   
    return loss_all, losses

def get_loss_hyper_pcn_FM(pcds_pred, partial, gt, sqrt=True):
    """Simplified loss function for single-stage output.

    Args:
        pcds_pred: Predicted point cloud, shape [B, 3, N] or a list with one element.
        partial: Input partial point cloud, shape [B, 3, M].
        gt: Ground-truth complete point cloud, shape [B, 3, N_gt].
        sqrt: Whether to use sqrt version of chamfer distance.

    Returns:
        loss_all: Total loss (scalar).
        losses: List containing only one element [cd2], for compatibility.
    """

    if sqrt:
        CD = chamfer_sqrt
    else:
        CD = chamfer

    # 如果 pcds_pred 是 list，取第一个
    if isinstance(pcds_pred, list):
        P2 = pcds_pred[0]
    else:
        P2 = pcds_pred

    # 与 ground-truth 点云计算 CD loss
    cd2 = calc_cd_like_hyperV2(P2, gt)

    loss_all = cd2
    losses = [cd2]

    return loss_all, losses








def calc_cd_like_hyperV2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = cham_loss(array1, array2)
    # dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    # d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)
    d1 = arcosh(1+ 1 * d1)
    d2 = arcosh(1+ 1 * d2)
    # print(d1.shape)
    # print(d2.shape)

    return torch.mean(d1) + torch.mean(d2)



def calc_cd_one_side_like_hyperV2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = cham_loss(array1, array2)
    # dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    # d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)
    d1 = arcosh(1+ 1 * d1)
    # d2 = arcosh(1+d2)
    # print(d1.shape)
    # print(d2.shape)

    return torch.mean(d1)



# distances = distances.clamp(-1 + eps, 1 - eps)
def arcosh(x, eps=1e-5):  # pragma: no cover
    # x = x.clamp(-1 + eps, 1 - eps)
    # x = x.clamp(1,)
    x = torch.clamp(x, min=1 + eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))












def get_loss_PM(pcds_pred, partial,  gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred

    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt)
    partial_matching = PM(partial, P2)


    loss_all = cdc + cd1 + cd2+partial_matching
    losses = [cdc, cd1, cd2]
    return loss_all, losses

# def calc_cd(output, gt, calc_f1=False):
#     cham_loss = dist_chamfer_3D.chamfer_3DDist()
#     dist1, dist2, _, _ = cham_loss(gt, output)
#     cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
#     cd_t = (dist1.mean(1) + dist2.mean(1))
#     if calc_f1:
#         f1, recall, precision = fscore(dist1, dist2)
#         return cd_p, cd_t, f1
#     else:
#         return cd_p, cd_t

def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    # cham_loss = cd()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res
