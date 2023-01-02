# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:03:22 2020
@author: islam

Code from https://github.com/rashid-islam/Differential_Fairness/

Edited by: #####
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#################################
## MULTICLASS STOCASTIC FUNCS  ##
#################################
## extends islam's code for    ##
## multiclass predictions      ##
#################################

# Loss and optimizer
def multiclass_fairness_loss(stochasticModel,
    num_classes, # number of labels (precomputed)
    base_fairness=0.0, # base fairness (may relax fairness if necessary) 0.2231 is 80% rule
    concentration=.1, # smoothing value (more smoothing == smoother loss but less accurate)):
    ):
    # DF-based penalty term
    dirichletAlpha = concentration/num_classes
    zeroTerm = torch.tensor(0.0)

    theta = (stochasticModel.countClass_hat + dirichletAlpha).T / \
        (stochasticModel.countTotal_hat + concentration)
    epsilonClass = differentialFairnessMulticlassOutcomeTrain(theta.T, num_classes)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))

# Loss and optimizer
def multiclass_equalized_opportunity_loss(stochasticModel,
    num_classes, # number of labels (precomputed)
    base_fairness=0.0, # base fairness (may relax fairness if necessary) 0.2231 is 80% rule
    concentration=.1, # smoothing value (more smoothing == smoother loss but less accurate)):
    ):
    # DF-based penalty term
    dirichletAlpha = concentration/num_classes
    zeroTerm = torch.tensor(0.0)

    # Calculating TPR
    TPR = torch.zeros((stochasticModel.no_of_groups, stochasticModel.no_of_classes), dtype=torch.float, device=stochasticModel.countTotal_hat.device.type)
    for g in range(stochasticModel.no_of_groups):
        for k in range(stochasticModel.no_of_classes):
            TPR[g,k] = stochasticModel.countClass_hat[g, k, k]

    theta = (TPR + dirichletAlpha) / \
        (stochasticModel.countTotal_hat + concentration)
    epsilonClass = differentialFairnessMulticlassOutcomeTrain(theta, num_classes)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))

   # Loss and optimizer


def multiclass_equalized_odds_loss(stochasticModel,
                                   # number of labels (precomputed)
                                   num_classes,
                                   # base fairness (may relax fairness if necessary) 0.2231 is 80% rule
                                   base_fairness=0.0,
                                   # smoothing value (more smoothing == smoother loss but less accurate)):
                                   concentration=.1,
                                   ):
    # DF-based penalty term
    dirichletAlpha = concentration/num_classes
    zeroTerm = torch.tensor(0.0)

    # Calculating TP counts
    TP = torch.zeros((stochasticModel.no_of_groups,
                     stochasticModel.no_of_classes), dtype=torch.float,
                     device=stochasticModel.countTotal_hat.device.type)
    for g in range(stochasticModel.no_of_groups):
        for k in range(stochasticModel.no_of_classes):
            TP[g, k] = stochasticModel.countClass_hat[g, k, k]
    # Calculating FP counts
    FP = torch.zeros((stochasticModel.no_of_groups,
                     stochasticModel.no_of_classes), dtype=torch.float, 
                     device=stochasticModel.countTotal_hat.device.type)
    for g in range(stochasticModel.no_of_groups):
        for k in range(stochasticModel.no_of_classes):
            FP[g, k] = torch.sum(
                stochasticModel.countClass_hat[g, :, k]) - stochasticModel.countClass_hat[g, k, k]
    # Calculating Neg counts
    N = torch.zeros((stochasticModel.no_of_groups,
                    stochasticModel.no_of_classes), dtype=torch.float, 
                    device=stochasticModel.countTotal_hat.device.type)
    for g in range(stochasticModel.no_of_groups):
        for k in range(stochasticModel.no_of_classes):
            N[g, k] = torch.sum(
                stochasticModel.countTotal_hat[g, :]) - stochasticModel.countTotal_hat[g, k]

    theta_tpr = (TP + dirichletAlpha) / \
        (stochasticModel.countTotal_hat + concentration)
    epsilonClass = differentialFairnessMulticlassOutcomeTrain(
        theta_tpr, num_classes)
    theta_fpr = (FP + dirichletAlpha) / \
        (N + concentration)
    epsilonClass = epsilonClass + \
        differentialFairnessMulticlassOutcomeTrain(theta_fpr, num_classes)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))


FAIR_LOSS_DICT = {
    "differential_fairness": multiclass_fairness_loss,
    "equalized_opportunity": multiclass_equalized_opportunity_loss,
    "equalized_odds": multiclass_equalized_odds_loss 
}

#################################
## MULTICLASS HARD COUNT FUNC  ##
#################################


def multiclass_equalized_odds(
        count_pos,
        count_total,
        num_classes,
        num_groups,
        base_fairness=0.0,
        concentration=.1,
        device='cpu'
    ):
    # DF-based penalty term
    dirichletAlpha = concentration / num_classes
    zeroTerm = torch.tensor(0.0)

    # Calculating TP counts
    TP = torch.zeros((num_groups, num_classes), dtype=torch.float, device=device)
    FP = torch.zeros((num_groups, num_classes), dtype=torch.float, device=device)
    N = torch.zeros((num_groups, num_classes), dtype=torch.float, device=device)
    for g in range(num_groups):
        for k in range(num_classes):
            TP[g, k] = count_pos[g, k, k]
            FP[g, k] = torch.sum(count_pos[g, :, k]) - count_pos[g, k, k]
            N[g, k] = torch.sum(count_total[g, :]) - count_total[g, k]

    theta_tpr = (TP + dirichletAlpha) / (count_total + concentration)
    epsilonClass = differentialFairnessMulticlassOutcomeTrain(theta_tpr, num_classes)
    theta_fpr = (FP + dirichletAlpha) / (N + concentration)
    epsilonClass = epsilonClass + \
        differentialFairnessMulticlassOutcomeTrain(theta_fpr, num_classes)
    return torch.max(zeroTerm, (epsilonClass - base_fairness))


def multiclass_differential_fairness(
        count_pos,
        count_total,
        num_classes,
        base_fairness=0.0,
        concentration=.1,
        ):
    # DF-based penalty term
    dirichletAlpha = concentration / num_classes
    zeroTerm = torch.tensor(0.0)

    theta = (count_pos + dirichletAlpha).T / \
        (count_total + concentration)
    epsilonClass = differentialFairnessMulticlassOutcomeTrain(theta.T, num_classes)
    return torch.max(zeroTerm, (epsilonClass - base_fairness))


#################################
## MULTICLASS COUNT FUNCS      ##
#################################

def differentialFairnessMulticlassOutcomeTrain(probabilitiesOfPositive, num_classes):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(
        len(probabilitiesOfPositive), dtype=torch.float)
    for i in range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0)  # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                for k in range(num_classes):
                    epsilon = torch.max(epsilon, torch.abs(torch.log(probabilitiesOfPositive[i,k])-torch.log(
                        probabilitiesOfPositive[j,k])))  # ratio of probabilities of positive outcome
        epsilonPerGroup[i] = epsilon  # DF per group
    epsilon = torch.max(epsilonPerGroup)  # overall DF of the algorithm
    return epsilon


def computeMulticlassBatchCounts(protectedAttributes, predictions, intersectGroups, num_classes, device, labels=None):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S)
    # can be maintained correctly among different batches
    count_pos_k = torch.zeros(
        (len(intersectGroups), num_classes), device=device)
    count_tot_k = torch.zeros((len(intersectGroups)), device=device)
    for i in range(len(protectedAttributes)):
        index = np.where((np.array(intersectGroups) ==
                         protectedAttributes[i]))[0][0]
        count_tot_k[index] = count_tot_k[index] + 1
        for k in range(num_classes):
            count_pos_k[index, k] = count_pos_k[index, k] + predictions[i, k]
    return count_pos_k, count_tot_k


def computeMulticlassBatchCountsPerLabel(protectedAttributes, predictions, intersectGroups, num_classes, device, labels=None):
    # intersectGroups should be pre-defined so that stochastic update of p(y_hat|S,y=1)
    # can be maintained correctly among different batches
    count_pos_k = torch.zeros(
        (len(intersectGroups), num_classes, num_classes), device=device)
    count_tot_k = torch.zeros((len(intersectGroups), num_classes), device=device)
    for i in range(len(protectedAttributes)):
        index = np.where((np.array(intersectGroups) ==
                         protectedAttributes[i]))[0][0]
        count_tot_k[index, labels[i]] = count_tot_k[index, labels[i]] + 1
        for k in range(num_classes):
            count_pos_k[index, labels[i], k] = count_pos_k[index, labels[i], k] + predictions[i, k]
    return count_pos_k, count_tot_k


#################################
## HARD COUNTS                 ##
#################################
def compute_multiclass_hard_batch_counts(
        protectedAttributes, 
        predictions, 
        intersectGroups, 
        num_classes, 
        device, 
        labels=None
    ):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S)
    # can be maintained correctly among different batches
    count_pos_k = torch.zeros((len(intersectGroups), num_classes), device=device)
    count_tot_k = torch.zeros((len(intersectGroups)), device=device)
    for i in range(len(protectedAttributes)):
        index = np.where((np.array(intersectGroups) ==
                         protectedAttributes[i]))[0][0]
        count_tot_k[index] = count_tot_k[index] + 1
        count_pos_k[index, predictions[i]] = count_pos_k[index, predictions[i]] + 1
    return count_pos_k, count_tot_k


def compute_multiclass_hard_batch_counts_per_label(
        protectedAttributes, 
        predictions, 
        intersectGroups, 
        num_classes, 
        device, 
        labels=None
    ):
    # intersectGroups should be pre-defined so that stochastic update of p(y_hat|S,y=1)
    # can be maintained correctly among different batches
    count_pos_k = torch.zeros(
        (len(intersectGroups), num_classes, num_classes), device=device)
    count_tot_k = torch.zeros(
        (len(intersectGroups), num_classes), device=device)
    for i in range(len(protectedAttributes)):
        index = np.where((np.array(intersectGroups) ==
                         protectedAttributes[i]))[0][0]
        count_tot_k[index, labels[i]] = count_tot_k[index, labels[i]] + 1
        count_pos_k[index, labels[i], predictions[i]] = count_pos_k[index, labels[i], predictions[i]] + 1
    return count_pos_k, count_tot_k


class stochasticCount(nn.Module):
    def __init__(self, N, batch_size, rho=.1):
        super(stochasticCount, self).__init__()
        self.countClass_hat = None
        self.countTotal_hat = None

        self.N = N
        self.batch_size = batch_size
        self.rho = rho

    def forward(self, countClass_batch, countTotal_batch):
        self.countClass_hat = (1-self.rho)*self.countClass_hat + \
            self.rho*(self.N/self.batch_size)*countClass_batch
        self.countTotal_hat = (1-self.rho)*self.countTotal_hat + \
            self.rho*(self.N/self.batch_size)*countTotal_batch


class stochasticMulticlassCountModel(stochasticCount):
    def __init__(self, no_of_groups, no_of_classes, N, batch_size, rho=.1, device='cpu'):
        super(stochasticMulticlassCountModel, self).__init__(N, batch_size, rho)
        self.no_of_groups = no_of_groups
        self.no_of_classes = no_of_classes
        self.countClass_hat = torch.ones(
            (no_of_groups, no_of_classes), device=device)
        self.countTotal_hat = torch.ones((no_of_groups), device=device)

        self.countClass_hat = self.countClass_hat * \
            (N/(batch_size*no_of_groups))
        self.countTotal_hat = self.countTotal_hat*(N/batch_size)


class stochasticMulticlassPerLabelCountModel(stochasticCount):
    def __init__(self, no_of_groups, no_of_classes, N, batch_size, rho=.1, device='cpu'):
        super(stochasticMulticlassPerLabelCountModel,
              self).__init__(N, batch_size, rho)
        self.no_of_groups = no_of_groups
        self.no_of_classes = no_of_classes
        self.countClass_hat = torch.ones(
            (no_of_groups, no_of_classes, no_of_classes), device=device)
        self.countTotal_hat = torch.ones((no_of_groups, no_of_classes), device=device)

        self.countClass_hat = self.countClass_hat * \
            (N/(batch_size*no_of_groups))
        self.countTotal_hat = self.countTotal_hat*(N/batch_size)

#################################
## OLD FUNCTIONS               ##
#################################

def eEqualizedOddsLoss(
    preds, # model output 
    protected_attributes, # list of protected atributes
    gold, # true labels
    num_classes, # number of labels (precomputed)
    intersect_groups, # list of all protected attributes (precomputed)
    device,
    base_fairness=0.0, # base fairness (may relax fairness if necessary)
    concentration=.1, # smoothing value (more smoothing == smoother loss but less accurate)
    ):
    ## Calculate  the counts
    groups = torch.tensor([intersect_groups.index(pa) for pa in protected_attributes['demographics']]).to(device)
    count_pos_k, count_tot_k = get_counts(preds, groups, gold, num_classes, len(intersect_groups))
    d_alpha = concentration / num_classes
    TPR = (count_pos_k + d_alpha) / (count_tot_k + concentration)
    FPR = (count_tot_k - count_pos_k + d_alpha) / (count_tot_k + concentration)
    # calculate epsilon tpr
    e_tpr = torch.max(torch.abs(torch.log(torch.min(TPR, dim=1)[0])-torch.log(torch.max(TPR, dim=1)[0])))
    e_fpr = torch.max(torch.abs(torch.log(torch.min(FPR, dim=1)[0])-torch.log(torch.max(FPR, dim=1)[0])))
    return torch.max(torch.tensor(0.0), (e_tpr + e_fpr) - base_fairness)

def eEqualizedOpportunityLoss(
    predictions, # model output 
    protected_attributes, # list of protected atributes
    y, # true labels
    num_classes, # number of labels (precomputed)
    intersect_groups, # list of all protected attributes (precomputed)
    base_fairness=0.0, # base fairness (may relax fairness if necessary)
    concentration=.1, # smoothing value (more smoothing == smoother loss but less accurate)
    ):
    ## Calculate  the counts
    preds = torch.argmax(predictions, dim=1).detach().cpu()
    groups = torch.tensor([intersect_groups.index(pa) for pa in protected_attributes['demographics']])
    gold = y.detach().cpu()
    count_pos_k, count_tot_k = get_counts(preds, groups, gold, num_classes, len(intersect_groups))
    d_alpha = concentration / num_classes
    TPR = (count_pos_k + d_alpha) / (count_tot_k + concentration)
    # calculate epsilon tpr
    e_tpr = torch.max(torch.abs(torch.log(torch.min(TPR, dim=1)[0])-torch.log(torch.max(TPR, dim=1)[0])))
    return torch.max(torch.tensor(0.0), e_tpr - base_fairness)


def get_counts(preds, groups, gold, num_classes, num_groups):
    count_pos_k = torch.zeros((num_classes, num_groups))
    count_tot_k = torch.zeros((num_classes, num_groups))
    for k in range(num_classes):
        for s in range(num_groups):
            TOT = (k==gold)&(s==groups)
            count_tot_k[k, s] = TOT.nonzero().size()[0]
            TP = (preds == gold) & (k == gold) & (s == groups)
            count_pos_k[k, s] = TP.nonzero().size()[0]
    return count_pos_k, count_tot_k            

#################################
## BINARY STOCASTIC FUNCS      ##
#################################

# Loss and optimizer
def fairness_loss(base_fairness, stochasticModel):
    # DF-based penalty term
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0)

    theta = (stochasticModel.countClass_hat + dirichletAlpha) / \
        (stochasticModel.countTotal_hat + concentrationParameter)
    epsilonClass = differentialFairnessBinaryOutcomeTrain(theta)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))
# Loss and optimizer


def sf_loss(base_fairness, stochasticModel):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0)
    population = sum(stochasticModel.countTotal_hat).detach()

    theta = (stochasticModel.countClass_hat + dirichletAlpha) / \
        (stochasticModel.countTotal_hat + concentrationParameter)
    alpha = (stochasticModel.countTotal_hat + dirichletAlpha) / \
        (population + concentrationParameter)
    gammaClass = subgroupFairnessTrain(theta, alpha)
    return torch.max(zeroTerm, (gammaClass-base_fairness))


def prule_loss(base_fairness, stochasticModel):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0)

    theta_minority = (stochasticModel.countClass_hat[0] + dirichletAlpha) / (
        stochasticModel.countTotal_hat[0] + concentrationParameter)
    theta_majority = (stochasticModel.countClass_hat[1] + dirichletAlpha) / (
        stochasticModel.countTotal_hat[1] + concentrationParameter)
    pruleClass = torch.min(theta_minority / theta_majority,
                           theta_majority / theta_minority) * 100.0
    return torch.max(zeroTerm, (base_fairness-pruleClass))


def differentialFairnessBinaryOutcomeTrain(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(
        len(probabilitiesOfPositive), dtype=torch.float)
    for i in range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0)  # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = torch.max(epsilon, torch.abs(torch.log(probabilitiesOfPositive[i])-torch.log(
                    probabilitiesOfPositive[j])))  # ratio of probabilities of positive outcome
                epsilon = torch.max(epsilon, torch.abs((torch.log(1-probabilitiesOfPositive[i]))-(
                    torch.log(1-probabilitiesOfPositive[j]))))  # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon  # DF per group
    epsilon = torch.max(epsilonPerGroup)  # overall DF of the algorithm
    return epsilon


def subgroupFairnessTrain(probabilitiesOfPositive, alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    spD = sum(probabilitiesOfPositive*alphaSP)
    gammaPerGroup = torch.zeros(
        len(probabilitiesOfPositive), dtype=torch.float)  # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i]*torch.abs(spD-probabilitiesOfPositive[i])
    gamma = torch.max(gammaPerGroup)  # overall SF of the algorithm
    return gamma


# stochastic count updates
def computeBatchCounts(protectedAttributes, intersectGroups, predictions):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S)
    # can be maintained correctly among different batches

    # compute counts for each intersectional group
    countsClassOne = torch.zeros((len(intersectGroups)), dtype=torch.float)
    countsTotal = torch.zeros((len(intersectGroups)), dtype=torch.float)
    for i in range(len(predictions)):
        index = np.where(
            (intersectGroups == protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne[index] = countsClassOne[index] + predictions[i]
    return countsClassOne, countsTotal


class stochasticCountModel(nn.Module):
    def __init__(self, no_of_groups, N, batch_size, rho=.1):
        super(stochasticCountModel, self).__init__()
        self.countClass_hat = torch.ones((no_of_groups))
        self.countTotal_hat = torch.ones((no_of_groups))

        self.countClass_hat = self.countClass_hat*(N/(batch_size*no_of_groups))
        self.countTotal_hat = self.countTotal_hat*(N/batch_size)
        self.N = N
        self.batch_size = batch_size
        self.rho = rho

    def forward(self, countClass_batch, countTotal_batch):
        self.countClass_hat = (1-self.rho)*self.countClass_hat + self.rho*(self.N/self.batch_size)*countClass_batch
        self.countTotal_hat = (1-self.rho)*self.countTotal_hat + \
            self.rho*(self.N/self.batch_size)*countTotal_batch
