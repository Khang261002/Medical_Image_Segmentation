import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = (SR > threshold)
    GT = (GT > threshold)

    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold)
    GT = (GT > threshold)

    # TP : True Positive
    # FN : False Negative
    TP = (SR & GT)
    FN = ((~SR) & GT)

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE

def get_specificity(SR, GT, threshold=0.5):
    SR = (SR > threshold)
    GT = (GT > threshold)

    # TN : True Negative
    # FP : False Positive
    TN = ((~SR) & (~GT))
    FP = (SR & (~GT))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP

def get_precision(SR, GT, threshold=0.5):
    SR = (SR > threshold)
    GT = (GT > threshold)

    # TP : True Positive
    # FP : False Positive
    TP = (SR & GT)
    FP = (SR & (~GT))

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC

def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = (2 * SE * PC) / (SE + PC + 1e-6)

    return F1

def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold)
    GT = (GT > threshold)
    
    Inter = torch.sum(SR & GT)
    Union = torch.sum(SR | GT)

    JS = float(Inter) / (float(Union) + 1e-6)
    
    return JS

def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold)
    GT = (GT > threshold)

    Inter = torch.sum(SR & GT)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC
