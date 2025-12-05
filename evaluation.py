import torch

# SR : Segmentation Result
# GT : Ground Truth

_eps = 1e-6

def _binarize_inputs(SR, GT, threshold=0.5):
    SRb = (SR > threshold)
    GTb = (GT > threshold)
    return SRb, GTb

def get_accuracy(SR, GT, threshold=0.5):
    SRb, GTb = _binarize_inputs(SR, GT, threshold)

    corr = torch.sum(SRb==GTb)
    tensor_size = SRb.size(0)*SRb.size(1)*SRb.size(2)*SRb.size(3)
    AC = float(corr) / float(tensor_size)

    return AC

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SRb, GTb = _binarize_inputs(SR, GT, threshold)

    # TP : True Positive
    # FN : False Negative
    TP = (SRb & GTb)
    FN = ((~SRb) & GTb)

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + _eps)

    return SE

def get_specificity(SR, GT, threshold=0.5):
    SRb, GTb = _binarize_inputs(SR, GT, threshold)

    # TN : True Negative
    # FP : False Positive
    TN = ((~SRb) & (~GTb))
    FP = (SRb & (~GTb))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + _eps)

    return SP

def get_precision(SR, GT, threshold=0.5):
    SRb, GTb = _binarize_inputs(SR, GT, threshold)

    # TP : True Positive
    # FP : False Positive
    TP = (SRb & GTb)
    FP = (SRb & (~GTb))

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + _eps)

    return PC

def get_F1(SR, GT, threshold=0.5):
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = (2 * SE * PC) / (SE + PC + _eps)

    return F1

def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SRb, GTb = _binarize_inputs(SR, GT, threshold)
    
    Inter = torch.sum(SRb & GTb)
    Union = torch.sum(SRb | GTb)

    JS = float(Inter) / (float(Union) + _eps)
    
    return JS

def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SRb, GTb = _binarize_inputs(SR, GT, threshold)

    Inter = torch.sum(SRb & GTb)
    DC = float(2 * Inter) / (float(torch.sum(SRb) + torch.sum(GTb)) + _eps)

    return DC
