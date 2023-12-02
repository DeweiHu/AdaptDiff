import numpy as np
from sklearn.metrics import roc_auc_score
from hausdorff import hausdorff_distance

class SegmentStat:

    def __init__(self, prediction, groundtruth):
        assert prediction.size == groundtruth.size, "Array size mismatch."

        difference = np.abs(prediction - groundtruth).astype(np.bool_)
        pred_bool = np.asarray(prediction).astype(np.bool_)
        gt_bool = np.asarray(groundtruth).astype(np.bool_)

        self.pred = prediction
        self.gt = groundtruth

        self.tp = np.sum(np.logical_and(pred_bool, gt_bool))
        self.tn = np.sum(np.logical_and(np.logical_not(pred_bool),
                                        np.logical_not(gt_bool))
                                        )
        self.fp = np.sum(np.logical_and(difference, 
                                        np.logical_not(gt_bool))
                                        )
        self.fn = np.sum(np.logical_and(difference, gt_bool))
    
    # recall / sensitivity
    def recall(self, ):
        return self.tp / (self.tp + self.fn)
    
    # precision / positive predictive value
    def precision(self, ):
        return self.tp / (self.tp + self.fp)
    
    # accuracy
    def accuracy(self, ):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    # specificity
    def specificity(self, ):
        return self.tn / (self.tn + self.fp)
    
    # Jaccard index / IoU
    def IoU(self, ):
        return self.tp / (self.tp + self.fp + self.fn)

    # Dice coefficient
    def dice(self, ):
        return 2. * self.tp / (2. * self.tp + self.fn + self.fp)

    # AUC
    def auc(self, ):
        return roc_auc_score(self.gt, self.pred)
    
    # hausdorff
    def hd_distance(self, ):
        return hausdorff_distance(self.pred, self.gt)
    
