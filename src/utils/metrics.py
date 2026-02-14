from torchmetrics import MeanSquaredError
from torchmetrics.classification import (
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)



class RMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(squared=False)

    def compute(self):
        return super().compute()


METRIC_MAPPING = {
    "multiclass_accuracy": MulticlassAccuracy,
    "multiclass_f1": MulticlassF1Score,
    "rmse": RMSE,
}
