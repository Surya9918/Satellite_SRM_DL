"""Early stopping tracking to prevent overfitting on validation metrics."""
from typing import Optional

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = current_metric
            return False

        if self.mode == "max":
            improved = current_metric > (self.best_score + self.min_delta)
        else:
            improved = current_metric < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
