from import_library import *

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, state_dict):
        improved = (
            self.best_score is None or
            (self.mode == 'max' and score > self.best_score + self.min_delta) or
            (self.mode == 'min' and score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in state_dict.items()}
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop
