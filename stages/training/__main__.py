import warnings

from sklearn.exceptions import ConvergenceWarning

from stages.training.core import run

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")
    run()
