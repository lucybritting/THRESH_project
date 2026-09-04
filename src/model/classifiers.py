"""The classifiers, as factories so every fold gets a fresh instance."""

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# factory dict to build new instance each call
CLASSIFIERS = {
        "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "balanced_rf": lambda: BalancedRandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        "catboost": lambda: CatBoostClassifier(n_estimators=100, random_seed=42, verbose=False),
    }
