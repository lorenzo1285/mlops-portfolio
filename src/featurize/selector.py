from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LinearRegression


@dataclass
class SelectionResult:
    selected_cols: list[str]
    dropped_cols: list[str]
    scores: dict[str, float] | None


class FeatureSelector:
    """Select features from preprocessed arrays using one of four strategies.

    Fits exclusively on training data; applies the same mask to val and test.

    Public interface
    ----------------
    fit(X, y, feature_names) → FeatureSelector
    transform(X) → tuple[np.ndarray, SelectionResult]
    """

    def __init__(self, method: str, n_features: int, threshold: float) -> None:
        self._method = method
        self._n_features = n_features
        self._threshold = threshold
        self._mask: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._scores: dict[str, float] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        feature_names: list[str],
    ) -> FeatureSelector:
        self._feature_names = list(feature_names)

        if self._method == "none":
            self._mask = np.ones(X.shape[1], dtype=bool)

        elif self._method == "mutual_info":
            sel = SelectKBest(mutual_info_classif, k=self._n_features)
            sel.fit(X, y)
            self._mask = sel.get_support()
            self._scores = dict(zip(feature_names, sel.scores_))

        elif self._method == "rfe":
            sel = RFE(
                RandomForestClassifier(n_estimators=100, random_state=0),
                n_features_to_select=self._n_features,
            )
            sel.fit(X, y)
            self._mask = sel.get_support()

        elif self._method == "correlation":
            self._mask = self._correlation_mask(X)

        elif self._method == "vif":
            self._mask = self._vif_mask(X)

        else:
            raise ValueError(f"Unknown feature selection method: {self._method!r}")

        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, SelectionResult]:
        if self._mask is None:
            raise RuntimeError("Call fit() before transform()")

        selected = [n for n, m in zip(self._feature_names, self._mask) if m]
        dropped = [n for n, m in zip(self._feature_names, self._mask) if not m]
        scores = (
            {k: v for k, v in self._scores.items() if k in selected}
            if self._scores else None
        )

        return X[:, self._mask], SelectionResult(
            selected_cols=selected,
            dropped_cols=dropped,
            scores=scores,
        )

    def _correlation_mask(self, X: np.ndarray) -> np.ndarray:
        mask = np.ones(X.shape[1], dtype=bool)
        corr = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(corr, 0.0)
        while True:
            active = np.where(mask)[0]
            if len(active) <= 1:
                break
            sub = corr[np.ix_(active, active)]
            if sub.max() <= self._threshold:
                break
            i_loc, j_loc = np.unravel_index(sub.argmax(), sub.shape)
            mask[active[j_loc]] = False
        return mask

    def _vif_mask(self, X: np.ndarray) -> np.ndarray:
        mask = np.ones(X.shape[1], dtype=bool)
        cols = list(range(X.shape[1]))
        while True:
            X_sub = X[:, cols]
            vifs = []
            for i in range(X_sub.shape[1]):
                x_i = X_sub[:, i]
                x_rest = np.delete(X_sub, i, axis=1)
                r2 = LinearRegression().fit(x_rest, x_i).score(x_rest, x_i)
                vifs.append(1.0 / (1.0 - r2) if r2 < 1.0 else float("inf"))
            if max(vifs) <= self._threshold:
                break
            drop = vifs.index(max(vifs))
            mask[cols[drop]] = False
            cols.pop(drop)
        return mask
