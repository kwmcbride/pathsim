#########################################################################################
##
##                    LOCAL SENSITIVITY & IDENTIFIABILITY ANALYSIS
##                                 (sensitivity.py)
##
##                                  Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np


# CLASS =================================================================================

class SensitivityResult:
    """Result of a local sensitivity and practical identifiability analysis.

    Computed at a specific parameter vector ``x*`` (typically the optimum
    returned by :meth:`ParameterEstimator.fit`).  All statistics derive from
    the weighted Jacobian **J** of the residual vector with respect to the
    parameters (residuals are already divided by ``sigma`` inside
    :meth:`ParameterEstimator.residuals`).

    Parameters
    ----------
    jacobian : np.ndarray, shape (n_residuals, n_params)
        Weighted Jacobian matrix ``∂r_i/∂θ_j`` at ``x*``.
    param_names : list of str
        Parameter names in the same order as columns of ``jacobian``.
    param_values : np.ndarray, shape (n_params,)
        Model-space parameter values at ``x*``.

    Attributes
    ----------
    jacobian : np.ndarray
        Weighted Jacobian, shape ``(n_residuals, n_params)``.
    param_names : list of str
    param_values : np.ndarray
    fim : np.ndarray
        Fisher Information Matrix ``Jᵀ J``, shape ``(n_params, n_params)``.
    covariance : np.ndarray
        Parameter covariance estimate ``pinv(FIM)``,
        shape ``(n_params, n_params)``.
    std_errors : np.ndarray
        Approximate standard errors ``√diag(covariance)``,
        shape ``(n_params,)``.  Expressed in optimizer space; for
        untransformed parameters this equals model space.
    correlation : np.ndarray
        Parameter correlation matrix, shape ``(n_params, n_params)``.
        Off-diagonal values near ±1 indicate strongly correlated
        (potentially unidentifiable) parameter pairs.
    eigenvalues : np.ndarray
        Eigenvalues of the FIM in descending order.  Small eigenvalues
        correspond to poorly constrained parameter directions.
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns), shape ``(n_params, n_params)``.
    condition_number : float
        Ratio of the largest to smallest *positive* FIM eigenvalue.
        Values above ~1e6 indicate practical non-identifiability.

    Notes
    -----
    The analysis is *local* — it linearises around ``x*``.  Near a flat
    optimum or with highly correlated parameters the results may be
    unreliable.  Use the condition number and eigenvalue spectrum together
    rather than relying on a single threshold.
    """

    def __init__(
        self,
        jacobian: np.ndarray,
        param_names: list,
        param_values: np.ndarray,
    ):
        self.jacobian    = np.asarray(jacobian, dtype=float)
        self.param_names = list(param_names)
        self.param_values = np.asarray(param_values, dtype=float)

        n_p = len(param_names)

        # Fisher Information Matrix
        self.fim = self.jacobian.T @ self.jacobian          # (n_p, n_p)

        # Covariance — pseudo-inverse handles rank-deficient FIM gracefully
        self.covariance = np.linalg.pinv(self.fim)         # (n_p, n_p)

        # Standard errors
        self.std_errors = np.sqrt(
            np.maximum(np.diag(self.covariance), 0.0)
        )                                                   # (n_p,)

        # Correlation matrix
        corr = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(n_p):
                denom = self.std_errors[i] * self.std_errors[j]
                if denom > 0.0:
                    corr[i, j] = self.covariance[i, j] / denom
                elif i == j:
                    corr[i, j] = 1.0
        self.correlation = corr                             # (n_p, n_p)

        # Eigendecomposition of FIM (symmetric → use eigh)
        eigenvalues, eigenvectors = np.linalg.eigh(self.fim)
        idx = np.argsort(eigenvalues)[::-1]                # descending
        self.eigenvalues  = eigenvalues[idx]               # (n_p,)
        self.eigenvectors = eigenvectors[:, idx]           # (n_p, n_p)

        # Condition number: λ_max / λ_min over positive eigenvalues only.
        # If fewer positive eigenvalues than parameters the FIM is rank-deficient
        # (non-identifiable); return inf in that case.
        pos_ev = self.eigenvalues[self.eigenvalues > 0.0]
        if len(pos_ev) == len(self.eigenvalues) and len(pos_ev) >= 2:
            self.condition_number = float(pos_ev[0] / pos_ev[-1])
        elif len(pos_ev) == 1 and len(self.eigenvalues) == 1:
            self.condition_number = 1.0          # single parameter, fully constrained
        else:
            self.condition_number = np.inf       # rank-deficient or empty


    # DISPLAY ===========================================================================

    def display(self) -> None:
        """Print a formatted sensitivity and identifiability summary.

        Prints a table of parameter values, standard errors, and relative
        errors, followed by the FIM condition number and any highly
        correlated parameter pairs.
        """
        n_p  = len(self.param_names)
        W    = 72
        line = "=" * W
        dash = "-" * W

        print(line)
        print("  Sensitivity & Identifiability Analysis")
        print(line)

        # Column header
        print(f"  {'Parameter':<22} {'Value':>12} {'Std Error':>12} "
              f"{'Rel Error':>10}  {'OK?':>4}")
        print(dash)

        for i, name in enumerate(self.param_names):
            val = self.param_values[i]
            se  = self.std_errors[i]

            if abs(val) > 1e-15 and np.isfinite(se):
                rel = se / abs(val)
                rel_str = f"{rel * 100:.2f}%"
            else:
                rel = np.inf
                rel_str = "N/A"

            flag = "✓" if (np.isfinite(rel) and rel < 0.5) else "✗"
            print(f"  {name:<22} {val:>12.4g} {se:>12.4g} "
                  f"{rel_str:>10}  {flag:>4}")

        print(dash)

        # Condition number
        cn = self.condition_number
        if cn < 1e3:
            cn_label = "excellent"
        elif cn < 1e6:
            cn_label = "acceptable"
        else:
            cn_label = "POOR — parameters may not be uniquely identifiable"

        print(f"\n  FIM condition number : {cn:.3g}  ({cn_label})")

        # Highly correlated pairs
        pairs = [
            (i, j, self.correlation[i, j])
            for i in range(n_p)
            for j in range(i + 1, n_p)
            if abs(self.correlation[i, j]) > 0.90
        ]

        if pairs:
            print(f"\n  Highly correlated pairs (|r| > 0.90):")
            for i, j, r in pairs:
                print(f"    {self.param_names[i]} ↔ {self.param_names[j]}"
                      f"  :  r = {r:+.3f}")
        else:
            print("\n  No highly correlated parameter pairs  (|r| ≤ 0.90)")

        print(line)


    # PLOT ==============================================================================

    def plot(self, *, figsize: tuple = (11, 4.5)):
        """Plot the correlation matrix heatmap and FIM eigenvalue spectrum.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray of matplotlib.axes.Axes, shape (2,)
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        n_p = len(self.param_names)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ── Correlation heatmap ────────────────────────────────────────────
        ax = axes[0]
        norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        im   = ax.imshow(self.correlation, cmap="RdBu_r", norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, label="Correlation")

        ax.set_xticks(range(n_p))
        ax.set_yticks(range(n_p))
        ax.set_xticklabels(self.param_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.param_names, fontsize=9)
        ax.set_title("Parameter Correlation Matrix")

        for i in range(n_p):
            for j in range(n_p):
                v = self.correlation[i, j]
                color = "white" if abs(v) > 0.65 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        # ── FIM eigenvalue spectrum ────────────────────────────────────────
        ax2   = axes[1]
        ev    = self.eigenvalues
        pos   = ev > 0.0
        colors = ["steelblue" if p else "salmon" for p in pos]
        ax2.bar(range(len(ev)), np.where(pos, ev, np.abs(ev)), color=colors)

        # Log scale when values span more than 2 orders of magnitude
        pos_vals = ev[pos]
        if len(pos_vals) > 1 and pos_vals.max() / pos_vals.min() > 100.0:
            ax2.set_yscale("log")

        ax2.set_xticks(range(len(ev)))
        ax2.set_xticklabels([f"λ{i + 1}" for i in range(len(ev))], fontsize=9)
        ax2.set_xlabel("Eigendirection")
        ax2.set_ylabel("Eigenvalue magnitude")
        ax2.set_title("FIM Eigenvalue Spectrum")
        ax2.grid(True, axis="y", alpha=0.3)

        if not all(pos):
            from matplotlib.patches import Patch
            ax2.legend(handles=[
                Patch(facecolor="steelblue", label="Positive"),
                Patch(facecolor="salmon",    label="Non-positive (not identifiable)"),
            ], fontsize=8)

        fig.suptitle("Sensitivity & Identifiability Analysis", fontweight="bold")
        plt.tight_layout()
        return fig, axes
