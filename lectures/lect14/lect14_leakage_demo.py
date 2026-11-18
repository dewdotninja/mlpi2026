"""
Week 14 demo — explainability as a debugging tool
=================================================

A predictive-maintenance classifier scores 0.979 and should never be deployed.
Accuracy cannot show you why. One explanation can.

THE SCENARIO
    A machine shop wants to predict spindle failure from its maintenance database.
    Five of the six columns are genuine physical signals. The sixth,
    `work_order_raised`, is a clerical field: the maintenance system sets it once a
    technician has already been dispatched to the machine.

    It is in the historical database. It correlates almost perfectly with failure.
    And at the moment you actually need a prediction, it is empty -- because nobody
    has been sent yet.

HOW TO RUN
    python week14_leakage_demo.py                # the demo
    python week14_leakage_demo.py --pause        # stop between steps (for live teaching)
    python week14_leakage_demo.py --plot         # also save the importance figure
    python week14_leakage_demo.py --correlated   # bonus: the limits of an explanation

Needs only numpy, scikit-learn, and matplotlib (for --plot).
The data is synthetic and the leak is planted deliberately, so the numbers below
reproduce exactly on any machine.
"""
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

SEED, N = 0, 4000
FEATURES = ["vibration_rms", "temperature", "hours_since_service",
            "load", "spindle_age", "work_order_raised"]
LEAK = "work_order_raised"


# ----------------------------------------------------------------------------
# 1. the data
# ----------------------------------------------------------------------------
def make_data():
    """Synthetic maintenance records. Returns (X, y, feature_names)."""
    rng = np.random.default_rng(SEED)

    # --- the real physics: wear and load drive failure -----------------------
    hours_since_service = rng.uniform(0, 500, N)
    load = rng.uniform(0.2, 1.0, N)
    spindle_age = rng.uniform(0, 60, N)

    risk = ((hours_since_service / 500) * 0.8
            + load * 0.6
            + (spindle_age / 60) * 0.4
            + rng.normal(0, 0.35, N))
    fails = (risk > 1.15).astype(int)

    # --- sensors that genuinely reflect the machine's state, with noise ------
    vibration_rms = 0.5 + 0.9 * fails + rng.normal(0, 0.75, N)
    temperature = 40 + 6.0 * fails + rng.normal(0, 5.5, N)

    # --- THE LEAK -----------------------------------------------------------
    # A work order is raised for 97% of genuine failures and 2% of healthy
    # machines. Note the direction of causation: the failure causes the work
    # order, not the other way round.
    work_order_raised = np.where(
        fails == 1,
        (rng.random(N) < 0.97).astype(int),
        (rng.random(N) < 0.02).astype(int))

    cols = [vibration_rms, temperature, hours_since_service,
            load, spindle_age, work_order_raised]
    return np.column_stack(cols), fails, list(FEATURES)


# ----------------------------------------------------------------------------
# 2. helpers
# ----------------------------------------------------------------------------
def fit(Xtr, ytr):
    return RandomForestClassifier(n_estimators=300, random_state=SEED,
                                  n_jobs=2).fit(Xtr, ytr)


def importances(model, Xte, yte, names):
    """Permutation importance: shuffle one column, measure the accuracy you lose."""
    pi = permutation_importance(model, Xte, yte, n_repeats=20,
                                random_state=SEED, n_jobs=2)
    order = np.argsort(pi.importances_mean)[::-1]
    return [(names[i], pi.importances_mean[i], pi.importances_std[i]) for i in order]


def show(rows, highlight=LEAK):
    print(f"      {'feature':>22}   {'accuracy lost':>13}")
    for name, mean, std in rows:
        mark = "  <-- the leak" if name == highlight else ""
        print(f"      {name:>22}   {mean:+13.4f} +/- {std:.4f}{mark}")


def wait(enabled, prompt="      [enter to continue] "):
    if enabled:
        try:
            input(prompt)
        except EOFError:
            pass
    print()


# ----------------------------------------------------------------------------
# 3. the demo
# ----------------------------------------------------------------------------
def main(args):
    X, y, names = make_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    baseline = max(yte.mean(), 1 - yte.mean())
    print(f"\n{N} machine-months, failure rate {y.mean():.3f}, "
          f"majority-class baseline {baseline:.3f}")

    # ---- STEP 1: the model anyone would ship -------------------------------
    print("\n" + "=" * 70)
    print("STEP 1 -- the model you would have shipped")
    print("=" * 70)
    model = fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    print(f"\n      random forest, all six features")
    print(f"      test accuracy ........ {acc:.3f}")
    print(f"      majority baseline .... {baseline:.3f}")
    print("\n      That would pass every review you have ever sat in.")
    print("      Would you deploy it?  Decide before reading on.")
    wait(args.pause)

    # ---- STEP 2: what is it actually using? --------------------------------
    print("=" * 70)
    print("STEP 2 -- what is the model actually using?")
    print("=" * 70)
    print("\n   Permutation importance: shuffle one column at a time and measure")
    print("   how much accuracy collapses. Nothing is retrained.\n")
    rows = importances(model, Xte, yte, names)
    show(rows)

    leak_imp = dict((n, m) for n, m, _ in rows)[LEAK]
    best_real = max(m for n, m, _ in rows if n != LEAK)
    # computed from the rounded values printed above, so it matches the table by hand
    ratio = round(leak_imp, 4) / round(best_real, 4)
    print(f"\n      The leak is roughly {ratio:.0f}x more important than the best")
    print(f"      physical sensor ({leak_imp:+.3f} against {best_real:+.3f}).")
    print("      The model ignored the physics almost entirely.")
    print(f"\n      '{LEAK}' is set AFTER a technician is dispatched.")
    print("      At prediction time it is empty. The model learned to read the answer.")
    wait(args.pause)

    # ---- STEP 3: the honest model ------------------------------------------
    print("=" * 70)
    print("STEP 3 -- the model that can actually be deployed")
    print("=" * 70)
    keep = [i for i, n in enumerate(names) if n != LEAK]
    honest_names = [names[i] for i in keep]
    honest = fit(Xtr[:, keep], ytr)
    acc_h = honest.score(Xte[:, keep], yte)
    print(f"\n      test accuracy ........ {acc_h:.3f}   (was {acc:.3f})")
    print(f"      majority baseline .... {baseline:.3f}")
    print(f"\n      Worse on paper -- and {acc_h - baseline:+.3f} above the baseline,")
    print("      so the physical sensors really do carry predictive signal.\n")
    show(importances(honest, Xte[:, keep], yte, honest_names), highlight=None)
    print("\n      And the ordering is physically sensible: vibration first, then")
    print("      temperature, then hours since service. That ordering is itself")
    print("      a free sanity check you get for nothing.")
    wait(args.pause)

    # ---- the verdict --------------------------------------------------------
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"""
      Model A   accuracy {acc:.3f}   and cannot produce a prediction when one is needed
      Model B   accuracy {acc_h:.3f}   worse on paper, works at prediction time

      Ship B.

      Three things worth keeping:

      1. This is NOT overfitting. Model A generalises perfectly to held-out
         historical data -- the train/test split is completely honest. The split
         is fine; the feature is not.

      2. No accuracy-based metric could have caught this, on any split, because
         the leak is present in all the historical data. Only looking at WHAT THE
         MODEL USES exposes it.

      3. A model that scores worse can be the better model.
""")

    if args.plot:
        save_plot(rows, importances(honest, Xte[:, keep], yte, honest_names),
                  acc, acc_h, baseline)

    if args.correlated:
        correlated_trap()


# ----------------------------------------------------------------------------
# 4. bonus: the limits of an explanation (Week 14, slide 10)
# ----------------------------------------------------------------------------
def correlated_trap(_=None):
    """Slide 10, measured: redundant copies of a feature divide its credit."""
    print("=" * 70)
    print("BONUS -- when permutation importance LIES to you")
    print("=" * 70)
    print("""
   Nothing below changes what the model relies on. The work-order signal is still
   the entire basis of the prediction. All that changes is HOW MANY COLUMNS record
   it -- as happens when a second maintenance system logs the same event.

   Each column is an independent, 95%-accurate observation of the same work order.
""")
    X0, y, names0 = make_data()
    truth = X0[:, names0.index(LEAK)].copy()
    base_cols = [X0[:, i] for i, n in enumerate(names0) if n != LEAK]
    base_names = [n for n in names0 if n != LEAK]

    print(f"      {'copies':>7}{'accuracy':>11}   {'importance of each copy':<34}{'best sensor':>12}")
    for k in (1, 2, 4):
        rng = np.random.default_rng(SEED)
        cols, names = list(base_cols), list(base_names)
        for j in range(k):
            cols.append(np.where(rng.random(N) < 0.95, truth, 1 - truth))
            names.append(f"work_order_{j + 1}")
        Xtr, Xte, ytr, yte = train_test_split(np.column_stack(cols), y,
                                              test_size=0.3, random_state=SEED, stratify=y)
        model = fit(Xtr, ytr)
        d = dict((n, m) for n, m, _s in importances(model, Xte, yte, names))
        leaks = sorted((v for n, v in d.items() if n.startswith("work_order")), reverse=True)
        sensor = max(v for n, v in d.items() if not n.startswith("work_order"))
        shown = ", ".join(f"{v:+.4f}" for v in leaks)
        print(f"      {k:>7}{model.score(Xte, yte):>11.3f}   {shown:<34}{sensor:>12.4f}")

    print("""
      Read the accuracy column first: it does not move. The model is just as
      dependent on the work-order signal with four copies as with one.

      Now read across. With one column the leak is unmissable. With two, each copy
      reports about a quarter of that. With four, the largest copy has sunk to the
      level of an ordinary sensor -- and an engineer scanning this table would
      conclude that nothing in it matters very much.

      The explanation did not become wrong. It answered a different question than
      the one you were asking: it reports what happens when you remove ONE column,
      and the answer is "nothing", because the others still carry the signal.

      THE LESSON: permutation importance cannot tell you what a GROUP of correlated
      features is doing. When you suspect a group, drop the whole group and refit.
      An explanation is evidence, not proof.
""")


def save_plot(leaky_rows, honest_rows, acc, acc_h, baseline, path="week14_importance.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5), dpi=150)
    for ax, rows, title in [
        (axes[0], leaky_rows, f"Model A - accuracy {acc:.3f}"),
        (axes[1], honest_rows, f"Model B - accuracy {acc_h:.3f}"),
    ]:
        names = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        errs = [r[2] for r in rows][::-1]
        cols = ["#C0392B" if n == LEAK else "#3E5C76" for n in names]
        ax.barh(np.arange(len(names)), vals, xerr=errs, color=cols, height=0.62,
                error_kw=dict(ecolor="#55606E", lw=1.2, capsize=3))
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlim(0, 0.42)
        ax.set_xlabel("accuracy lost when this column is shuffled", fontsize=10)
        ax.set_title(title, fontsize=13, weight="bold", color="#1B2A41")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"      figure written to {path}\n")


# ----------------------------------------------------------------------------
# TRY THIS YOURSELF
# ----------------------------------------------------------------------------
#  1. In make_data, weaken the leak: change 0.97 to 0.60, so a work order is
#     raised for only 60% of failures. Does permutation importance still flag it?
#     At what point does the leak stop being visible?
#
#  2. Run with --correlated. Then fix the diagnosis: drop BOTH work-order columns
#     and refit. What accuracy do you get, and what does that tell you that the
#     importance table could not?
#
#  3. Delete `vibration_rms` from the honest model. Does `temperature` become
#     more important? Explain the result in terms of what the two sensors measure.
#
#  4. Hardest: invent a leak for YOUR capstone dataset -- a column that would be
#     available in your historical data but not at prediction time. If you cannot
#     think of one, check again; most real datasets have at least one.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pause", action="store_true", help="stop between steps")
    ap.add_argument("--plot", action="store_true", help="save the importance figure")
    ap.add_argument("--correlated", action="store_true",
                    help="bonus: show how correlated inputs fool the explanation")
    main(ap.parse_args())
