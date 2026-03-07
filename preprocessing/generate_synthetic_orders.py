import ast
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- helpers ----------
def safe_list_parse(x):
    """Parse strings like '[1,2,3]' safely; return None if not parseable."""
    if pd.isna(x):
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    try:
        return list(ast.literal_eval(s))
    except Exception:
        return None


def build_quantity_sampler(items_df: pd.DataFrame):
    """
    Returns a dict: item_code -> (values, probs)
    Uses item_quantity_order_unique and item_quantity_order_unique_frequency if available.
    """
    samplers = {}
    for _, r in items_df.iterrows():
        code = int(r["item_code"])
        vals = safe_list_parse(r.get("item_quantity_order_unique"))
        freqs = safe_list_parse(r.get("item_quantity_order_unique_frequency"))

        if vals and freqs and len(vals) == len(freqs) and sum(freqs) > 0:
            probs = np.array(freqs, dtype=float)
            probs = probs / probs.sum()
            samplers[code] = (np.array(vals, dtype=int), probs)
        else:
            # fallback: use mean/std if unique list not available
            mu = r.get("item_quantity_order_mean", 1.0)
            sd = r.get("item_quantity_order_std", 0.0)
            samplers[code] = ("normal_fallback", (float(mu), float(sd)))
    return samplers


def sample_quantity(rng: np.random.Generator, item_code: int, samplers: dict):
    s = samplers.get(item_code)
    if s is None:
        return 1
    if isinstance(s[0], str) and s[0] == "normal_fallback":
        mu, sd = s[1]
        if sd <= 0:
            return max(1, int(round(mu)))
        q = int(round(rng.normal(mu, sd)))
        return max(1, q)
    vals, probs = s
    return int(rng.choice(vals, p=probs))


def normalize_rows(mat: np.ndarray, eps=1e-12):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return mat / row_sums


# ---------- main synthetic generator ----------
def main():
    # ====== CHANGE PATHS HERE if needed ======
    ITEMS_CSV = Path("items_dictionary.csv")
    ASSIGN_ORDER_CSV = Path("assign_order.csv")
    OUT_CSV = Path("synthetic_orders.csv")

    # How many synthetic orders to generate?
    N_SYNTH_ORDERS = 5000

    # Random seed for reproducibility
    SEED = 42
    rng = np.random.default_rng(SEED)

    # ---------- load data ----------
    items = pd.read_csv(ITEMS_CSV)

    # assign_order.csv is semicolon separated in your file
    orders = pd.read_csv(ASSIGN_ORDER_CSV, sep=";")

    # Your assign_order uses item_id (looks like 1..n), while items_dictionary uses item_code (big numbers).
    # If item_id is NOT the same as item_code, you must provide a mapping.
    # Here we assume item_id corresponds to row index or a separate mapping already exists.
    # We'll try both: if item_id values are mostly NOT in item_code, we fallback to using item_id as "item_key".
    item_codes = set(items["item_code"].astype(int).tolist())
    order_item_vals = set(orders["item_id"].astype(int).tolist())

    use_item_code = len(order_item_vals.intersection(item_codes)) / max(1, len(order_item_vals)) > 0.5

    if use_item_code:
        item_key_col_items = "item_code"
        item_key_col_orders = "item_id"
        print("Detected: order item_id mostly matches items_dictionary item_code. Using item_code.")
        all_items = sorted(item_codes)
    else:
        # Use item_id as the universe, and only keep items that appear in order history
        # (or you can expand this with your own mapping later).
        print("Detected: order item_id does NOT match items_dictionary item_code well.")
        print("Using item_id from assign_order as item keys for affinity + generation.")
        all_items = sorted(order_item_vals)
        item_key_col_items = None  # no quantity sampler from dictionary guaranteed
        item_key_col_orders = "item_id"

    # ---------- basket size distribution from real history ----------
    basket_sizes = orders.groupby("order_id")[item_key_col_orders].nunique()
    size_values = basket_sizes.value_counts().sort_index()
    size_probs = (size_values / size_values.sum()).to_dict()

    # restrict sizes a bit (optional safety)
    possible_sizes = np.array(list(size_probs.keys()), dtype=int)
    possible_probs = np.array(list(size_probs.values()), dtype=float)
    possible_probs = possible_probs / possible_probs.sum()

    # ---------- item popularity (base probability) ----------
    item_counts = orders[item_key_col_orders].value_counts()
    pop = pd.Series(0.0, index=all_items)
    pop.loc[item_counts.index.astype(int)] = item_counts.values.astype(float)
    pop = pop + 1.0  # smoothing so unseen items still possible (if any)
    pop = pop / pop.sum()
    pop_probs = pop.values

    # ---------- build affinity matrix from co-occurrence ----------
    # co-occurrence counts: C[i,j] = number of orders containing both i and j
    idx = {it: k for k, it in enumerate(all_items)}
    C = np.zeros((len(all_items), len(all_items)), dtype=float)

    # build order -> items sets
    grouped = orders.groupby("order_id")[item_key_col_orders].apply(lambda s: sorted(set(s.astype(int).tolist())))
    for items_in_order in grouped:
        for a in items_in_order:
            ia = idx.get(a)
            if ia is None:
                continue
            for b in items_in_order:
                ib = idx.get(b)
                if ib is None:
                    continue
                if ia != ib:
                    C[ia, ib] += 1.0

    # Convert to conditional probabilities P(j | i)
    # Add smoothing to avoid zero rows
    P = C + 0.1
    P = normalize_rows(P)

    # ---------- quantity sampler ----------
    quantity_samplers = {}
    if use_item_code:
        quantity_samplers = build_quantity_sampler(items)

    # ---------- generate synthetic orders ----------
    rows = []
    start_order_id = int(orders["order_id"].max()) + 1 if len(orders) else 1

    # simple arrival times: increasing integers (you can change to Poisson inter-arrivals)
    arrival = 0

    for o in range(N_SYNTH_ORDERS):
        order_id = start_order_id + o
        basket_size = int(rng.choice(possible_sizes, p=possible_probs))
        basket_size = max(1, basket_size)

        # pick first item from popularity
        first = int(rng.choice(all_items, p=pop_probs))
        basket = [first]

        # add the rest using affinity
        while len(basket) < basket_size:
            last = basket[-1]
            last_idx = idx[last]
            candidates_prob = P[last_idx].copy()

            # avoid duplicates
            for x in basket:
                candidates_prob[idx[x]] = 0.0
            s = candidates_prob.sum()
            if s <= 0:
                # fallback to popularity if affinity collapses
                nxt = int(rng.choice(all_items, p=pop_probs))
            else:
                candidates_prob = candidates_prob / s
                nxt = int(rng.choice(all_items, p=candidates_prob))
            if nxt not in basket:
                basket.append(nxt)

        # quantities
        for item in basket:
            if use_item_code:
                qty = sample_quantity(rng, item, quantity_samplers)
            else:
                # no reliable dictionary mapping -> default qty ~ 1..3
                qty = int(rng.integers(1, 4))
            rows.append(
                {
                    "order_id": order_id,
                    "item_id": item,
                    "item_quantity": qty,
                    "order_arrival": arrival,
                }
            )

        arrival += 1

    syn = pd.DataFrame(rows)
    syn.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved synthetic orders to: {OUT_CSV.resolve()}")
    print(syn.head())


if __name__ == "__main__":
    main()