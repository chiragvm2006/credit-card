"""Credit card recommendation engine."""

import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

CATEGORIES = ["travel", "food", "shopping", "fuel", "general", "online"]

# Features the model matches on, and how much each matters.
# Reward rate is weighted highest because it's recurring value — a one-off
# joining bonus shouldn't outrank a card you'll earn more on every month.
FEATURES = ["reward_rate_top_category", "annual_fee", "joining_bonus_value", "lounge_access"]
FEATURE_WEIGHTS = [3.0, 2.0, 0.5, 0.5]

# Resolve the CSV path relative to this file, not the working directory —
# the working directory Streamlit Cloud runs from isn't guaranteed to be
# the project root, so a bare "data/cards.csv" can silently break there
# even though it works fine locally.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CARDS_PATH = os.path.join(_HERE, "data", "cards.csv")


def load_cards(path=None):
    return pd.read_csv(path or DEFAULT_CARDS_PATH)


def find_eligible(cards, income, credit_score, max_fee=None):
    """Keep only cards the user can actually get approved for."""
    ok = cards[
        (cards.min_income <= income) & (cards.min_credit_score <= credit_score)
    ]
    if max_fee is not None:
        ok = ok[ok.annual_fee <= max_fee]
    return ok.reset_index(drop=True)


def nearest_gap(cards, income, credit_score, n=3):
    """When nothing qualifies, show the closest cards and what's missing."""
    gap = ((cards.min_income - income).clip(lower=0) / 10000
           + (cards.min_credit_score - credit_score).clip(lower=0) / 50)
    closest = cards.assign(gap=gap).nsmallest(n, "gap")

    top = closest.iloc[0]
    needs = []
    if top.min_income > income:
        needs.append(f"₹{top.min_income - income:,.0f} more monthly income")
    if top.min_credit_score > credit_score:
        needs.append(f"{top.min_credit_score - credit_score:.0f} more credit score points")

    msg = f"You're closest to {top.card_name} — you need {' and '.join(needs)}."
    return closest.drop(columns="gap"), msg


def recommend(cards, income, credit_score, category, second=None, max_fee=None, n=3):
    """
    Rank eligible cards using k-Nearest Neighbours.

    We build an "ideal card" from the user's preferences, then ask the model
    which real cards sit closest to it in feature space. Cards matching the
    user's spending category get a boost, since that's the strongest signal
    of whether rewards actually get used.
    """
    eligible = find_eligible(cards, income, credit_score, max_fee)
    if eligible.empty:
        return None, eligible

    X = eligible[FEATURES]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X) * FEATURE_WEIGHTS

    # The ideal card: best rewards, no fee, big bonus, lounge access
    ideal_raw = pd.DataFrame([[X.reward_rate_top_category.max(), 0,
                               X.joining_bonus_value.max(), 1]], columns=FEATURES)
    ideal = scaler.transform(ideal_raw) * FEATURE_WEIGHTS

    knn = NearestNeighbors(n_neighbors=len(eligible), metric="euclidean")
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(ideal)

    # Convert distance to a 0-1 similarity score
    d = distances[0]
    similarity = 1 - (d / d.max()) if d.max() > 0 else pd.Series(1.0, index=range(len(d)))

    scored = eligible.iloc[indices[0]].copy()
    scored["similarity"] = similarity

    # Category match: full credit for top category, half for second
    scored["category_fit"] = 0.1
    scored.loc[scored.top_category == category, "category_fit"] = 1.0
    if second:
        scored.loc[scored.top_category == second, "category_fit"] = 0.5

    scored["score"] = 0.5 * scored.category_fit + 0.5 * scored.similarity

    return scored.nlargest(n, "score").reset_index(drop=True), eligible


def breakeven(fee, reward_rate):
    """Monthly spend needed for rewards to cover the annual fee."""
    if fee == 0 or reward_rate == 0:
        return 0
    return fee / (reward_rate / 100) / 12


def explain(card, category, avg_rate):
    """Plain-English reason for the recommendation."""
    if card.category_fit == 1.0:
        why = f"{category.title()} is your top spending area and this card earns {card.reward_rate_top_category:.1f}% there"
    elif card.category_fit == 0.5:
        why = f"this card fits your secondary spending category, earning {card.reward_rate_top_category:.1f}%"
    else:
        why = f"it scores well on rewards and value overall, earning {card.reward_rate_top_category:.1f}%"

    compare = "above" if card.reward_rate_top_category > avg_rate else "around"
    text = f"We recommend {card.card_name} because {why} — {compare} the {avg_rate:.1f}% average across cards you qualify for."

    if card.annual_fee == 0:
        text += " It has no annual fee, so all rewards are pure gain."
    else:
        be = breakeven(card.annual_fee, card.reward_rate_top_category)
        text += f" The ₹{card.annual_fee:,.0f} fee pays for itself once you spend ₹{be:,.0f}/month in this category."

    if card.joining_bonus_value > 0:
        text += f" You also get a ₹{card.joining_bonus_value:,.0f} welcome bonus."

    return text


if __name__ == "__main__":
    cards = load_cards()
    print(f"{len(cards)} cards loaded\n")

    tests = [
        ("Traveller, high income", 300000, 800, "travel", "shopping", None),
        ("Low income, no fee", 20000, 700, "general", None, 0),
        ("Foodie, mid income", 45000, 720, "food", "online", None),
    ]

    for name, inc, cs, cat, sec, fee in tests:
        top, eligible = recommend(cards, inc, cs, cat, sec, fee)
        avg = eligible.reward_rate_top_category.mean()
        print(f"--- {name} ---")
        for _, c in top.iterrows():
            print(f"  {c.card_name} ({c.bank}) — {c.score:.0%}")
        print(f"  → {explain(top.iloc[0], cat, avg)}\n")
