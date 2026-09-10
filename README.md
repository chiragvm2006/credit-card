# 💳 Credit Card Finder

Most people pick a credit card based on brand recognition or whatever their bank pushes — not based on how they actually spend. The result is paying an annual fee for rewards you never earn.

This tool asks 5 questions and recommends a card that matches your real spending, with a plain-English explanation of why.

**[→ Try the live demo](#)** *(add your Streamlit Cloud link here)*

![screenshot](screenshot.png)

---

## How it works

**1. Filter by eligibility.** Cards where `min_income` or `min_credit_score` exceed the user's are removed first. A recommendation you can't get approved for isn't a recommendation.

**2. Rank with k-Nearest Neighbours.** An "ideal card" vector is built (best rewards, no fee, biggest bonus, lounge access) and the model finds which real cards sit closest to it. Features are weighted so recurring reward rate matters more than a one-off joining bonus.

**3. Blend with category fit.** KNN similarity is combined 50/50 with how well the card's reward category matches the user's spending — because a 10% cashback card is worthless if it's 10% on something you never buy.

**4. Explain the pick.** Every recommendation includes the reward rate vs. the average card the user qualifies for, and the monthly spend needed for the fee to break even (`fee ÷ reward_rate ÷ 12`).

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Test the engine on its own:

```bash
python recommender.py
```

---

## Example output

| User | Top recommendation | Why |
|---|---|---|
| Traveller, ₹3L/mo | IRCTC SBI Platinum | 10% on travel vs 3.7% average; ₹500 fee breaks even at ₹417/mo |
| ₹20k/mo, no fee | Axis My Zone Easy | Free card, 2% on general spends |
| Foodie, ₹45k/mo | Swiggy HDFC | 10% on food vs 3.8% average |

---

## Data

102 cards across 5 banks (SBI, HDFC, Axis, ICICI, IDFC FIRST), compiled from published bank pages.

**Known gaps:** eligibility criteria aren't published by most Indian banks, so `min_income` and `min_credit_score` are estimated from fee tier. ICICI and IDFC are under-represented. Reward rates are simplified to a single headline percentage per card.

---

## What I'd do next

- **Use real transaction data** instead of a self-reported category — this is the weakest input in the whole pipeline.
- **Handle cannibalisation.** The engine doesn't know what cards you already own, so it can recommend something redundant.
- **Learn the weights.** Right now the 50/50 category-vs-similarity split and the feature weights are reasoned, not learned. With real click and application logs, this becomes a proper ranking model — logistic regression on `P(apply | card, user)` — with the current features as the starting set.

**Metrics I'd track if this were live:** precision@1 (does the #1 pick match what people actually apply for), recommendation-to-application rate, and cards issued per month as the north star.
