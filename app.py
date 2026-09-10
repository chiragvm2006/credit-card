"""Streamlit UI for the credit card recommender."""

import streamlit as st
import pandas as pd
import recommender as rec

st.set_page_config(page_title="Card Finder", page_icon="💳")

cards = rec.load_cards()

st.title("💳 Credit Card Finder")
st.caption("Answer 5 questions. Get a card that actually matches how you spend.")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Monthly income (₹)", 5000, 1000000, 50000, step=5000)
    score = st.slider("Credit score", 300, 900, 750,
                      help="Not sure? 700 is a safe middle estimate.")
with col2:
    category = st.selectbox("You spend most on", rec.CATEGORIES)
    others = ["None"] + [c for c in rec.CATEGORIES if c != category]
    second = st.selectbox("Second most on", others)
    second = None if second == "None" else second

cap_fee = st.checkbox("Set a maximum annual fee")
max_fee = st.slider("Max annual fee (₹)", 0, 15000, 1000, step=250) if cap_fee else None

if st.button("Find my card", type="primary", use_container_width=True):
    top, eligible = rec.recommend(cards, income, score, category, second, max_fee)

    if top is None:
        closest, msg = rec.nearest_gap(cards, income, score)
        st.warning(msg)
        st.dataframe(closest[["card_name", "bank", "annual_fee",
                              "min_income", "min_credit_score"]],
                     hide_index=True, use_container_width=True)
        st.stop()

    avg_rate = eligible.reward_rate_top_category.mean()
    best = top.iloc[0]

    st.success(f"Found {len(eligible)} cards you qualify for. Here's the best fit:")

    st.header(best.card_name)
    st.caption(best.bank)

    a, b, c, d = st.columns(4)
    a.metric("Annual fee", "Free" if best.annual_fee == 0 else f"₹{best.annual_fee:,.0f}")
    b.metric("Rewards", f"{best.reward_rate_top_category:.1f}%")
    c.metric("Category", best.top_category.title())
    d.metric("Match", f"{best.score:.0%}")

    st.info(rec.explain(best, category, avg_rate))

    st.subheader("Your top 3")
    st.bar_chart(top.set_index("card_name")["score"], horizontal=True)

    with st.expander("See the runners-up"):
        for _, card in top.iloc[1:].iterrows():
            st.markdown(f"**{card.card_name}** · {card.bank} · {card.score:.0%} match")
            st.caption(rec.explain(card, category, avg_rate))

    with st.expander("How this was scored"):
        st.write(
            "Cards are first filtered by whether you'd actually be approved "
            "(income and credit score). The remaining cards are ranked by a "
            "k-Nearest Neighbours model that finds which real cards sit closest "
            "to an 'ideal card' — best rewards, lowest fee, biggest bonus, lounge "
            "access. That similarity is then blended 50/50 with how well the card's "
            "reward category matches your spending."
        )
        st.dataframe(
            top[["card_name", "reward_rate_top_category", "annual_fee",
                 "joining_bonus_value", "category_fit", "similarity", "score"]],
            hide_index=True, use_container_width=True)

st.divider()
st.caption(f"{len(cards)} cards across {cards.bank.nunique()} banks. "
           "Always check the bank's website before applying.")
