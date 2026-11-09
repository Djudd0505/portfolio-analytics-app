import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Portfolio Analytics", page_icon="📈", layout="wide")

st.title("📈 Portfolio Analytics & Investment Calculator")
st.write(
    "Explore a sample investment portfolio, estimate its expected return and risk, "
    "and project how an investment could grow over time."
)

st.info(
    "This educational calculator uses the assumptions you enter. It does not predict "
    "actual market performance or provide personalized investment advice."
)

st.header("1. Enter your portfolio")

default_assets = pd.DataFrame(
    {
        "Asset": ["Stock A", "Stock B", "Bond Fund"],
        "Allocation (%)": [50.0, 30.0, 20.0],
        "Expected Annual Return (%)": [10.0, 8.0, 4.0],
        "Annual Volatility (%)": [18.0, 12.0, 5.0],
    }
)

edited_assets = st.data_editor(
    default_assets,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "Asset": st.column_config.TextColumn("Asset", required=True),
        "Allocation (%)": st.column_config.NumberColumn(
            "Allocation (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f"
        ),
        "Expected Annual Return (%)": st.column_config.NumberColumn(
            "Expected Annual Return (%)", min_value=-99.0, max_value=1000.0, step=0.5, format="%.2f"
        ),
        "Annual Volatility (%)": st.column_config.NumberColumn(
            "Annual Volatility (%)", min_value=0.0, max_value=500.0, step=0.5, format="%.2f"
        ),
    },
)

st.caption("Edit the sample values, or add/remove rows using the table controls.")

risk_col, rate_col = st.columns(2)
with risk_col:
    correlation = st.slider(
        "Assumed correlation between each pair of assets",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="A simplified model assumes the same correlation for every pair of assets.",
    )
with rate_col:
    risk_free_rate = st.number_input(
        "Risk-free rate (%)",
        min_value=-5.0,
        max_value=30.0,
        value=4.0,
        step=0.25,
    )

# Clean and validate the editable table.
assets = edited_assets.copy()
assets["Asset"] = assets["Asset"].fillna("").astype(str).str.strip()
numeric_columns = [
    "Allocation (%)",
    "Expected Annual Return (%)",
    "Annual Volatility (%)",
]
for column in numeric_columns:
    assets[column] = pd.to_numeric(assets[column], errors="coerce")

assets = assets.dropna(subset=numeric_columns)
assets = assets[assets["Asset"] != ""]

valid = True
if assets.empty:
    st.error("Add at least one asset with complete numeric inputs.")
    valid = False
elif (assets["Allocation (%)"] < 0).any():
    st.error("Allocations cannot be negative.")
    valid = False
elif (assets["Annual Volatility (%)"] < 0).any():
    st.error("Annual volatility cannot be negative.")
    valid = False

allocation_total = float(assets["Allocation (%)"].sum()) if not assets.empty else 0.0
if valid and not math.isclose(allocation_total, 100.0, abs_tol=0.01):
    st.warning(f"Your allocations currently total {allocation_total:.2f}%. They must total 100%.")
    valid = False

n_assets = len(assets)
minimum_correlation = -1.0 if n_assets <= 1 else max(-1.0, -1.0 / (n_assets - 1))
if valid and correlation < minimum_correlation - 1e-9:
    st.error(
        f"For {n_assets} assets, the equal-correlation assumption must be at least "
        f"{minimum_correlation:.2f} to produce a valid covariance matrix."
    )
    valid = False

if valid:
    weights = assets["Allocation (%)"].to_numpy(dtype=float) / 100.0
    returns = assets["Expected Annual Return (%)"].to_numpy(dtype=float) / 100.0
    volatilities = assets["Annual Volatility (%)"].to_numpy(dtype=float) / 100.0

    # Expected portfolio return = sum(weight × expected asset return).
    portfolio_return = float(np.sum(weights * returns))

    # Portfolio variance with a common pairwise correlation assumption:
    # sum(w_i^2 × sigma_i^2) + 2*rho*sum(i<j)(w_i*w_j*sigma_i*sigma_j).
    variance = float(np.sum((weights * volatilities) ** 2))
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            variance += (
                2.0
                * correlation
                * weights[i]
                * weights[j]
                * volatilities[i]
                * volatilities[j]
            )
    portfolio_volatility = math.sqrt(max(variance, 0.0))
    sharpe_ratio = (
        (portfolio_return - risk_free_rate / 100.0) / portfolio_volatility
        if portfolio_volatility > 0
        else float("nan")
    )

    st.header("2. Portfolio results")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Expected annual return", f"{portfolio_return * 100:.2f}%")
    metric2.metric("Estimated annual volatility", f"{portfolio_volatility * 100:.2f}%")
    metric3.metric(
        "Estimated Sharpe ratio",
        f"{sharpe_ratio:.2f}" if math.isfinite(sharpe_ratio) else "N/A",
    )

    left, right = st.columns(2)
    with left:
        allocation_chart = px.pie(
            assets,
            names="Asset",
            values="Allocation (%)",
            title="Portfolio allocation",
            hole=0.35,
        )
        st.plotly_chart(allocation_chart, use_container_width=True)
    with right:
        return_chart = px.bar(
            assets,
            x="Asset",
            y="Expected Annual Return (%)",
            title="Expected annual return by asset",
            labels={"Expected Annual Return (%)": "Expected return (%)"},
        )
        st.plotly_chart(return_chart, use_container_width=True)

    st.subheader("Calculation breakdown")
    st.write(
        "**Expected portfolio return** = Σ (asset weight × expected asset return)"
    )
    breakdown = assets[["Asset", "Allocation (%)", "Expected Annual Return (%)"]].copy()
    breakdown["Weighted return contribution (%)"] = (
        breakdown["Allocation (%)"] / 100
        * breakdown["Expected Annual Return (%)"]
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)
    st.write(
        f"**Portfolio expected return:** "
        f"{' + '.join(f'({w:.2f} × {r:.2f}%)' for w, r in zip(weights, returns * 100))} "
        f"= **{portfolio_return * 100:.2f}%**"
    )
    st.caption(
        "Risk estimate: portfolio variance includes each asset's weighted variance "
        "and the pairwise covariance terms using the selected common correlation."
    )

    st.header("3. Investment growth projection")
    growth1, growth2, growth3 = st.columns(3)
    with growth1:
        initial_investment = st.number_input(
            "Initial investment ($)", min_value=0.0, value=10000.0, step=500.0
        )
    with growth2:
        monthly_contribution = st.number_input(
            "Monthly contribution ($)", min_value=0.0, value=250.0, step=25.0
        )
    with growth3:
        years = st.number_input(
            "Investment period (years)", min_value=1, max_value=50, value=20, step=1
        )

    if portfolio_return <= -1:
        st.error("The expected annual return must be greater than -100% for this projection.")
    else:
        monthly_rate = (1.0 + portfolio_return) ** (1.0 / 12.0) - 1.0
        months = int(years) * 12
        balances = []
        total_contributed = []
        balance = float(initial_investment)
        contributions = float(initial_investment)

        for month in range(months + 1):
            if month > 0:
                balance = balance * (1.0 + monthly_rate) + monthly_contribution
                contributions += monthly_contribution
            balances.append(balance)
            total_contributed.append(contributions)

        projection = pd.DataFrame(
            {
                "Month": range(months + 1),
                "Portfolio value ($)": balances,
                "Total contributions ($)": total_contributed,
            }
        )
        projection["Year"] = projection["Month"] / 12.0

        final_value = balances[-1]
        growth_metric1, growth_metric2, growth_metric3 = st.columns(3)
        growth_metric1.metric("Projected ending value", f"${final_value:,.2f}")
        growth_metric2.metric("Total amount contributed", f"${contributions:,.2f}")
        growth_metric3.metric("Projected investment growth", f"${final_value - contributions:,.2f}")

        growth_chart = px.line(
            projection,
            x="Year",
            y=["Portfolio value ($)", "Total contributions ($)"],
            title="Projected portfolio growth",
            labels={"value": "Value ($)", "Year": "Years invested", "variable": "Series"},
        )
        st.plotly_chart(growth_chart, use_container_width=True)

        st.caption(
            "Projection assumes the portfolio earns the same effective monthly rate "
            "every month, contributions are made at the end of each month, and there "
            "are no taxes, fees, or withdrawals. Actual returns will vary."
        )

    st.download_button(
        "Download portfolio inputs as CSV",
        data=assets.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_inputs.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Educational use only. Expected returns, volatility, and correlation are user-entered "
    "assumptions, not guarantees. This tool is not financial advice."
)
