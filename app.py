
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Finance helpers
# -----------------------------
def pmt(rate_per_period: float, nper: int, pv: float) -> float:
    """Fixed payment for an amortizing loan (positive number = payment outflow)."""
    if nper <= 0:
        return 0.0
    if abs(rate_per_period) < 1e-12:
        return pv / nper
    return pv * (rate_per_period * (1 + rate_per_period) ** nper) / ((1 + rate_per_period) ** nper - 1)


def amortize_one_month(balance: float, annual_rate: float, term_months: int, month_index: int) -> Tuple[float, float, float]:
    """
    Returns (interest, principal, new_balance) for a given month_index (0-based).
    If month_index >= term_months => no payment.
    """
    if month_index >= term_months or balance <= 1e-9:
        return 0.0, 0.0, 0.0

    r = annual_rate / 12.0
    payment = pmt(r, term_months, balance if month_index == 0 else balance)  # constant if balance is consistent
    interest = balance * r
    principal = max(payment - interest, 0.0)
    new_balance = max(balance - principal, 0.0)
    return interest, principal, new_balance


# -----------------------------
# Models
# -----------------------------
@dataclass
class PropertyPlan:
    name: str
    purchase_price: float
    down_payment_pct: float
    mortgage_years: int
    annual_interest_rate: float  # e.g., 0.045 for 4.5%
    annual_rent: float
    annual_service_charge: float
    upfront_fees_pct: float  # e.g., 0.04 for DLD+broker+other estimate
    rent_start_delay_months: int  # 0 for ready; e.g., 24 for off-plan until handover
    annual_rent_growth: float  # e.g., 0.02
    annual_service_growth: float  # e.g., 0.02


@dataclass
class SimulationInputs:
    start_date: str  # YYYY-MM-01 recommended
    horizon_years: int
    salary_monthly: float
    other_income_monthly: float
    expenses_monthly: float
    monthly_saving_extra: float  # optional additional saving (can be 0)
    initial_cash: float
    max_monthly_commitment: float  # max total mortgage payments allowed
    rent_credit_pct: float  # portion of active rent counted toward commitment
    min_cash_buffer: float  # keep this cash after buying
    buy_unit_when_possible: bool
    max_units: int


# -----------------------------
# Simulation
# -----------------------------
def simulate_portfolio(inputs: SimulationInputs, plan: PropertyPlan) -> pd.DataFrame:
    start = pd.to_datetime(inputs.start_date)
    months = inputs.horizon_years * 12
    dates = pd.date_range(start=start, periods=months, freq="MS")

    # State
    cash = inputs.initial_cash
    units = []  # each unit: dict with loan_balance, loan_month_index, purchase_date, rent_start_date, etc.

    rows = []

    for i, date in enumerate(dates):
        # incomes & expenses baseline
        base_income = inputs.salary_monthly + inputs.other_income_monthly + inputs.monthly_saving_extra
        base_net = base_income - inputs.expenses_monthly  # cash added before portfolio flows

        # rental & service for all units
        rent_total = 0.0
        service_total = 0.0

        # mortgages
        mortgage_payment_total = 0.0
        interest_total = 0.0
        principal_total = 0.0

        # Update each unit for this month
        for u in units:
            # rent/service (only after rent_start_date)
            if date >= u["rent_start_date"]:
                # apply growth from rent_start_date annually (simple annual comp, monthly applied)
                months_since_rent = (date.year - u["rent_start_date"].year) * 12 + (date.month - u["rent_start_date"].month)
                years_since = months_since_rent / 12.0
                rent = (plan.annual_rent / 12.0) * ((1 + plan.annual_rent_growth) ** years_since)
                service = (plan.annual_service_charge / 12.0) * ((1 + plan.annual_service_growth) ** years_since)
                rent_total += rent
                service_total += service

            # mortgage
            if u["loan_balance"] > 0 and u["loan_month_index"] < u["term_months"]:
                r = plan.annual_interest_rate / 12.0
                payment = pmt(r, u["term_months"], u["loan_original"])
                interest = u["loan_balance"] * r
                principal = max(payment - interest, 0.0)
                u["loan_balance"] = max(u["loan_balance"] - principal, 0.0)
                u["loan_month_index"] += 1

                mortgage_payment_total += payment
                interest_total += interest
                principal_total += principal

        # Apply month cash flow
        cash += base_net
        cash += rent_total
        cash -= service_total
        cash -= mortgage_payment_total

        # Buying logic (at most 1 unit per month to keep it realistic)
        bought = False
        reason = ""
        if inputs.buy_unit_when_possible and len(units) < inputs.max_units:
            # Upfront needed
            down = plan.purchase_price * plan.down_payment_pct
            fees = plan.purchase_price * plan.upfront_fees_pct
            upfront_needed = down + fees

            # Total mortgage payment if buy new unit
            r = plan.annual_interest_rate / 12.0
            term_months = plan.mortgage_years * 12
            loan_amount = plan.purchase_price - down
            new_payment = pmt(r, term_months, loan_amount)

            commitment_ok = (mortgage_payment_total + new_payment) <= (inputs.max_monthly_commitment + inputs.rent_credit_pct * rent_total) + 1e-9
            cash_ok = (cash - upfront_needed) >= inputs.min_cash_buffer

            if commitment_ok and cash_ok:
                # Execute purchase at end of month
                cash -= upfront_needed

                purchase_date = date
                rent_start_date = purchase_date + pd.offsets.MonthBegin(plan.rent_start_delay_months)

                units.append({
                    "purchase_date": purchase_date,
                    "rent_start_date": rent_start_date,
                    "loan_original": loan_amount,
                    "loan_balance": loan_amount,
                    "loan_month_index": 0,
                    "term_months": term_months,
                    "unit_id": len(units) + 1,
                })
                bought = True
                reason = f"Bought unit #{len(units)}"
            else:
                if not commitment_ok:
                    reason = "Not bought: commitment limit reached (after rent credit)"
                elif not cash_ok:
                    reason = "Not bought: insufficient cash (buffer rule)"
        else:
            if len(units) >= inputs.max_units:
                reason = "Target units reached"

        total_debt = sum(u["loan_balance"] for u in units)
        total_assets = len(units) * plan.purchase_price  # conservative: assume no price growth
        net_worth = cash + total_assets - total_debt

        rows.append({
            "Date": date.date().isoformat(),
            "Units": len(units),
            "BoughtThisMonth": bought,
            "Note": reason,
            "BaseNetFromSalaryEtc": base_net,
            "RentIn": rent_total,
            "ServiceChargeOut": service_total,
            "MortgagePaymentOut": mortgage_payment_total,
            "InterestPart": interest_total,
            "PrincipalPart": principal_total,
            "CashEnd": cash,
            "TotalDebtEnd": total_debt,
            "TotalAssetsValue": total_assets,
            "NetWorth": net_worth,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="UAE Real Estate Portfolio Simulator", layout="wide")
st.title("UAE / GCC Real‑Estate Portfolio Simulator (Excel‑style logic, but as a web tool)")

st.caption("This tool simulates building a rental portfolio under cash and bank‑commitment constraints. "
           "Educational / planning use only — not financial advice.")

with st.sidebar:
    st.header("1) Simulation Inputs")

    start_date = st.text_input("Start date (YYYY-MM-01)", value="2027-08-01")
    horizon_years = st.slider("Horizon (years)", 5, 40, 25)

    salary_monthly = st.number_input("Salary (monthly)", min_value=0.0, value=70000.0, step=1000.0)
    other_income_monthly = st.number_input("Other income (monthly)", min_value=0.0, value=15000.0, step=500.0)
    expenses_monthly = st.number_input("Expenses (monthly)", min_value=0.0, value=30000.0, step=500.0)
    monthly_saving_extra = st.number_input("Extra saving / additional net (monthly)", min_value=0.0, value=0.0, step=500.0)

    initial_cash = st.number_input("Initial cash available", min_value=0.0, value=150000.0, step=5000.0)

    max_monthly_commitment = st.number_input("Max total mortgage payments allowed", min_value=0.0, value=25000.0, step=500.0)
    rent_credit_pct = st.slider("Bank rent credit % (counts toward commitment)", 0, 100, 70) / 100.0
    min_cash_buffer = st.number_input("Minimum cash buffer after buying", min_value=0.0, value=30000.0, step=1000.0)

    buy_unit_when_possible = st.checkbox("Auto-buy next unit when rules allow", value=True)
    max_units = st.slider("Max units to acquire", 1, 50, 10)

    st.divider()
    st.header("2) Property Assumptions")

    purchase_price = st.number_input("Purchase price (AED)", min_value=0.0, value=1300000.0, step=10000.0)
    down_payment_pct = st.slider("Down payment %", 5, 50, 20) / 100.0
    mortgage_years = st.slider("Mortgage term (years)", 5, 30, 20)
    annual_interest_rate = st.number_input("Interest rate (APR as decimal, e.g. 0.045)", min_value=0.0, value=0.045, step=0.001)

    annual_rent = st.number_input("Annual rent (AED)", min_value=0.0, value=85000.0, step=1000.0)
    annual_service_charge = st.number_input("Annual service charge (AED)", min_value=0.0, value=12000.0, step=500.0)

    upfront_fees_pct = st.slider("Upfront fees % (DLD+broker+misc estimate)", 0, 10, 4) / 100.0

    mode = st.radio("Unit type", ["Ready (rent starts immediately)", "Off‑plan (rent starts after delay)"], index=0)
    rent_start_delay_months = 0 if mode.startswith("Ready") else st.slider("Delay until rent starts (months)", 6, 60, 24)

    annual_rent_growth = st.slider("Rent growth % (annual)", 0, 10, 2) / 100.0
    annual_service_growth = st.slider("Service charge growth % (annual)", 0, 10, 2) / 100.0

    st.divider()
    st.header("3) Run")
    run = st.button("Run simulation", type="primary")

if run:
    # Build inputs
    sim_inputs = SimulationInputs(
        start_date=start_date,
        horizon_years=int(horizon_years),
        salary_monthly=float(salary_monthly),
        other_income_monthly=float(other_income_monthly),
        expenses_monthly=float(expenses_monthly),
        monthly_saving_extra=float(monthly_saving_extra),
        initial_cash=float(initial_cash),
        max_monthly_commitment=float(max_monthly_commitment),
        rent_credit_pct=float(rent_credit_pct),
        min_cash_buffer=float(min_cash_buffer),
        buy_unit_when_possible=bool(buy_unit_when_possible),
        max_units=int(max_units),
    )

    plan = PropertyPlan(
        name="BasePlan",
        purchase_price=float(purchase_price),
        down_payment_pct=float(down_payment_pct),
        mortgage_years=int(mortgage_years),
        annual_interest_rate=float(annual_interest_rate),
        annual_rent=float(annual_rent),
        annual_service_charge=float(annual_service_charge),
        upfront_fees_pct=float(upfront_fees_pct),
        rent_start_delay_months=int(rent_start_delay_months),
        annual_rent_growth=float(annual_rent_growth),
        annual_service_growth=float(annual_service_growth),
    )

    df = simulate_portfolio(sim_inputs, plan)

    # KPI cards
    last = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Units (end)", int(last["Units"]))
    col2.metric("Cash (end)", f'{last["CashEnd"]:,.0f} AED')
    col3.metric("Debt (end)", f'{last["TotalDebtEnd"]:,.0f} AED')
    col4.metric("Net worth (end)", f'{last["NetWorth"]:,.0f} AED')

    st.subheader("Timeline (monthly)")
    st.dataframe(df, use_container_width=True, height=420)

    st.subheader("Quick charts")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.set_index("Date")[["CashEnd", "NetWorth"]])
    with c2:
        st.line_chart(df.set_index("Date")[["Units", "TotalDebtEnd"]])

    # Export
    st.subheader("Export")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="portfolio_simulation.csv", mime="text/csv")

else:
    st.info("Set your assumptions in the sidebar, then click **Run simulation**.")
