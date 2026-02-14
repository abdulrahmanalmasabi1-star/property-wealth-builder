
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# PDF + charts
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from io import BytesIO


# -----------------------------
# Finance helpers
# -----------------------------
def pmt(rate_per_period: float, nper: int, pv: float) -> float:
    if nper <= 0:
        return 0.0
    if abs(rate_per_period) < 1e-12:
        return pv / nper
    return pv * (rate_per_period * (1 + rate_per_period) ** nper) / ((1 + rate_per_period) ** nper - 1)


# -----------------------------
# Models
# -----------------------------
@dataclass
class PropertyPlan:
    purchase_price: float
    down_payment_pct: float
    mortgage_years: int
    annual_interest_rate: float
    annual_rent: float
    annual_service_charge: float
    annual_maintenance: float
    upfront_fees_pct: float
    rent_start_delay_months: int
    vacancy_pct: float
    annual_rent_growth: float
    annual_service_growth: float
    annual_property_growth: float


@dataclass
class TaxSettings:
    enabled: bool
    rental_tax_pct: float
    corporate_tax_pct: float
    cap_gains_tax_pct: float
    tax_starts_year: int


@dataclass
class SimulationInputs:
    start_date: str
    horizon_years: int
    salary_monthly: float
    other_income_monthly: float
    expenses_monthly: float
    initial_cash: float

    strategy: str  # Personal only | Company only | Hybrid (auto)
    max_monthly_commitment: float
    rent_credit_pct: float
    dbr_limit: float
    dscr_min: float
    min_cash_buffer: float

    auto_buy: bool
    max_units: int


# -----------------------------
# Simulation
# -----------------------------
def simulate_portfolio(inputs: SimulationInputs, plan: PropertyPlan, taxes: TaxSettings) -> pd.DataFrame:
    start = pd.to_datetime(inputs.start_date)
    months = inputs.horizon_years * 12
    dates = pd.date_range(start=start, periods=months, freq="MS")

    cash = inputs.initial_cash
    units: List[Dict] = []
    company_mode_active = (inputs.strategy == "Company only")

    rows = []

    for i, date in enumerate(dates):
        year_index = (i // 12) + 1

        base_income = inputs.salary_monthly + inputs.other_income_monthly
        base_net = base_income - inputs.expenses_monthly

        rent_total = 0.0
        service_total = 0.0
        maint_total = 0.0

        mortgage_payment_total = 0.0
        interest_total = 0.0
        principal_total = 0.0

        # Update all units for this month
        for u in units:
            if date >= u["rent_start_date"]:
                months_since_rent = (date.year - u["rent_start_date"].year) * 12 + (date.month - u["rent_start_date"].month)
                years_since = months_since_rent / 12.0

                gross_rent = (plan.annual_rent / 12.0) * ((1 + plan.annual_rent_growth) ** years_since)
                gross_rent *= (1.0 - plan.vacancy_pct)

                service = (plan.annual_service_charge / 12.0) * ((1 + plan.annual_service_growth) ** years_since)
                maint = (plan.annual_maintenance / 12.0)

                rent_total += gross_rent
                service_total += service
                maint_total += maint

            if u["loan_balance"] > 1e-9 and u["loan_month_index"] < u["term_months"]:
                r = u["annual_rate"] / 12.0
                payment = u["payment"]
                interest = u["loan_balance"] * r
                principal = max(payment - interest, 0.0)
                u["loan_balance"] = max(u["loan_balance"] - principal, 0.0)
                u["loan_month_index"] += 1

                mortgage_payment_total += payment
                interest_total += interest
                principal_total += principal

        # Taxes (optional) — simplified: on positive net rental profit
        rental_tax = 0.0
        corporate_tax = 0.0
        if taxes.enabled and year_index >= taxes.tax_starts_year:
            net_rent_profit = max(rent_total - service_total - maint_total, 0.0)
            rental_tax = net_rent_profit * taxes.rental_tax_pct
            if company_mode_active:
                corporate_tax = net_rent_profit * taxes.corporate_tax_pct

        # Apply month cash flow
        cash += base_net
        cash += rent_total
        cash -= (service_total + maint_total + rental_tax + corporate_tax)
        cash -= mortgage_payment_total

        # Buying logic (max 1 unit / month)
        bought = False
        note = ""

        down = plan.purchase_price * plan.down_payment_pct
        fees = plan.purchase_price * plan.upfront_fees_pct
        upfront_needed = down + fees

        term_months = plan.mortgage_years * 12
        loan_amount = plan.purchase_price - down
        r = plan.annual_interest_rate / 12.0
        new_payment = pmt(r, term_months, loan_amount)

        active_rent_credit = inputs.rent_credit_pct * rent_total

        denom_income = (inputs.salary_monthly + inputs.other_income_monthly + inputs.rent_credit_pct * rent_total)
        dbr = (mortgage_payment_total + new_payment) / max(denom_income, 1e-9)

        commitment_ok = (mortgage_payment_total + new_payment) <= (inputs.max_monthly_commitment + active_rent_credit) + 1e-9
        dbr_ok = dbr <= inputs.dbr_limit + 1e-9
        personal_ok = commitment_ok and dbr_ok

        est_gross_rent_new = (plan.annual_rent / 12.0) * (1.0 - plan.vacancy_pct)
        est_noi_new = max(est_gross_rent_new - (plan.annual_service_charge / 12.0) - (plan.annual_maintenance / 12.0), 0.0)
        dscr = est_noi_new / max(new_payment, 1e-9)
        company_ok = dscr >= inputs.dscr_min - 1e-9

        if inputs.auto_buy and len(units) < inputs.max_units:
            cash_ok = (cash - upfront_needed) >= inputs.min_cash_buffer

            if inputs.strategy == "Personal only":
                mode_ok = personal_ok
                if not personal_ok:
                    note = "Not bought: Personal limits reached (DBR/commitment)"
            elif inputs.strategy == "Company only":
                company_mode_active = True
                mode_ok = company_ok
                if not company_ok:
                    note = "Not bought: Company DSCR below minimum"
            else:  # Hybrid (auto)
                if not company_mode_active:
                    if personal_ok:
                        mode_ok = True
                    else:
                        company_mode_active = True
                        mode_ok = company_ok
                        note = "Switched to Company mode (personal limit reached)" if mode_ok else "Not bought: Switched to Company but DSCR below minimum"
                else:
                    mode_ok = company_ok
                    if not company_ok:
                        note = "Not bought: Company DSCR below minimum"

            if cash_ok and mode_ok:
                cash -= upfront_needed
                purchase_date = date
                rent_start_date = purchase_date + pd.offsets.MonthBegin(plan.rent_start_delay_months)

                units.append({
                    "unit_id": len(units) + 1,
                    "purchase_date": purchase_date,
                    "rent_start_date": rent_start_date,
                    "purchase_price": plan.purchase_price,
                    "loan_balance": loan_amount,
                    "loan_month_index": 0,
                    "term_months": term_months,
                    "annual_rate": plan.annual_interest_rate,
                    "payment": new_payment,
                })
                bought = True
                note = note or f"Bought unit #{len(units)}"
            else:
                if not cash_ok and note == "":
                    note = "Not bought: Insufficient cash (buffer rule)"
        else:
            if len(units) >= inputs.max_units:
                note = "Target units reached"

        total_debt = sum(u["loan_balance"] for u in units)

        total_assets = 0.0
        total_equity = 0.0
        for u in units:
            months_since_buy = (date.year - u["purchase_date"].year) * 12 + (date.month - u["purchase_date"].month)
            years_since_buy = months_since_buy / 12.0
            value = u["purchase_price"] * ((1 + plan.annual_property_growth) ** years_since_buy)
            total_assets += value
            total_equity += max(value - u["loan_balance"], 0.0)

        net_worth = cash + total_assets - total_debt

        rows.append({
            "Date": date,
            "Year": year_index,
            "Units": len(units),
            "CompanyModeActive": company_mode_active,
            "BoughtThisMonth": bought,
            "Note": note,
            "BaseNet": base_net,
            "RentIn": rent_total,
            "ServiceOut": service_total,
            "MaintenanceOut": maint_total,
            "RentalTaxOut": rental_tax,
            "CorporateTaxOut": corporate_tax,
            "MortgagePaymentOut": mortgage_payment_total,
            "InterestPart": interest_total,
            "PrincipalPart": principal_total,
            "DBR": dbr,
            "DSCR_new_unit": dscr,
            "CashEnd": cash,
            "TotalDebtEnd": total_debt,
            "TotalAssetsValue": total_assets,
            "TotalEquity": total_equity,
            "NetWorth": net_worth,
        })

    return pd.DataFrame(rows)


def yearly_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf = df.groupby("Year").agg(
        Units_End=("Units", "last"),
        Rent=("RentIn", "sum"),
        Service=("ServiceOut", "sum"),
        Maintenance=("MaintenanceOut", "sum"),
        Taxes=("RentalTaxOut", "sum"),
        MortgagePayments=("MortgagePaymentOut", "sum"),
        Interest=("InterestPart", "sum"),
        Principal=("PrincipalPart", "sum"),
        BaseNet=("BaseNet", "sum"),
        CashEnd=("CashEnd", "last"),
        CompanyMode=("CompanyModeActive", "last"),
    ).reset_index()

    perf["NetCashFlow"] = perf["BaseNet"] + perf["Rent"] - perf["Service"] - perf["Maintenance"] - perf["Taxes"] - perf["MortgagePayments"]

    bal = df.groupby("Year").agg(
        Units_End=("Units", "last"),
        Cash=("CashEnd", "last"),
        Assets=("TotalAssetsValue", "last"),
        Debt=("TotalDebtEnd", "last"),
        Equity=("TotalEquity", "last"),
        NetWorth=("NetWorth", "last"),
        CompanyMode=("CompanyModeActive", "last"),
    ).reset_index()

    return perf, bal


# -----------------------------
# Presentation helpers
# -----------------------------
def make_chart_png(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_cols])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{currency_code} / Units")
    ax.legend(y_cols)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def df_to_rl_table(df: pd.DataFrame, fmt_map: Dict[str, str], max_rows: int = 30) -> Table:
    d = df.copy()
    if max_rows and len(d) > max_rows:
        d = d.head(max_rows)

    # format
    for c, f in fmt_map.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) and pd.notna(v) else v)

    data = [list(d.columns)] + d.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]))
    return table


def build_pdf_report(
    sim: SimulationInputs,
    plan: PropertyPlan,
    taxes: TaxSettings,
    perf: pd.DataFrame,
    bal: pd.DataFrame,
    charts: Dict[str, bytes],
) -> bytes:
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Global Real‑Estate Portfolio Builder – Report", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Educational / planning use only. Not financial advice.", styles["Italic"]))
    story.append(Spacer(1, 0.4 * cm))

    # Inputs summary
    story.append(Paragraph("Assumptions Summary", styles["Heading2"]))
    summary_lines = [
        f"Start date: {sim.start_date} | Horizon: {sim.horizon_years} years | Strategy: {sim.strategy}",
        f"Income (monthly): Salary {sim.salary_monthly:,.0f} + Other {sim.other_income_monthly:,.0f} | Expenses {sim.expenses_monthly:,.0f}",
        f"Initial cash: {sim.initial_cash:,.0f} | Max units: {sim.max_units} | Cash buffer: {sim.min_cash_buffer:,.0f}",
        f"Personal: Base cap {sim.max_monthly_commitment:,.0f} | Rent credit {sim.rent_credit_pct*100:.0f}% | DBR limit {sim.dbr_limit*100:.0f}%",
        f"Company: DSCR min {sim.dscr_min:.2f}",
        f"Property: Price {plan.purchase_price:,.0f} | DP {plan.down_payment_pct*100:.0f}% | Term {plan.mortgage_years}y | Rate {plan.annual_interest_rate*100:.2f}%",
        f"Rent: {plan.annual_rent:,.0f}/yr | Vacancy {plan.vacancy_pct*100:.0f}% | Service {plan.annual_service_charge:,.0f}/yr | Maint {plan.annual_maintenance:,.0f}/yr",
        f"Growth: Property {plan.annual_property_growth*100:.0f}% | Rent {plan.annual_rent_growth*100:.0f}% | Service {plan.annual_service_growth*100:.0f}%",
        f"Tax enabled: {'Yes' if taxes.enabled else 'No'} (Rental {taxes.rental_tax_pct*100:.0f}%, Corporate {taxes.corporate_tax_pct*100:.0f}%, Starts year {taxes.tax_starts_year})",
    ]
    for line in summary_lines:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    # KPIs
    last_bal = bal.iloc[-1].to_dict()
    story.append(Paragraph("Key Results (End of Horizon)", styles["Heading2"]))
    kpi_data = [
        ["Units", f"Cash ({currency_code})", f"Assets ({currency_code})", f"Debt ({currency_code})", f"Net Worth ({currency_code})"],
        [
            f'{int(last_bal["Units_End"])}',
            f'{last_bal["Cash"]:,.0f}',
            f'{last_bal["Assets"]:,.0f}',
            f'{last_bal["Debt"]:,.0f}',
            f'{last_bal["NetWorth"]:,.0f}',
        ],
    ]
    kpi_tbl = Table(kpi_data)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # Charts
    story.append(Paragraph("Charts", styles["Heading2"]))
    for title, png_bytes in charts.items():
        story.append(Paragraph(title, styles["Heading3"]))
        img = RLImage(BytesIO(png_bytes))
        img.drawHeight = 7.5 * cm
        img.drawWidth = 16.5 * cm
        story.append(img)
        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())

    # Tables
    story.append(Paragraph("Yearly Performance (P&L)", styles["Heading2"]))
    perf_tbl = df_to_rl_table(
        perf,
        fmt_map={
            "Rent": "{:,.0f}", "Service": "{:,.0f}", "Maintenance": "{:,.0f}", "Taxes": "{:,.0f}",
            "MortgagePayments": "{:,.0f}", "Interest": "{:,.0f}", "Principal": "{:,.0f}",
            "BaseNet": "{:,.0f}", "NetCashFlow": "{:,.0f}", "CashEnd": "{:,.0f}",
        },
        max_rows=40
    )
    story.append(perf_tbl)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Yearly Balance Sheet", styles["Heading2"]))
    bal_tbl = df_to_rl_table(
        bal,
        fmt_map={"Cash": "{:,.0f}", "Assets": "{:,.0f}", "Debt": "{:,.0f}", "Equity": "{:,.0f}", "NetWorth": "{:,.0f}"},
        max_rows=40
    )
    story.append(bal_tbl)

    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.2*cm, rightMargin=1.2*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    doc.build(story)
    return out.getvalue()


HOW_TO_USE_EN = """
### How to use this tool (English)

1. **Set your starting point**  
   Choose the *Start date*, *Horizon*, your monthly *Income*, *Expenses*, and *Initial cash*.

2. **Choose a financing strategy**  
   - **Personal only**: buy until DBR/commitment limits are reached.  
   - **Company only**: underwriting is based on DSCR (rental coverage).  
   - **Hybrid (auto)**: starts personal, then switches to company mode when personal limits are reached.

3. **Adjust bank rules**
   - **Base cap (Currency)**: your maximum mortgage payments allowed.
   - **Rent credit %**: portion of active rent counted toward your cap and income.
   - **DBR limit** (personal) and **DSCR min** (company).

4. **Set the property assumptions**
   Price, down payment, mortgage term, interest rate, rent, vacancy, service charges, maintenance, and fees.

5. **(Optional) Growth & Tax**
   Turn on property/rent growth and taxes if relevant for your scenario.

6. Click **Run simulation**  
   You will see:
   - **Yearly Performance (P&L)** table
   - **Yearly Balance Sheet** table
   - Charts (Net worth / Assets vs Debt, Cashflow trend)

7. **Export**
   Use the buttons to download CSV files or a **PDF report**.
"""


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Property Wealth Builder", layout="wide")
st.title("Global Real‑Estate Portfolio Builder")
st.caption("Clean dashboard + yearly summaries. (Planning/education only — not financial advice.)")

with st.sidebar:
    st.header("1) Simulation")
# Currency (display only)
CURRENCIES = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "AED": "AED",
    "SAR": "SAR",
    "QAR": "QAR",
    "KWD": "KD",
    "BHD": "BD",
    "OMR": "OMR",
    "INR": "₹",
    "PKR": "PKR",
    "EGP": "EGP",
    "TRY": "₺",
    "SGD": "S$",
    "HKD": "HK$",
    "AUD": "A$",
    "CAD": "C$",
    "JPY": "¥",
}
currency_code = st.selectbox("Currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index("AED"))
currency_symbol = CURRENCIES[currency_code]

    start_date = st.text_input("Start_date (YYYY-MM-01)", value="2027-08-01")
    horizon_years = st.slider("Horizon (years)", 5, 40, 25)

    salary_monthly = st.number_input("Salary (monthly)", min_value=0.0, value=70000.0, step=1000.0)
    other_income_monthly = st.number_input("Other income (monthly)", min_value=0.0, value=15000.0, step=500.0)
    expenses_monthly = st.number_input("Expenses (monthly)", min_value=0.0, value=30000.0, step=500.0)
    initial_cash = st.number_input("Initial cash", min_value=0.0, value=150000.0, step=5000.0)

    st.divider()
    st.header("2) Financing")
    strategy = st.selectbox("Strategy", ["Personal only", "Company only", "Hybrid (auto)"], index=2)

    max_monthly_commitment = st.number_input(f"Base cap ({currency_code})", min_value=0.0, value=25000.0, step=500.0,
                                             help="Your max total mortgage payments allowed (before rent credit).")
    rent_credit_pct = st.slider("Rent credit %", 0, 100, 70, help="Portion of ACTIVE rent counted toward cap and income.") / 100.0

    dbr_limit = st.slider("DBR limit % (personal)", 40, 60, 50, help="Debt Burden Ratio limit for personal financing.") / 100.0
    dscr_min = st.slider("DSCR minimum (company)", 1.0, 1.8, 1.2, step=0.05, help="Debt service coverage ratio requirement for company financing.")

    min_cash_buffer = st.number_input("Cash buffer after buying", min_value=0.0, value=30000.0, step=1000.0,
                                      help="Minimum cash you want to keep after paying down payment + fees.")

    auto_buy = st.checkbox("Auto-buy when allowed", value=True)
    max_units = st.slider("Max units", 1, 50, 10)

    st.divider()
    st.header("3) Property")
    purchase_price = st.number_input("Purchase price (Currency)", min_value=0.0, value=1300000.0, step=10000.0)
    down_payment_pct = st.slider("Down payment %", 5, 50, 20) / 100.0
    mortgage_years = st.slider("Mortgage term (years)", 5, 30, 20)
    annual_interest_rate = st.number_input("Interest rate (APR as decimal)", min_value=0.0, value=0.045, step=0.001)

    annual_rent = st.number_input("Annual rent (Currency)", min_value=0.0, value=85000.0, step=1000.0)
    vacancy_pct = st.slider("Vacancy %", 0, 30, 5) / 100.0
    annual_service_charge = st.number_input("Annual service charge (Currency)", min_value=0.0, value=12000.0, step=500.0)
    annual_maintenance = st.number_input("Annual maintenance (Currency)", min_value=0.0, value=6000.0, step=500.0)
    upfront_fees_pct = st.slider("Upfront fees % (DLD+broker+misc)", 0, 10, 4) / 100.0

    delivery = st.radio("Delivery", ["Ready", "Off‑plan"], index=0)
    rent_start_delay_months = 0 if delivery == "Ready" else st.slider("Rent start delay (months)", 6, 60, 24)

    st.divider()
    st.header("4) Growth (optional)")
    annual_property_growth = st.slider("Property value growth %", 0, 10, 0) / 100.0
    annual_rent_growth = st.slider("Rent growth %", 0, 10, 2) / 100.0
    annual_service_growth = st.slider("Service growth %", 0, 10, 2) / 100.0

    st.divider()
    st.header("5) Tax (optional)")
    tax_enabled = st.checkbox("Enable tax", value=False)
    rental_tax_pct = st.slider("Rental income tax %", 0, 40, 0) / 100.0
    corporate_tax_pct = st.slider("Corporate tax % (company mode)", 0, 40, 0) / 100.0
    cap_gains_tax_pct = st.slider("Capital gains tax %", 0, 40, 0) / 100.0
    tax_starts_year = st.slider("Tax starts at year", 1, 30, 1)

    st.divider()
    run = st.button("Run simulation", type="primary")


tabs = st.tabs(["Dashboard", "Monthly (advanced)", "How to use (EN)"])

with tabs[2]:
    st.markdown(HOW_TO_USE_EN)

with tabs[0]:
    st.info("Set assumptions in the sidebar, then click **Run simulation**.")

with tabs[1]:
    st.info("Monthly table will appear after you run the simulation.")


if run:
    sim = SimulationInputs(
        start_date=start_date,
        horizon_years=int(horizon_years),
        salary_monthly=float(salary_monthly),
        other_income_monthly=float(other_income_monthly),
        expenses_monthly=float(expenses_monthly),
        initial_cash=float(initial_cash),

        strategy=strategy,
        max_monthly_commitment=float(max_monthly_commitment),
        rent_credit_pct=float(rent_credit_pct),
        dbr_limit=float(dbr_limit),
        dscr_min=float(dscr_min),
        min_cash_buffer=float(min_cash_buffer),

        auto_buy=bool(auto_buy),
        max_units=int(max_units),
    )

    plan = PropertyPlan(
        purchase_price=float(purchase_price),
        down_payment_pct=float(down_payment_pct),
        mortgage_years=int(mortgage_years),
        annual_interest_rate=float(annual_interest_rate),
        annual_rent=float(annual_rent),
        annual_service_charge=float(annual_service_charge),
        annual_maintenance=float(annual_maintenance),
        upfront_fees_pct=float(upfront_fees_pct),
        rent_start_delay_months=int(rent_start_delay_months),
        vacancy_pct=float(vacancy_pct),
        annual_rent_growth=float(annual_rent_growth),
        annual_service_growth=float(annual_service_growth),
        annual_property_growth=float(annual_property_growth),
    )

    taxes = TaxSettings(
        enabled=bool(tax_enabled),
        rental_tax_pct=float(rental_tax_pct),
        corporate_tax_pct=float(corporate_tax_pct),
        cap_gains_tax_pct=float(cap_gains_tax_pct),
        tax_starts_year=int(tax_starts_year),
    )

    df = simulate_portfolio(sim, plan, taxes)
    perf, bal = yearly_summaries(df)

    # Charts (for UI + PDF)
    bal_chart = make_chart_png(bal, "Year", ["NetWorth", "Assets", "Debt"], "Balance Sheet Trend")
    perf_chart = make_chart_png(perf, "Year", ["NetCashFlow", "Rent", "MortgagePayments"], "Cashflow Trend")

    last = bal.iloc[-1]
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Units (end)", int(last["Units_End"]))
        c2.metric("Cash (end)", f'{last["Cash"]:,.0f} {currency_code}')
        c3.metric("Debt (end)", f'{last["Debt"]:,.0f} {currency_code}')
        c4.metric("Net worth (end)", f'{last["NetWorth"]:,.0f} {currency_code}')

        st.subheader("Yearly Performance (P&L)")
        st.dataframe(perf.style.format({
            "Rent": "{:,.0f}", "Service": "{:,.0f}", "Maintenance": "{:,.0f}", "Taxes": "{:,.0f}",
            "MortgagePayments": "{:,.0f}", "Interest": "{:,.0f}", "Principal": "{:,.0f}",
            "BaseNet": "{:,.0f}", "NetCashFlow": "{:,.0f}", "CashEnd": "{:,.0f}",
        }), use_container_width=True, height=340)

        st.subheader("Yearly Balance Sheet")
        st.dataframe(bal.style.format({
            "Cash": "{:,.0f}", "Assets": "{:,.0f}", "Debt": "{:,.0f}", "Equity": "{:,.0f}", "NetWorth": "{:,.0f}",
        }), use_container_width=True, height=320)

        st.subheader("Charts")
        st.image(bal_chart, caption="Balance Sheet Trend", use_container_width=True)
        st.image(perf_chart, caption="Cashflow Trend", use_container_width=True)

        st.subheader("Export")
        st.download_button("Download monthly CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="portfolio_monthly.csv", mime="text/csv")
        st.download_button("Download yearly performance CSV", data=perf.to_csv(index=False).encode("utf-8"),
                           file_name="portfolio_yearly_performance.csv", mime="text/csv")
        st.download_button("Download yearly balance sheet CSV", data=bal.to_csv(index=False).encode("utf-8"),
                           file_name="portfolio_yearly_balance.csv", mime="text/csv")

        pdf_bytes = build_pdf_report(
            sim=sim, plan=plan, taxes=taxes,
            perf=perf, bal=bal,
            charts={"Balance Sheet Trend": bal_chart, "Cashflow Trend": perf_chart},
        )
        st.download_button("Download PDF report", data=pdf_bytes, file_name="portfolio_report.pdf", mime="application/pdf")

    with tabs[1]:
        st.dataframe(df, use_container_width=True, height=520)
