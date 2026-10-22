from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_app_artifacts.joblib"

TIER_ACTIONS = {
    "Low": "No outreach needed. Below the 0.30 contact threshold.",
    "Medium": "Low-cost touch: email or check-in call.",
    "High": "Proactive outreach: service review and a retention offer.",
    "Critical": "Priority: personal call and a contract-upgrade offer.",
}
TIER_COLORS = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "violet"}
MIN_EFFECT = 0.05  # ignore tiny contributions when listing drivers
ADD_ON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
MAX_ROWS = 10_000

# Real rows from the public IBM Telco dataset, used for the downloadable template
TEMPLATE_ROWS = [
    ["5178-LMXOP", "Male", 1, "Yes", "No", 1, "Yes", "Yes", "Fiber optic", "No", "No", "No", "No",
     "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 95.10, 95.10],
    ["7590-VHVEG", "Female", 0, "Yes", "No", 1, "No", "No phone service", "DSL", "No", "Yes", "No",
     "No", "No", "No", "Month-to-month", "Yes", "Electronic check", 29.85, 29.85],
    ["5575-GNVDE", "Male", 0, "No", "No", 34, "Yes", "No", "DSL", "Yes", "No", "Yes", "No", "No",
     "No", "One year", "No", "Mailed check", 56.95, 1889.50],
    ["7469-LKBCI", "Male", 0, "No", "No", 16, "Yes", "No", "No", "No internet service",
     "No internet service", "No internet service", "No internet service", "No internet service",
     "No internet service", "Two year", "No", "Credit card (automatic)", 18.95, 326.80],
]


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACTS_PATH)


# ---------- Shared model logic ----------

def explain_many(customers, artifacts):
    """Grouped SHAP values for many customers: weight × (customer value − training average)."""
    pipe = artifacts["pipeline"]
    x = pipe.named_steps["preprocess"].transform(customers)
    coef = pipe.named_steps["model"].coef_[0]
    contributions = pd.DataFrame(coef * (x - artifacts["feature_means"]),
                                 columns=artifacts["feature_names"], index=customers.index)
    return contributions.T.groupby(artifacts["feature_groups"]).sum().T


def assign_tiers(probabilities, artifacts):
    return pd.cut(probabilities, bins=artifacts["risk_bins"],
                  labels=artifacts["risk_labels"], right=False)


# ---------- Single customer ----------

def collect_inputs(options, ranges):
    """Sidebar form. Keeps dependent fields consistent with the training data."""
    st.sidebar.subheader("Account")
    tenure = st.sidebar.slider("Tenure (months)", 0, int(ranges["tenure"][1]), 12)
    contract = st.sidebar.selectbox("Contract", options["Contract"])
    payment = st.sidebar.selectbox("Payment method", options["PaymentMethod"], index=2)
    paperless = st.sidebar.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    monthly = st.sidebar.number_input(
        "Monthly charges ($)", min_value=ranges["MonthlyCharges"][0],
        max_value=ranges["MonthlyCharges"][1], value=70.0, step=5.0)
    total = round(tenure * monthly, 2)
    st.sidebar.caption(f"Total charges estimated as tenure × monthly charges: ${total:,.2f}")

    st.sidebar.subheader("Services")
    phone = st.sidebar.radio("Phone service", ["Yes", "No"], horizontal=True)
    if phone == "Yes":
        multiple_lines = st.sidebar.radio("Multiple lines", ["No", "Yes"], horizontal=True)
    else:
        multiple_lines = "No phone service"

    internet = st.sidebar.selectbox("Internet service", options["InternetService"], index=1)
    add_ons = {}
    for service in ADD_ON_SERVICES:
        if internet == "No":
            add_ons[service] = "No internet service"
        else:
            add_ons[service] = st.sidebar.radio(service, ["No", "Yes"], horizontal=True)

    st.sidebar.subheader("Demographics")
    gender = st.sidebar.radio("Gender", options["gender"], horizontal=True)
    senior = st.sidebar.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = st.sidebar.radio("Partner", ["No", "Yes"], horizontal=True)
    dependents = st.sidebar.radio("Dependents", ["No", "Yes"], horizontal=True)

    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        **add_ons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def driver_label(group, inputs):
    if group == "Tenure & billing":
        return f"Tenure & billing ({inputs['tenure']} mo, ${inputs['MonthlyCharges']:.0f}/mo)"
    value = inputs[group]
    if group == "SeniorCitizen":
        value = "Yes" if value == 1 else "No"
    return f"{group} = {value}"


def plot_drivers(grouped, inputs, top_n=8):
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).index[:top_n])[::-1]
    labels = [driver_label(group, inputs) for group in top.index]
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in top.values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, top.values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Push on churn score (red = raises risk, blue = lowers risk)")
    fig.tight_layout()
    return fig


def single_customer_page(artifacts):
    st.write("Enter a customer's details in the sidebar to see their churn risk, "
             "the recommended action, and what drives the prediction.")

    inputs = collect_inputs(artifacts["category_options"], artifacts["numeric_ranges"])
    customer = pd.DataFrame([inputs])[artifacts["input_columns"]]

    probability = artifacts["pipeline"].predict_proba(customer)[0, 1]
    tier = assign_tiers(np.array([probability]), artifacts)[0]
    flagged = probability >= artifacts["threshold"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{probability:.0%}")
    col2.markdown(f"**Risk level**\n\n:{TIER_COLORS[tier]}[**{tier}**]")
    col3.markdown(f"**Contact this customer?**\n\n{'Yes' if flagged else 'No'}")
    st.info(f"**Recommended action:** {TIER_ACTIONS[tier]}")

    grouped = explain_many(customer, artifacts).iloc[0]
    raising = grouped[grouped >= MIN_EFFECT].sort_values(ascending=False).head(3)
    lowering = grouped[grouped <= -MIN_EFFECT].sort_values().head(3)

    st.subheader("Why this prediction?")
    left, right = st.columns(2)
    with left:
        st.markdown("**Raising churn risk**")
        for group in raising.index:
            st.markdown(f"- {driver_label(group, inputs)}")
        if raising.empty:
            st.markdown("- No major risk factors")
    with right:
        st.markdown("**Lowering churn risk**")
        for group in lowering.index:
            st.markdown(f"- {driver_label(group, inputs)}")
        if lowering.empty:
            st.markdown("- No major protective factors")

    st.pyplot(plot_drivers(grouped, inputs))


# ---------- Batch upload ----------

def template_csv(artifacts):
    columns = ["customerID"] + artifacts["input_columns"]
    return pd.DataFrame(TEMPLATE_ROWS, columns=columns).to_csv(index=False)


def clean_and_validate(raw, artifacts):
    """Return (clean data, list of error messages). Scoring only happens if there are no errors."""
    data = raw.copy()
    data.columns = data.columns.str.strip()
    errors = []

    missing = [col for col in artifacts["input_columns"] if col not in data.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"]
    if len(data) == 0:
        return None, ["The file has no rows."]
    if len(data) > MAX_ROWS:
        return None, [f"The file has {len(data):,} rows. The limit is {MAX_ROWS:,}."]

    row_numbers = pd.Series(np.arange(1, len(data) + 1), index=data.index)

    for col in artifacts["category_options"]:
        if col in NUMERIC_COLUMNS:
            continue
        values = data[col].astype(str).str.strip()
        data[col] = values
        allowed = artifacts["category_options"][col]
        bad = ~values.isin(allowed)
        if bad.any():
            rows = ", ".join(map(str, row_numbers[bad].head(5)))
            examples = ", ".join(f"'{v}'" for v in values[bad].unique()[:3])
            errors.append(f"**{col}**: unknown value(s) {examples} in row(s) {rows}"
                          f"{'…' if bad.sum() > 5 else ''}. Allowed: {', '.join(map(str, allowed))}.")

    # Same rule as the notebook's cleaning: blank TotalCharges for new customers (tenure 0) means $0
    total = data["TotalCharges"].astype(str).str.strip()
    tenure = pd.to_numeric(data["tenure"], errors="coerce")
    data["TotalCharges"] = total.where(~(total.eq("") & tenure.eq(0)), "0")

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[col], errors="coerce")
        bad = converted.isna()
        if bad.any():
            rows = ", ".join(map(str, row_numbers[bad].head(5)))
            errors.append(f"**{col}**: missing or non-numeric value(s) in row(s) {rows}"
                          f"{'…' if bad.sum() > 5 else ''}.")
        data[col] = converted

    bad_senior = ~data["SeniorCitizen"].isin([0, 1]) & data["SeniorCitizen"].notna()
    if bad_senior.any():
        rows = ", ".join(map(str, row_numbers[bad_senior].head(5)))
        errors.append(f"**SeniorCitizen**: must be 0 or 1 (row(s) {rows}).")

    return data, errors


def score_batch(data, artifacts):
    customers = data[artifacts["input_columns"]]
    probabilities = artifacts["pipeline"].predict_proba(customers)[:, 1]
    grouped = explain_many(customers, artifacts)

    if "customerID" in data.columns:
        ids = data["customerID"].astype(str)
    else:
        ids = pd.Series([f"Row {i}" for i in range(1, len(data) + 1)], index=data.index)

    drivers = grouped.apply(lambda row: ", ".join(row[row > 0].nlargest(2).index), axis=1)

    results = pd.DataFrame({
        "Customer ID": ids,
        "Churn Probability": probabilities.round(2),
        "Risk Level": assign_tiers(probabilities, artifacts),
        "Contact": np.where(probabilities >= artifacts["threshold"], "Yes", "No"),
        "Key Churn Drivers": drivers.replace("", "No major risk factor"),
        "Monthly Charge": data["MonthlyCharges"],
        "Contract": data["Contract"],
        "Tenure (months)": data["tenure"],
        "Expected Monthly Loss": (probabilities * data["MonthlyCharges"]).round(2),
    }, index=data.index)
    return results.sort_values("Churn Probability", ascending=False)


def batch_page(artifacts):
    st.write("Upload a CSV of customers to score them all at once. The file needs the same columns "
             "as the IBM Telco dataset. `customerID` is optional and `Churn` is ignored.")

    st.download_button("Download template CSV", template_csv(artifacts),
                       file_name="churn_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload customer CSV", type="csv")
    st.caption("Files are processed in memory and not stored. "
               "Please don't upload real personal customer data to this public demo.")

    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded, dtype={"customerID": str})
    except Exception as error:
        st.error(f"Could not read the file as CSV: {error}")
        return

    data, errors = clean_and_validate(raw, artifacts)
    if errors:
        st.error("The file has problems. Please fix them and upload again.")
        for message in errors:
            st.markdown(f"- {message}")
        return

    results = score_batch(data, artifacts)
    flagged = results["Contact"] == "Yes"

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers scored", f"{len(results):,}")
    col2.metric("Flagged for contact", f"{flagged.sum():,} ({flagged.mean():.0%})")
    col3.metric("Expected monthly revenue at risk",
                f"${results.loc[flagged, 'Expected Monthly Loss'].sum():,.0f}")

    st.subheader("Risk tiers")
    tiers = results.groupby("Risk Level", observed=False).agg(
        customers=("Customer ID", "size"),
        avg_probability=("Churn Probability", "mean"),
        monthly_revenue=("Monthly Charge", "sum"),
        expected_loss=("Expected Monthly Loss", "sum"),
    )
    tiers.columns = ["Customers", "Avg. probability", "Monthly revenue", "Expected monthly loss"]
    tiers["Action"] = [TIER_ACTIONS[tier] for tier in tiers.index]
    st.dataframe(tiers.style.format({
        "Avg. probability": "{:.0%}", "Monthly revenue": "${:,.0f}", "Expected monthly loss": "${:,.0f}",
    }, na_rep="–"), width="stretch")

    st.subheader("Customers ranked by churn risk")
    st.dataframe(results, hide_index=True, width="stretch")
    st.download_button("Download results CSV", results.to_csv(index=False),
                       file_name="churn_predictions.csv", mime="text/csv")


# ---------- App ----------

def main():
    st.set_page_config(page_title="Churn Risk Predictor", layout="wide")
    st.title("Customer Churn Risk Predictor")

    artifacts = load_artifacts()
    mode = st.sidebar.radio("Mode", ["Single customer", "Batch upload (CSV)"])
    st.sidebar.divider()

    if mode == "Single customer":
        single_customer_page(artifacts)
    else:
        st.sidebar.markdown("Upload a CSV in the main panel. Use the template for the expected format.")
        batch_page(artifacts)

    st.caption("Explanations are SHAP values from the Logistic Regression model, compared with the "
               "average training customer. They show what the model associates with churn, not "
               "proven causes. Model trained on the IBM Telco Customer Churn sample dataset.")


if __name__ == "__main__":
    main()
