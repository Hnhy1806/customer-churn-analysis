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


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACTS_PATH)


def collect_inputs(options, ranges):
    """Sidebar form. Keeps dependent fields consistent with the training data."""
    st.sidebar.header("Customer details")

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


def risk_tier(probability, bins, labels):
    for lower, upper, label in zip(bins[:-1], bins[1:], labels):
        if lower <= probability < upper:
            return label
    return labels[-1]


def explain(customer, artifacts):
    """Grouped SHAP values for one customer: weight × (customer value − training average)."""
    pipe = artifacts["pipeline"]
    x = pipe.named_steps["preprocess"].transform(customer)[0]
    coef = pipe.named_steps["model"].coef_[0]
    contributions = pd.Series(coef * (x - artifacts["feature_means"]),
                              index=artifacts["feature_names"])
    return contributions.groupby(artifacts["feature_groups"]).sum()


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


def main():
    st.set_page_config(page_title="Churn Risk Predictor", layout="wide")
    st.title("Customer Churn Risk Predictor")
    st.write("Enter a customer's details in the sidebar to see their churn risk, "
             "the recommended action, and what drives the prediction.")

    artifacts = load_artifacts()
    inputs = collect_inputs(artifacts["category_options"], artifacts["numeric_ranges"])
    customer = pd.DataFrame([inputs])[artifacts["input_columns"]]

    probability = artifacts["pipeline"].predict_proba(customer)[0, 1]
    tier = risk_tier(probability, artifacts["risk_bins"], artifacts["risk_labels"])
    flagged = probability >= artifacts["threshold"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{probability:.0%}")
    col2.markdown(f"**Risk level**\n\n:{TIER_COLORS[tier]}[**{tier}**]")
    col3.markdown(f"**Contact this customer?**\n\n{'Yes' if flagged else 'No'}")
    st.info(f"**Recommended action:** {TIER_ACTIONS[tier]}")

    grouped = explain(customer, artifacts)
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

    st.caption("Explanations are SHAP values from the Logistic Regression model, compared with the "
               "average training customer. They show what the model associates with churn, not "
               "proven causes. Model trained on the IBM Telco Customer Churn sample dataset.")


if __name__ == "__main__":
    main()
