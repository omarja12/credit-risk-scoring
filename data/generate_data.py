"""
Synthetic credit application data generator.

Generates a loan-applicant dataset with realistic, non-linear relationships
between applicant attributes and default risk, so that a scorecard trained
on it actually has to learn something rather than memorize noise.

This is 100% synthetic — no real applicant data, no proprietary data or
methodology from any employer is used here.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_applicants(n: int = 60000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(40, 12, n), 18, 75).round().astype(int)

    employment_length_years = np.clip(
        rng.gamma(shape=2.0, scale=3.5, size=n), 0, 40
    ).round(1)

    base_income = rng.lognormal(mean=10.6, sigma=0.45, size=n)
    income_age_bump = np.clip((age - 18) / 40, 0, 1) * 8000
    annual_income = np.clip(base_income + income_age_bump, 12000, 400000).round(0)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"], size=n, p=[0.42, 0.40, 0.18]
    )

    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car", "medical", "small_business"],
        size=n, p=[0.32, 0.24, 0.14, 0.12, 0.09, 0.09]
    )

    loan_amount = np.clip(rng.lognormal(mean=9.2, sigma=0.55, size=n), 1000, 60000).round(0)
    loan_term_months = rng.choice([36, 60], size=n, p=[0.65, 0.35])

    dti = np.clip(
        rng.beta(2.2, 5.0, size=n) * 60 + (loan_amount / annual_income) * 8, 0, 65
    ).round(1)

    num_credit_lines = np.clip(rng.poisson(6, n), 0, 25)
    credit_utilization = np.clip(rng.beta(2.0, 3.0, size=n) * 100, 0, 100).round(1)

    num_late_payments_2y = rng.poisson(
        lam=np.clip(0.15 + credit_utilization / 120 + (dti / 100), 0.05, 3.0)
    )
    months_since_last_delinquency = np.where(
        num_late_payments_2y == 0,
        999,
        rng.integers(1, 48, n)
    )
    has_bankruptcy_10y = (rng.random(n) < 0.045).astype(int)

    credit_history_years = np.clip(
        (age - 18) * rng.uniform(0.3, 0.9, n), 0.5, 55
    ).round(1)

    # ---- Latent default risk (logit) driven by the above, plus noise ----
    logit = (
        -3.1
        + 0.038 * (dti - 20)
        + 0.022 * (credit_utilization - 40)
        + 0.55 * np.log1p(num_late_payments_2y)
        + 0.9 * has_bankruptcy_10y
        - 0.22 * np.log1p(employment_length_years)
        - 0.65 * np.log1p(annual_income / 30000)
        - 0.15 * np.log1p(credit_history_years)
        + 0.10 * (loan_term_months == 60).astype(float)
        + np.where(home_ownership == "RENT", 0.18, 0.0)
        + np.where(loan_purpose == "small_business", 0.45, 0.0)
        + np.where(loan_purpose == "medical", 0.15, 0.0)
        + rng.normal(0, 0.55, n)  # idiosyncratic noise
    )
    pd_true = _sigmoid(logit)
    default = (rng.random(n) < pd_true).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_length_years": employment_length_years,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "dti": dti,
        "num_credit_lines": num_credit_lines,
        "credit_utilization": credit_utilization,
        "num_late_payments_2y": num_late_payments_2y,
        "months_since_last_delinquency": months_since_last_delinquency,
        "has_bankruptcy_10y": has_bankruptcy_10y,
        "credit_history_years": credit_history_years,
        "default": default,
    })
    return df


if __name__ == "__main__":
    out_path = "data/applicants.csv"
    df = generate_applicants()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")
