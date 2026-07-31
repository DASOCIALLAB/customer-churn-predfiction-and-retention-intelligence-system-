"""Retention campaign ROI economics."""


def calculate_roi(work, threshold, cost_per_customer, success_rate):
    targeted = work[work["churn_prob"] >= threshold]
    n_targeted = len(targeted)
    campaign_cost = n_targeted * cost_per_customer
    avg_charge = targeted["MonthlyCharges"].mean() if n_targeted > 0 else 0
    monthly_revenue_saved = n_targeted * success_rate * avg_charge
    net_roi = monthly_revenue_saved - campaign_cost
    return {
        "targeted": targeted,
        "n_targeted": n_targeted,
        "campaign_cost": campaign_cost,
        "avg_charge": avg_charge,
        "monthly_revenue_saved": monthly_revenue_saved,
        "net_roi": net_roi,
    }
