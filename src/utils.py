import pandas as pd

def lift_summary(df_ev, col_target="class", col_pred="dist", qs = [0.0, 0.8, 0.9, 0.95, 0.99, 1.0]):
    """"Calculates the lift at different cuts, returns a Pandas DataFrame with that info

    Args:
        df_ev: Pandas Dataframe under evaluation
        col_target: Name of the column with the target variable
        col_pred: Name of the column with the prediction
        qs: quantiles used to set the cuts at which the lift will be estimated

    Returns:
        Lift summary in a Pandas DataFrame
    """
    sample_mean = df_ev[col_target].sum() / len(df_ev)
    df_lift = df_ev.sort_values(by="dist", ascending=False)
    points = [100.*(1 - q) for q in qs[:-1]]
    cuts = df_lift[col_pred].quantile(qs)
    df_lift[col_pred + "_cut"] = pd.cut(df_lift[col_pred], cuts)
    sum_series = df_lift.groupby(col_pred + "_cut")[col_target].sum()
    size_series = df_lift.groupby(col_pred + "_cut")[col_target].size()
    df_lift_out = pd.DataFrame({"Nb": size_series,
                                "Sum": sum_series,
                                "Pcts": points})

    df_lift_out = df_lift_out.sort_values(by=col_pred + "_cut", ascending=False)
    df_lift_out = df_lift_out.reset_index()

    df_lift_out["Cum_sum"] = df_lift_out["Sum"].cumsum()
    df_lift_out["Mean_acum"] = df_lift_out["Sum"] / df_lift_out["Nb"]
    df_lift_out["Lift"] = df_lift_out["Mean_acum"] / sample_mean

    df_lift_out = df_lift_out.drop(columns = ["dist_cut", "Sum", "Mean_acum"])
    df_lift_out = df_lift_out[["Pcts", "Nb", "Cum_sum", "Lift"]]

    return df_lift_out