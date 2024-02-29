from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_activity_dates(
    df,
    activity_col,
    activity_name,
    duration_col,
    predecessors_col,
    crit_list,
    start_date_col="start_date",
    end_date_col="end_date",
):
    df = df.copy()
    # Convert 0 to None in the predecessors column within lists
    df[predecessors_col] = df[predecessors_col].apply(
        lambda x: [y if y != "0" else None for y in str(x).split(",")]
    )
    # Create columns to store start and end dates
    df[start_date_col] = None
    df[end_date_col] = None

    # Function to calculate start and end dates for each activity
    def calculate_dates():
        for activity_id, row in df.iterrows():
            if pd.isnull(row[start_date_col]):
                predecessors = row[predecessors_col]
                pred_end_dates = []
                for pred in predecessors:
                    if pred is not None and pred != "":
                        pred_id = int(pred)
                        pred_end_date = df.at[pred_id - 1, end_date_col]
                        if pred_end_date is not None:
                            pred_end_dates.append(pred_end_date)
                if pred_end_dates:
                    start_date = max(pred_end_dates) + timedelta(days=1)
                else:
                    start_date = datetime(2000, 1, 1, 6, 0)  # Default start date
                duration = row[duration_col]
                end_date = start_date + timedelta(days=duration)

                df.at[activity_id, start_date_col] = start_date
                df.at[activity_id, end_date_col] = end_date

    # Calculate dates for each activity
    calculate_dates()

    df["days_to_start"] = (
        pd.to_datetime(df[start_date_col]) - df[start_date_col].min()
    ).dt.days
    df["days_to_end"] = (
        pd.to_datetime(df[end_date_col]) - df[start_date_col].min()
    ).dt.days

    # Plotting function
    fig, ax = plt.subplots(figsize=(20, 10))

    # Color the critical activities in red
    crit_list = [str(item) for item in crit_list]
    crit_activities = df[df[activity_name].astype(str).isin(crit_list)]

    for i, row in df.iterrows():
        if row[duration_col] == 0:
            if row[activity_name] in crit_list:
                plt.plot(
                    row["days_to_start"] + 1,
                    row[activity_name],
                    "D",
                    markersize=6,
                    color="red",
                )
            else:
                plt.plot(
                    row["days_to_start"] + 1,
                    row[activity_name],
                    "D",
                    markersize=6,
                    color="blue",
                )
        else:
            plt.barh(
                y=row[activity_name],
                width=row[duration_col],
                left=row["days_to_start"] + 1,
                color="blue",
            )

    # Plot bars for critical activities
    plt.barh(
        y=crit_activities[activity_name],
        width=crit_activities[duration_col],
        left=crit_activities["days_to_start"] + 1,
        color="red",
    )

    plt.title("Project Management Gantt Chart without risk or delay", fontsize=15)
    plt.gca().invert_yaxis()

    xticks = np.arange(
        5, df["days_to_end"].max() + 2, 100
    )  # Define ticks at every 100 days interval
    xticklabels = pd.date_range(
        start=df[start_date_col].min() + timedelta(days=4), end=df[end_date_col].max()
    ).strftime("%d/%m")

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xticklabels[::100], rotation=45, ha="right"
    )  # Display ticks every 100 days

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))  # Format date on x-axis
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=1)
    )  # Show every month on x-axis

    ax.xaxis.grid(True, alpha=0.5)

    # Color the y-axis label red for critical activities
    for label in ax.get_yticklabels():
        if label.get_text() in crit_list:
            label.set_color("red")

    plt.tight_layout()
    plt.show()

    return
