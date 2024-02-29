"""
Python code of the mitigation control support tool MiCo

Copyright (c) 2024. Royal Boskalis, Niels Roeders, Lukas Teuber
"""

import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from betapert import mpert
from criticalpath import Node
from preferendus import GeneticAlgorithm
from scipy.interpolate import pchip_interpolate
from scipy.stats import bernoulli
from scipy.stats import norm

from plotcrit import calculate_activity_dates
from preference_functions import PreferenceCurve

warnings.filterwarnings("ignore")


# Import ###############################################################################
# Import project data from Excel file, change to relevant file
data_file = "demonstration_case_data.xlsx"

# Read Excel file and pick specific sheets
data_act = pd.read_excel(data_file, skiprows=2, sheet_name="Activities")
data_mit = pd.read_excel(data_file, skiprows=2, sheet_name="Mitigations")
data_risk = pd.read_excel(data_file, skiprows=2, sheet_name="Risks")

# Activities
activities = pd.DataFrame(
    {
        "activityID": data_act["Activity ID"],
        "activitydescription": data_act["Activity description"],
        "durationopt": data_act["Optimistic"],
        "durationml": data_act["Most-Likely"],
        "durationpes": data_act["Pessimistic"],
        "durationdep": data_act["Predecessor activity"],
    }
)

# Mitigation
objectives = data_mit.shape[1] - 7
mitigation = pd.DataFrame(
    {
        "mitigationID": data_mit["Mitigation ID"],
        "mitigationdescription": data_mit["Mitigation measure"],
        "capacityopt": data_mit["Minimum"],
        "capacitynml": data_mit["Most likely"],
        "capacitypes": data_mit["Maximum"],
        "activityID": data_mit["Relations"],
        "dependency": data_mit["dependency factor (eta)"],
    }
)

for i in range(objectives):
    mitigation[f"mitact{i+1}"] = data_mit[
        f"Objective {i+1}"
    ]  # Calculates all objective value

# Risk
risk = pd.DataFrame(
    {
        "riskID": data_risk["Risk event ID"],
        "riskdescription": data_risk["Risk event description"],
        "riskopt": data_risk["Minimum"],
        "risknml": data_risk["Most likely"],
        "riskpes": data_risk["Maximum"],
        "activityID": data_risk["Affected activities"],
        "riskprob": data_risk["Risk probability"],
    }
)


# Data parsing #########################################################################
def random_number(low, middle, high, scale_factor=4):
    """
    Random number generator based on beta-PERT distribution
    :param low: minimal distribution number
    :param middle: Mean distribution number
    :param high: Maximum distribution number
    :param scale_factor: shaping parameter for modified beta-PERT (standard=4)
    :return: Random number
    """

    if low == middle == high:
        return int(middle)
    else:
        mdist = mpert(low, middle, high, lambd=scale_factor)
        try:
            return int(mdist.rvs())
        except ValueError:
            return int(middle)


def random_number_act(row):
    return random_number(
        row["durationopt"], row["durationml"], row["durationpes"], scale_factor=4
    )


def activities_delay(row):
    return row["durationact"] - row["durationml"]


def delay_occur(row):
    return row["delay"] > 0


def random_number_mit(row):
    return random_number(
        row["capacityopt"], row["capacitynml"], row["capacitypes"], scale_factor=4
    )


def random_number_risk(row):
    return random_number(row["riskopt"], row["risknml"], row["riskpes"], scale_factor=4)


def occurence_risk(row):
    return bool(bernoulli.rvs(row["riskprob"]))


def total_time(row):
    return row["durationact"] + (row["riskact"] * row["riskoccur"]) + row["delay"]


def risk_imp(row):
    return row["riskoccur"] * row["riskact"]


# Critical path algorithm ##############################################################
def critical_path(activities, mitigation, risk):
    def find_critical_path(df, duration_col):
        activities = df.copy()
        nactivities = len(activities["activityID"])

        # Create the project node
        project = Node("project")

        # Create nodes for each activity and add them to the project
        for i in range(nactivities):
            activities.loc[i, "activitydescription"] = project.add(
                Node(
                    list(activities["activitydescription"])[i],
                    duration=list(activities[duration_col])[i],
                )
            )

        # Link the activities based on the durationdep column
        for i in range(nactivities):
            duration_dep = list(activities["durationdep"])[i]
            # Check if for dependencies
            if duration_dep != 0:
                # Check if there are multiple dependencies
                if isinstance(duration_dep, str):
                    split = duration_dep.split(",")
                    for j in range(len(split)):
                        a = activities.loc[int(split[j]) - 1, "activitydescription"]
                        b = activities.loc[i, "activitydescription"]
                        project.link(a, b)
                else:
                    a = activities.loc[duration_dep - 1, "activitydescription"]
                    b = activities.loc[i, "activitydescription"]
                    project.link(a, b)

        # Update the project to calculate the critical path
        project.update_all()

        # Get the critical path
        crit_path = project.get_critical_path()

        # Filter the DataFrame to include only the activities in the critical path
        critical_df = activities[activities["activitydescription"].isin(crit_path)]

        # Drop all columns except activityID and activitydiscription
        critical_df = critical_df[
            [
                "activityID",
                "delayocc",
                "activitydescription",
                duration_col,
                "riskoccur",
                "mitigationID",
                "capacityact",
                "mitcost1",
                "mitcost2",
                "mitcost3",
                "riskact",
            ]
        ]

        # Calculate the sum of the durations of activities on the critical path
        critical_path_duration = critical_df[duration_col].sum()

        return crit_path, critical_df, critical_path_duration

    # draw random numbers for activities, mitigations, and risks
    activities["durationact"] = activities.apply(random_number_act, axis=1)
    activities["delay"] = activities.apply(activities_delay, axis=1)
    activities["delayocc"] = activities.apply(delay_occur, axis=1)
    mitigation["capacityact"] = mitigation.apply(random_number_mit, axis=1)
    risk["riskact"] = risk.apply(random_number_risk, axis=1)

    # draw if a risk occours or not (boolean)
    risk["riskoccur"] = risk.apply(occurence_risk, axis=1)

    # Calculates the objectives value for each mitigation measure
    for i in range(objectives):
        mitigation[f"mitcost{i + 1}"] = (
            mitigation[f"mitact{i + 1}"]
            * mitigation[f"capacityact"]
            * mitigation[f"dependency"]
        )

    # Create merged dataframe of activities and related mitigations and risks
    merged_df = pd.merge(activities, mitigation, on="activityID", how="left")
    merged_df = pd.merge(merged_df, risk, on="activityID", how="left")
    merged_df = merged_df.fillna(0)
    merged_df["durationtot"] = merged_df.apply(total_time, axis=1)

    # Find critical path without risk, and with risk

    # Network cosntruction with risk
    crit_path_nr, critical_df_nr, critical_path_duration_nr = find_critical_path(
        merged_df, "durationml"
    )

    crit_path_r, critical_df_r, critical_path_duration_r = find_critical_path(
        merged_df, "durationtot"
    )
    delay = critical_path_duration_r - critical_path_duration_nr
    dv = critical_df_r.loc[
        (
            critical_df_r["delayocc"].eq(True) | critical_df_r["riskoccur"].eq(True)
        ).idxmax() :
    ]
    dv = dv.drop(
        ["activityID", "activitydescription", "durationtot", "riskoccur", "delayocc"],
        axis="columns",
    )  # normal drop if risk is inside network construction

    """
    Network construction without risk, the following lines should only be used for 
    verification against:
    -   Kammouh, O., Nogal, M., Binnekamp, R., & Wolfert, R. (2021, August). Mitigation 
    Controller: Adaptive Simulation Approach for Planning Control Measures in Large 
    Construction Projects. Journal of Construction Engineering and Management, 147 (8),
    04021093. Retrieved from 
    https://ascelibrary.org/doi/10.1061/%28ASCE%29CO.1943-7862.0002126
    """

    # crit_path_r, critical_df_r, critical_path_duration_r = find_critical_path(
    #     merged_df, "durationact"
    # )
    # delay = (
    #     critical_path_duration_r
    #     - critical_path_duration_nr
    #     + critical_df_r.apply(risk_imp, axis=1).sum()
    # )
    # dv = critical_df_r.loc[
    #     (
    #         critical_df_r["delayocc"].eq(True) | critical_df_r["riskoccur"].eq(True)
    #     ).idxmax() :
    # ]
    # dv = dv.drop(
    #     [
    #         "activityID",
    #         "activitydescription",
    #         "durationact",
    #         "riskoccur",
    #         "delayocc",
    #         "riskact",
    #     ],
    #     axis="columns",
    # )

    dv = dv[(dv != 0).all(1)]

    return delay, critical_df_r, merged_df, critical_path_duration_nr, dv, crit_path_nr


(
    delay,
    critical_df_r,
    merged_df,
    critical_path_duration_nr,
    dv,
    crit_path_nr,
) = critical_path(activities, mitigation, risk)

# Extract critical path
_, _, merged_df, critical_path_duration, _, crit_path = critical_path(
    activities, mitigation, risk
)

# Plot
calculate_activity_dates(
    merged_df,
    "activityID",
    "activitydescription",
    "durationml",
    "durationdep",
    crit_path,
)


# Objective functions ##################################################################
def objective_cost(variables):
    """
    Objective to minimize cost.

    :param variables: array with design variable values per member of the population.
    Can be split by using array slicing.
    :return: 1D-array with the calculated costs for the members of the population.
    """

    # amount of design variables
    num_variables = variables.shape[1]

    # initiate initial cost, time and duration
    cost = np.zeros(len(variables))
    time = np.zeros(len(variables))
    t_delayed = np.ones(len(variables)) * (critical_path_duration_nr + delay)
    t_target = np.ones(len(variables)) * critical_path_duration_nr

    # iterate over all variabels (mitigation measures)
    for i in range(num_variables):
        x = variables[:, i]
        # index 0 is for the mitigation id (can iterate over the length of mitigation
        # measures)
        factor_id = int(list(dv["mitigationID"])[i])

        # Get related value of mitigation measure out of dataframe
        factor_cost = float((dv.iloc[:, 2].loc[dv["mitigationID"] == factor_id]).values)
        factor_time = float((dv.iloc[:, 1].loc[dv["mitigationID"] == factor_id]).values)

        # Add up cost by variabel (boolean) times mitigation cost
        cost += x * factor_cost
        time += x * factor_time

    # Calculate penalty and incentive
    # Calculate mitigation delta: delayed time - mitigation time - target time
    delta = (t_delayed - time) - t_target
    # if delta is positive: penalty for late completion
    dneg = np.where(delta > 0, delta, 0)
    # if delta is positive: reward for early completion
    dpos = np.where(delta < 0, t_target - (t_delayed - time), 0)

    # minimize (sum(binary * mitigation cost) + delay * penalty - reduction * incentive)
    return cost + dneg * penalty - dpos * incentive


def objective_time(variables):
    """
    Objective to minimize time.

    :param variables: array with design variable values per member of the population.
    Can be split by using array slicing.
    :return: 1D-array with the calculated costs for the members of the population.
    """

    # amount of design variables
    num_variables = variables.shape[1]

    # set the delayed timeframe as initial value
    time = np.ones(len(variables)) * (critical_path_duration_nr + delay)

    for i in range(num_variables):
        x = variables[:, i]
        factor_id = int(
            list(dv["mitigationID"])[i]
        )  # index 0 is for the mitigation id (can iterate over the length of
        # mitigation measures)
        factor = float((dv.iloc[:, 1].loc[dv["mitigationID"] == factor_id]).values)
        time += -x * factor

    return time


def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions
    that are declared above. Objective can be used both with IMAP as with the minmax
    aggregation method. Declare which to use by the method argument.

    :param variables: array with design variable values per member of the population.
    Can be split by using array slicing
    :return: 1D-array with aggregated preference scores for the members of the
    population
    """

    # calculate objectives
    cost = objective_cost(variables)
    time = objective_time(variables)

    # calculate preference scores based on objective values
    p_cost = pchip_interpolate(x_points_1, p_points_1, cost)
    p_time = pchip_interpolate(x_points_2, p_points_2, time)

    # initiate solver
    return [w1, w2], [p_cost, p_time]


# IMAP and preference functions ########################################################
def IMAP(
    pref1,
    pref2,
    critical_path_duration_nr,
    delay,
    objective,
    cons,
    bounds,
    method,
    npop=150,
    n_runs=5,
):
    print("Run IMAP")

    # make dictionary with parameter settings for the GA
    options = {
        "n_bits": 20,
        "n_iter": 400,
        "n_pop": npop,
        "r_cross": 0.85,
        "max_stall": 10,
        "aggregation": "IMAP",
        "var_type_mixed": ["int" for _ in range(len(dv))],
        "lsd_aggregation": method,
    }

    save_array_IMAP = list()  # list to save the results from every run to

    # initiate GA
    ga = GeneticAlgorithm(
        objective=objective, constraints=cons, bounds=bounds, options=options
    )

    # run the GA and print its result
    for i in range(n_runs):
        # print(f'Initialize run {i + 1}')
        score_IMAP, design_variables_IMAP, _ = ga.run(verbose=False)

        # print(
        #     f"Optimal mitigation combination: result for type = "
        #     f"{design_variables_IMAP} and score of {score_IMAP}."
        # )
        # print()

        save_array_IMAP.append(design_variables_IMAP)
        # print(f'Finished run {i + 1}')

    x_points_1 = pref1[0]
    x_points_2 = pref2[0]
    p_points_1 = pref1[1]
    p_points_2 = pref2[1]

    # arrays for plotting continuous preference curves
    c1 = np.linspace(min(x_points_1), max(x_points_1))
    c2 = np.linspace(min(x_points_2), max(x_points_2))

    # calculate the preference functions
    p1 = pchip_interpolate(x_points_1, p_points_1, c1)
    p2 = pchip_interpolate(x_points_2, p_points_2, c2)

    # # make numpy array of results, to allow for array splicing
    variable = np.array(save_array_IMAP)

    # calculate individual preference scores for the results of the GA
    c1_res = objective_cost(variable)
    c2_res = objective_time(variable)

    p1_res = pchip_interpolate(x_points_1, p_points_1, c1_res)
    p2_res = pchip_interpolate(x_points_2, p_points_2, c2_res)

    """
    Create figure that plots all preference curves and the preference scores of the
    returned results of the GA. Only enable the following code if MC iteration are very 
    low otherwise you could run into 'Too Many Requests' error
    """

    # Display project duration
    # print(f'Project duration without Delay: {critical_path_duration_nr} days.')
    # print(f'Delay of {delay} days.')  # Display delay
    # display(dv)  # Display possible mitigation measures
    # print(f'Mitigated by {mit_time_MC} days.')  # Print the mitigated time
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    #
    # ax1.plot(c1, p1, zorder=3, label='Preference curve')
    # ax1.scatter(c1_res, p1_res, label='Optimal solution Tetra', marker='D', color='g')
    # ax1.set_xlim((min(x_points_1)-100, max(x_points_1)+50))
    # ax1.set_ylim((0, 105))
    # ax1.set_title('Cost')
    # ax1.set_xlabel('Euro')
    # ax1.set_ylabel('Preference score')
    # ax1.legend()
    # ax1.grid()
    #
    # ax2.plot(c2, p2, zorder=3, label='Preference curve')
    # ax2.scatter(c2_res, p2_res, label='Optimal solution Tetra', marker='D', color='g')
    # ax2.axvline(critical_path_duration_nr)
    # ax2.axvline(critical_path_duration_nr+delay)
    # ax2.set_xlim((min(x_points_2)-100, max(x_points_2)+50))
    # ax2.set_ylim((0, 105))
    # ax2.set_title('Time')
    # ax2.set_xlabel('Days')
    # ax2.legend()
    # ax2.grid()

    return variable


# Optimisation setting #################################################################

# set weights for the different objectives
w1 = 0.70  # cost
w2 = 0.30  # time

# define preferences cost (euros)
cost_bound = 0

# define preference time (days)
dur_shorter = 1000  # maximum allowable time of delivering project early

# Set penalty per day of delay and incentive per day of finishing early (in cost)
penalty = 0  # penalty for late completion
incentive = 0  # reward for early completion

# MC simulation ########################################################################

# Set number of simulations
n = 100

# Initiate timer
start_time = time.time()

# set empty list for respective objectives
mitigation_list = list()  # mitigation activities
activity_list = list()  # activities

all_crit_time = list()  # time all available measures (critical path)
opt_mit_time = list()  # time optimal mesures
perm_mit_time = list()  # time all mitigation measures
no_mit_time = list()  # time no measures

all_crit_cost = list()  # cost all available measures (critical path)
opt_mit_cost = list()  # cost optimal measures
perm_mit_cost = list()  # cost all mitigation measures

# set count for simulations without delay
count = 0

for i in range(n):
    # Find critical path and the delay
    delay, critical_df, merged_df, critical_path_duration_nr, dv, _ = critical_path(
        activities, mitigation, risk
    )

    # Check if max cost preference is given or not
    if cost_bound == 0:
        p_cost_min = sum(
            dv["mitcost1"]
        )  # if not given use the maximum possible mitigation cost of that run
    else:
        p_cost_min = cost_bound  # use specified cost boundary

    # Set preference curves
    time_p = PreferenceCurve(dur_shorter, critical_path_duration_nr, delay)
    cost_p = PreferenceCurve(0.01, 0.01, p_cost_min)
    time_p.beta_PERT()
    cost_p.linear()

    x_points_1, p_points_1 = [list(cost_p.x_values), list(cost_p.y_values)]  # cost
    x_points_2, p_points_2 = [list(time_p.x_values), list(time_p.y_values)]  # time
    pref1 = [x_points_1, p_points_1]
    pref2 = [x_points_2, p_points_2]

    # if there is a delay, start the MC simulation
    if delay > 0:
        print(f"MC progress: {(((i+1)/n) * 100):.1f}%")  # progress statement

        # Define boundary for all mitigation measures
        bounds = [[0, 1] for _ in range(len(dv))]

        # Define constraints
        cons = []

        # run IMAP optimization algorithm
        variable = IMAP(
            pref1,
            pref2,
            critical_path_duration_nr,
            delay,
            objective,
            cons,
            bounds,
            method="fast",
            npop=len(dv) * 10,
            n_runs=1,
        )

        # Store results
        mitigation_list.append(
            ((list(dv["mitigationID"]) * variable).ravel()).tolist()
        )  # Mitigation measures
        activity_list.append(list(critical_df["activityID"]))  # Activities

        # Time
        total_time_MC = (
            critical_path_duration_nr + delay
        )  # original time with delay (no mitigation)
        mit_time_MC = sum((list(dv["capacityact"]) * variable).ravel())

        all_crit_time.append(
            total_time_MC - (dv["capacityact"]).sum()
        )  # time all available measures (critical path)
        opt_mit_time.append(total_time_MC - mit_time_MC)  # time optimal mesures
        perm_mit_time.append(
            total_time_MC - (merged_df["capacityact"]).sum()
        )  # time all mitigation measures
        no_mit_time.append(total_time_MC)  # time no measures

        # Cost
        mit_cost_MC = sum((list(dv["mitcost1"]) * variable).ravel())

        all_crit_cost.append(
            (dv["mitcost1"]).sum()
        )  # cost all available measures (critical path)
        opt_mit_cost.append(mit_cost_MC)  # cost optimal measures
        perm_mit_cost.append(
            merged_df["mitcost1"].sum()
        )  # cost all mitigation measures

    # If there is no delay add to the count
    else:
        count += 1

# End of MC
print(
    f"---------- MONTE CARLO IS FINISHED AFTER {n} ITERATIONS AND "
    f"{round((time.time() - start_time)/60)} MINUTES,"
    f"{count} RUN(S) WITHOUT DELAY-----------"
)

# Visualization of results #############################################################


def plot_all_combinations(combinations, figsize=(20, 10)):
    # Convert floats to integers, ignore zeros, and flatten the list of combinations
    flattened_combinations = [
        tuple(int(num) for num in comb if num != 0.0) for comb in combinations
    ]

    # Count the frequency of each combination
    frequencies = Counter(flattened_combinations)

    # Prepare data for the bar graph
    data = [
        (comb, value / len(combinations) * 100) for comb, value in frequencies.items()
    ]  # Normalize to percentage

    # Sort data by percentage in descending order
    data.sort(key=lambda x: x[1], reverse=True)

    # Separate labels and values
    labels = [" & ".join(map(str, comb)) for comb, _ in data]
    labels = [value if value != "" else "None" for value in labels]
    values = [value for _, value in data]

    # Create the bar graph
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="grey", edgecolor="black")
    ax.yaxis.grid(True)
    ax.set_title(f"Distribution for all {len(values)} occurring combinations")
    ax.set_xlabel("Mitigation measures combinations", fontsize=16)
    ax.set_ylabel("Percentage of mitigation combination occurrence (%)", fontsize=16)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_selected_combinations(combinations, threshold, figsize=(20, 10)):
    # Convert floats to integers, ignore zeros, and flatten the list of combinations
    flattened_combinations = [
        tuple(int(num) for num in comb if num != 0.0) for comb in combinations
    ]

    # Count the frequency of each combination
    frequencies = Counter(flattened_combinations)

    # Prepare data for the bar graph
    data = []
    for comb, value in frequencies.items():
        percentage = value / len(combinations) * 100
        if percentage > threshold:
            data.append((" & ".join(map(str, comb)), percentage))

    # Sort data by percentage in descending order
    data.sort(key=lambda x: x[1], reverse=True)

    # Separate labels and values
    labels, values = zip(*data)
    labels = [value if value != "" else "None" for value in labels]

    # Create the bar graph
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="grey", edgecolor="black")
    ax.yaxis.grid(True)
    ax.set_title(
        f"Distribution of combinations with threshold of {threshold}% occurrence"
    )
    ax.set_xlabel("Mitigation measures combinations", fontsize=16)
    ax.set_ylabel("Percentage of mitigation combination occurrence (%)", fontsize=16)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_all_elements(combinations, name, figsize=(20, 10)):
    # Flatten the list of combinations
    elements = [num for comb in combinations for num in comb if num != 0.0]

    # Get unique elements
    unique_elements = set(elements)

    # Count the frequency of each element
    frequencies = Counter(elements)

    # Get all available mitigation measures
    available_measures = set(mitigation["mitigationID"])

    # Add missing measures with a frequency of 0
    for measure in available_measures:
        if measure not in frequencies:
            frequencies[measure] = 0

    # Prepare data for the bar graph
    data = [
        (element, value / (n - count) * 100) for element, value in frequencies.items()
    ]

    # Sort data by element in ascending order
    data.sort(key=lambda x: x[0])

    # Separate labels and values
    labels = [str(element) for element, _ in data]
    values = [value for _, value in data]

    # Create the bar graph
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="grey", edgecolor="black")
    ax.yaxis.grid(True)
    ax.set_title(f"Criticality index of {name}", fontsize=20)
    ax.set_xlabel(name, fontsize=16)
    ax.set_ylabel("Percentage of element occurrence (%)", fontsize=20)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
    fig.tight_layout()
    plt.show()


# Create plots
plot_all_combinations(mitigation_list)
plot_selected_combinations(mitigation_list, 2.5)
plot_all_elements(mitigation_list, "Mitigation measures")
plot_all_elements(activity_list, "Activities")

# S-curves objectives ##################################################################


def plot_cdf(
    list1, list2, list3, plot_name="CDF Plot", figsize=(20, 10), xaxis="values", vline=0
):
    # Calculate the CDF for each list
    cdf1 = np.cumsum(np.sort(list1)) / np.sum(list1)
    cdf2 = np.cumsum(np.sort(list2)) / np.sum(list2)
    cdf3 = np.cumsum(np.sort(list3)) / np.sum(list3)

    # Create the figure and add subplots for each CDF
    fig, ax = plt.subplots(figsize=figsize)

    # Plot CDFs
    ax.axvline(x=vline, color="black", linewidth=1, label="Original planning")
    ax.plot(
        np.sort(list1),
        cdf1,
        color="red",
        linewidth=1,
        label="All mitigation measures (critical path)",
    )
    ax.plot(
        np.sort(list2),
        cdf2,
        color="green",
        linewidth=1,
        label="Optimized mitigation measures",
    )
    ax.plot(
        np.sort(list3), cdf3, color="blue", linewidth=1, label="No mitigation measures"
    )

    # Set plot labels and legend
    ax.set_title(plot_name, fontsize=20)
    ax.set_xlabel(xaxis, fontsize=16)
    ax.set_ylabel("Cumulative Probability", fontsize=16)
    ax.legend(loc="lower right")

    # Set font size
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add grid
    ax.grid(True)

    # Show the plot
    plt.show()


def plot_pdf(
    list1, list2, list3, plot_name="PDF Plot", figsize=(20, 10), xaxis="Cost [$]"
):
    mu1, std1 = norm.fit(list1)
    x1 = np.linspace(min(list1), max(list1), 100)
    pdf1 = norm.pdf(x1, mu1, std1)

    mu2, std2 = norm.fit(list2)
    x2 = np.linspace(min(list2), max(list2), 100)
    pdf2 = norm.pdf(x2, mu2, std2)

    mu3, std3 = norm.fit(list3)
    x3 = np.linspace(min(list3), max(list3), 100)
    pdf3 = norm.pdf(x3, mu3, std3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x1, pdf1, color="black", label="PDF Permanent cost (all measures)")
    ax.hist(
        list1,
        bins=50,
        density=True,
        alpha=0.2,
        color="purple",
        label="Histogram Permanent cost",
    )
    ax.plot(x2, pdf2, color="grey", label="PDF Permanent cost (measures critical path)")
    ax.hist(
        list2,
        bins=50,
        density=True,
        alpha=0.2,
        color="red",
        label="Histogram Permanent cost (measures critical path)",
    )
    ax.plot(x3, pdf3, color="black", label="PDF Optimised cost (IMAP)")
    ax.hist(
        list3,
        bins=50,
        density=True,
        alpha=0.2,
        color="green",
        label="Histogram Optimised cost",
    )

    # Set plot labels and legend
    ax.set_title(plot_name, fontsize=20)
    ax.set_xlabel(xaxis, fontsize=16)
    ax.set_ylabel("Probability Distribution Function", fontsize=16)
    ax.legend(loc="upper right")

    # Set font size
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add grid
    ax.grid(True)

    # Show the plot
    plt.show()


# Plot S curve for objective time
plot_cdf(
    all_crit_time,
    opt_mit_time,
    no_mit_time,
    plot_name="S-curve objective time",
    xaxis="Time [d]",
    vline=critical_path_duration,
)

# Plot PDF and histogram for objective cost
plot_pdf(
    perm_mit_cost,
    all_crit_cost,
    opt_mit_cost,
    plot_name="PDF objective cost",
    xaxis="Cost [$]",
)

# Plot CDF for objective cost
plot_cdf(
    all_crit_cost,
    opt_mit_cost,
    perm_mit_cost,
    plot_name="CDF objective cost",
    xaxis="Cost [$]",
)
