
"""Levene's test to assess if assumptions for ANOVA are met"""
import pandas as pd
from scipy import stats
import numpy as np

# Load data from CSV files
data1 = pd.read_csv('all_objective_results_5_500_1111111_years.csv')
data2 = pd.read_csv('all_objective_results_10_500_1111111_years.csv')
data3 = pd.read_csv('all_objective_results_15_500_1111111_years.csv')
data4 = pd.read_csv('all_objective_results_20_500_1111111_years.csv')

# Assuming each dataset has a single column of data
# If your data has column names, you may need to specify the correct column, e.g., data1['column_name']
data1 = data1.iloc[:, 0]
data2 = data2.iloc[:, 0]
data3 = data3.iloc[:, 0]
data4 = data4.iloc[:, 0]

# Descriptive statistics
print("Descriptive Statistics:")
for i, data in enumerate([data1, data2, data3, data4], 1):
    print(f"Group {i} - Mean: {np.mean(data):.2f}, Std Dev: {np.std(data, ddof=1):.2f}, N: {len(data)}")

# Perform Levene's test for homogeneity of variances
levene_stat, levene_p_value = stats.levene(data1, data2, data3, data4)
print(f"\nLevene's Test for Homogeneity of Variance: F-statistic = {levene_stat:.4f}, p-value = {levene_p_value:.4f}")

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(data1, data2, data3, data4)

# Output ANOVA results
print(f"\nOne-way ANOVA Results:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

# Conclusion based on p-value
alpha = 0.09
if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. There is a significant difference between the means.")
else:
    print("Conclusion: Fail to reject the null hypothesis. The means are not significantly different.")

# Interpret Levene's test
if levene_p_value < alpha:
    print("Levene's Test: The assumption of homogeneity of variances is violated.")
else:
    print("Levene's Test: The assumption of homogeneity of variances is met.")

"""Kruskal-Wallis Test non-parametric test"""
import pandas as pd
from scipy import stats
import numpy as np

# Load data from CSV files
data1 = pd.read_csv('all_objective_results_5_500_years.csv')
data2 = pd.read_csv('all_objective_results_10_500_years.csv')
data3 = pd.read_csv('all_objective_results_15_500_years.csv')
data4 = pd.read_csv('all_objective_results_20_500_years.csv')

# Ensure all data is a 1D array
data1 = data1.values.flatten()
data2 = data2.values.flatten()
data3 = data3.values.flatten()
data4 = data4.values.flatten()

# Print shapes of the data to check if they are 1D
print(f"Data1 shape: {data1.shape}")
print(f"Data2 shape: {data2.shape}")
print(f"Data3 shape: {data3.shape}")
print(f"Data4 shape: {data4.shape}")

kruskal_stat, p_value = stats.kruskal(data1, data2, data3, data4)

# Output results
print(f"\nKruskal-Wallis H-statistic: {kruskal_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Conclusion
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference between the groups.")
    
"""Post-Hoc Dunn's Test"""

import pandas as pd
import scikit_posthocs as sp

# Load data from CSV files
data1 = pd.read_csv('all_objective_results_5_500_years.csv').iloc[:, 0].values.flatten()
data2 = pd.read_csv('all_objective_results_10_500_years.csv').iloc[:, 0].values.flatten()
data3 = pd.read_csv('all_objective_results_15_500_years.csv').iloc[:, 0].values.flatten()
data4 = pd.read_csv('all_objective_results_20_500_years.csv').iloc[:, 0].values.flatten()


# Print shapes of the data to confirm they are 1D
print(f"Data1 shape: {data1.shape}")  # should be (n,)
print(f"Data2 shape: {data2.shape}")  # should be (n,)
print(f"Data3 shape: {data3.shape}")  # should be (n,)
print(f"Data4 shape: {data4.shape}")  # should be (n,)

# Combine data into a single DataFrame and add group labels
data_combined = pd.concat([pd.Series(data1), pd.Series(data2), pd.Series(data3), pd.Series(data4)], axis=0).reset_index(drop=True)
groups = ['Group 1'] * len(data1) + ['Group 2'] * len(data2) + ['Group 3'] * len(data3) + ['Group 4'] * len(data4)

# Create a DataFrame with data and corresponding group labels
df = pd.DataFrame({'Values': data_combined, 'Group': groups})

# Perform Dunn's test
dunn_results = sp.posthoc_dunn(df, val_col='Values', group_col='Group', p_adjust='holm')

# Output results
print("Dunn's Test Results:")
print(dunn_results)

"""Plots: Initial conditions are fixed"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t
import re
from IPython.display import display

# Set directory path
directory_path = r'C:\Users\Alejandro Yaber\Desktop\Simulation\October -2024-Optimization\Comparison inter-scenario'

# Define the IRR (10%) as the discount rate
IRR = 0.025

def calculate_equivalent_annual_annuity(mean_value, years, irr):
    """
    Calculate the Equivalent Annual Annuity (EAA) from a present value.
    """
    # First calculate the annuity factor
    if irr == 0:
        annuity_factor = years
    else:
        annuity_factor = (1 - (1 + irr)**-years) / irr
        
    # Calculate EAA
    eaa = mean_value / annuity_factor
    
    return eaa

def process_scenario_data(data, years, irr):
    """
    Process scenario data to calculate EAA and confidence intervals
    """
    # Calculate mean PV
    mean_pv = data['Objective Value'].mean()
    
    # Calculate EAA
    eaa_value = calculate_equivalent_annual_annuity(mean_pv, years, irr)
    
    # Calculate confidence interval for the mean
    confidence = 0.95
    n = len(data['Objective Value'])
    std_err = sem(data['Objective Value'])
    ci = std_err * t.ppf((1 + confidence) / 2, n - 1)
    
    # Convert confidence interval to EAA terms
    ci_eaa = calculate_equivalent_annual_annuity(ci, years, irr)
    
    return eaa_value, ci_eaa

# Initialize a dictionary to store grouped data by scenario
grouped_data_by_scenario = {}

# Loop through all files in the directory
for file in os.listdir(directory_path):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_path, file)
        
        # Extract the planning horizon and scenario code from the file name
        match = re.search(r'all_objective_results_(\d+)_\d+_(\d{7})_years\.csv', file)
        if match:
            years = int(match.group(1))
            scenario = match.group(2)
            horizon = f"{years} years"
            
            # Ensure scenario exists in grouped_data_by_scenario
            if scenario not in grouped_data_by_scenario:
                grouped_data_by_scenario[scenario] = {
                    '5 years': [],
                    '10 years': [],
                    '15 years': [],
                    '20 years': []
                }
            
            # Read CSV file
            data = pd.read_csv(file_path)
            
            # Calculate EAA and confidence interval
            eaa_value, ci_eaa = process_scenario_data(data, years, IRR)
            
            # Store results
            grouped_data_by_scenario[scenario][horizon].append(
                (f"{file} (Scenario {scenario})", eaa_value, ci_eaa)
            )
        else:
            print(f"File {file} doesn't match pattern, skipping.")

# Create visualization functions
def plot_scenario_comparison(grouped_data_by_scenario):
    """
    Create a grouped bar plot comparing scenarios and horizons
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Prepare data for plotting
    scenario_labels = list(grouped_data_by_scenario.keys())
    x = np.arange(len(scenario_labels))
    bar_width = 0.2

    # Prepare data for the table
    table_data = []

    # Color palette for different planning horizons
    colors = plt.cm.Set3(np.linspace(0, 1, 4))

    # Add bars to the plot
    for i, scenario in enumerate(scenario_labels):
        horizons = grouped_data_by_scenario[scenario]
        horizon_labels = list(horizons.keys())
        
        # Get data for each horizon in the current scenario
        eaas = []
        conf_intervals = []
        labels = []
        
        for idx, horizon in enumerate(horizon_labels):
            if horizons[horizon]:
                horizon_files, horizon_eaas, horizon_cis = zip(*horizons[horizon])
                eaas.append(np.mean(horizon_eaas))
                conf_intervals.append(np.mean(horizon_cis))
                labels.append(horizon)

                # Append data to table
                for horizon_file, horizon_eaa, horizon_ci in zip(horizon_files, horizon_eaas, horizon_cis):
                    if [horizon_file, f"{horizon_eaa:.2e}", f"{horizon_ci:.2e}"] not in table_data:
                        table_data.append([horizon_file, f"{horizon_eaa:.2e}", f"{horizon_ci:.2e}"])

        positions = x[i] + np.arange(len(eaas)) * bar_width
        
        bars = ax.bar(positions, eaas, bar_width, yerr=conf_intervals, 
                     capsize=5, label=horizon_labels[0:len(eaas)],
                     color=colors[:len(eaas)], alpha=0.7)
        
        # Annotate bars with EAA values
        for bar, eaa in zip(bars, eaas):
            height = bar.get_height() 
            ax.annotate(f'{eaa:.2e}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0,8),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10,
                        rotation=0)
        
        # Annotate bars with planning horizon labels
        for bar, label in zip(bars, labels):
            height = bar.get_height() + bar.get_height() * 0.16  # Shift the label slightly above the bar
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10,
                        rotation=0)

    # Customize plot
    ax.set_xticks(x + bar_width * (len(horizon_labels) - 1) / 2)
    ax.set_xticklabels(scenario_labels)
    plt.xticks(rotation=0)
    plt.ylabel('Equivalent Annuity (EA)')
    plt.xlabel('Initial conditions scenarios')
    plt.title('Comparison of Equivalent Annuity by Scenario and Planning Horizon')
    plt.legend(title="Planning Horizons")

    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, table_data

# Create and display visualizations
fig, table_data = plot_scenario_comparison(grouped_data_by_scenario)

# Display the plot
plt.show()

# Create and display the table
table_columns = ['File Name (Scenario)', 'Equivalent Annuity', 'Confidence Interval']
df_table = pd.DataFrame(table_data, columns=table_columns)
display(df_table)

# Additional summary statistics
print("\nSummary Statistics by Scenario and Planning Horizon:")
for scenario in grouped_data_by_scenario:
    print(f"\nScenario {scenario}:")
    for horizon, entries in grouped_data_by_scenario[scenario].items():
        if entries:
            eaas = [entry[1] for entry in entries]
            print(f"  {horizon}:")
            print(f"    Mean EAA: {np.mean(eaas):.2e}")
            print(f"    Std Dev: {np.std(eaas):.2e}")
            print(f"    Min: {np.min(eaas):.2e}")
            print(f"    Max: {np.max(eaas):.2e}")

"""Plots: Planning horizons are fixed"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t
import re
from IPython.display import display

# Set directory path
directory_path = r'C:\Users\Alejandro Yaber\Desktop\Simulation\October -2024-Optimization\Comparison inter-scenario'

# Define the IRR (10%) as the discount rate
IRR = 0.025

def calculate_equivalent_annual_annuity(mean_value, years, irr):
    """
    Calculate the Equivalent Annual Annuity (EAA) from a present value.
    """
    # First calculate the annuity factor
    if irr == 0:
        annuity_factor = years
    else:
        annuity_factor = (1 - (1 + irr)**-years) / irr
        
    # Calculate EAA
    eaa = mean_value / annuity_factor
    
    return eaa

def process_scenario_data(data, years, irr):
    """
    Process scenario data to calculate EAA and confidence intervals
    """
    # Calculate mean PV
    mean_pv = data['Objective Value'].mean()
    
    # Calculate EAA
    eaa_value = calculate_equivalent_annual_annuity(mean_pv, years, irr)
    
    # Calculate confidence interval for the mean
    confidence = 0.95
    n = len(data['Objective Value'])
    std_err = sem(data['Objective Value'])
    ci = std_err * t.ppf((1 + confidence) / 2, n - 1)
    
    # Convert confidence interval to EAA terms
    ci_eaa = calculate_equivalent_annual_annuity(ci, years, irr)
    
    return eaa_value, ci_eaa

# Initialize a dictionary to store grouped data by scenario
grouped_data_by_scenario = {}

# Loop through all files in the directory
for file in os.listdir(directory_path):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_path, file)
        
        # Extract the planning horizon and scenario code from the file name
        match = re.search(r'all_objective_results_(\d+)_\d+_(\d{7})_years\.csv', file)
        if match:
            years = int(match.group(1))
            scenario = match.group(2)
            horizon = f"{years} years"
            
            # Ensure scenario exists in grouped_data_by_scenario
            if scenario not in grouped_data_by_scenario:
                grouped_data_by_scenario[scenario] = {
                    '5 years': [],
                    '10 years': [],
                    '15 years': [],
                    '20 years': []
                }
            
            # Read CSV file
            data = pd.read_csv(file_path)
            
            # Calculate EAA and confidence interval
            eaa_value, ci_eaa = process_scenario_data(data, years, IRR)
            
            # Store results
            grouped_data_by_scenario[scenario][horizon].append(
                (f"{file} (Scenario {scenario})", eaa_value, ci_eaa)
            )
        else:
            print(f"File {file} doesn't match pattern, skipping.")

# Create visualization functions
def plot_scenario_comparison(grouped_data_by_scenario):
    """
    Create a grouped bar plot comparing scenarios for each planning horizon
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Prepare data for plotting
    horizon_labels = ['5 years', '10 years', '15 years', '20 years']
    x = np.arange(len(horizon_labels))  # Position of bars for each horizon
    bar_width = 0.2  # Width of the bars

    # Prepare data for the table
    table_data = []

    # Color palette for different scenarios
    colors = plt.cm.Set3(np.linspace(0, 1, len(grouped_data_by_scenario)))

    # Add bars to the plot
    for i, horizon in enumerate(horizon_labels):
        # Collect data for all scenarios for this horizon
        eaas = []
        conf_intervals = []
        labels = []

        # Get data for each scenario in the current horizon
        for scenario, horizons in grouped_data_by_scenario.items():
            if horizons[horizon]:
                horizon_files, horizon_eaas, horizon_cis = zip(*horizons[horizon])
                eaas.append(np.mean(horizon_eaas))
                conf_intervals.append(np.mean(horizon_cis))
                labels.append(scenario)

                # Append data to table
                for horizon_file, horizon_eaa, horizon_ci in zip(horizon_files, horizon_eaas, horizon_cis):
                    if [horizon_file, f"{horizon_eaa:.2e}", f"{horizon_ci:.2e}"] not in table_data:
                        table_data.append([horizon_file, f"{horizon_eaa:.2e}", f"{horizon_ci:.2e}"])

        positions = x[i] + np.arange(len(eaas)) * bar_width
        
        # Create bars for each scenario for this horizon
        bars = ax.bar(positions, eaas, bar_width, yerr=conf_intervals, 
                     capsize=5, label=f"{horizon} EAA", color=colors[:len(eaas)], alpha=0.7)

        # Annotate bars with values
        for bar, eaa, label in zip(bars, eaas, labels):
            height = bar.get_height()
            ax.annotate(f'{eaa:.2e}\n({label})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10,
                        rotation=0)

    # Customize plot
    ax.set_xticks(x + bar_width * (len(grouped_data_by_scenario) - 1) / 2)
    ax.set_xticklabels(horizon_labels)
    plt.xticks(rotation=0)
    plt.xlabel('Planning horizons')
    plt.ylabel('Equivalent Annuity (EA)')
    plt.title('Comparison of Equivalent Annuity by Scenario and Planning Horizon')
    plt.legend(title="Scenarios", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, table_data

# Create and display visualizations
fig, table_data = plot_scenario_comparison(grouped_data_by_scenario)

# Display the plot
plt.show()

# Create and display the table
table_columns = ['File Name (Scenario)', 'Equivalent Annual Annuity', 'Confidence Interval']
df_table = pd.DataFrame(table_data, columns=table_columns)
display(df_table)

# Additional summary statistics
print("\nSummary Statistics by Scenario and Planning Horizon:")
for scenario in grouped_data_by_scenario:
    print(f"\nScenario {scenario}:")
    for horizon, entries in grouped_data_by_scenario[scenario].items():
        if entries:
            eaas = [entry[1] for entry in entries]
            print(f"  {horizon}:")
            print(f"    Mean EAA: {np.mean(eaas):.2e}")
            print(f"    Std Dev: {np.std(eaas):.2e}")
            print(f"    Min: {np.min(eaas):.2e}")
            print(f"    Max: {np.max(eaas):.2e}")

""""Box plots: Optimization vs Benchmark"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

"""15 years"""

# Function to load and prepare data from CSV files
def load_and_prepare_data(opt_file, bench_file, label):
    optimization_result = pd.read_csv(opt_file)
    benchmark_result = pd.read_csv(bench_file)

    optimization_values = optimization_result['Objective Value']
    benchmark_values = benchmark_result['Objective Value']

    data = pd.DataFrame({
        'Model': ['Optimization Model'] * len(optimization_values) + ['Benchmark Simulation'] * len(benchmark_values),
        'Objective Value': pd.concat([optimization_values, benchmark_values], ignore_index=True),
        'Comparison': label
    })
    return data

# Function to calculate mean and confidence interval
def calculate_mean_and_ci(data):
    """
    Calculate the mean and 95% confidence interval for the data.
    """
    mean_value = np.mean(data)  # Mean of the values
    sem = stats.sem(data)  # Standard error of the mean
    ci = stats.t.interval(0.95, len(data)-1, loc=mean_value, scale=sem)  # 95% confidence interval
    return ci, mean_value

# Load and process data from CSV files
files = [
    ('all_objective_results_15_500_0000000_years.csv', 'all_objective_results_15_500_0000000_years_b.csv', '0000000-15 years'),
    ('all_objective_results_15_500_0001001_years.csv', 'all_objective_results_15_500_0001001_years_b.csv', '0001001-15 years'),
    ('all_objective_results_15_500_0110010_years.csv', 'all_objective_results_15_500_0110010_years_b.csv', '0110010-15 years'),
    ('all_objective_results_15_500_1111111_years.csv', 'all_objective_results_15_500_1111111_years_b.csv', '1111111-15 years')
]

combined_data = pd.concat([load_and_prepare_data(opt, bench, label) for opt, bench, label in files], ignore_index=True)

# Calculate mean and confidence intervals for all comparisons and models
ci_data = []
for comp in combined_data['Comparison'].unique():
    for model in combined_data['Model'].unique():
        subset = combined_data[(combined_data['Comparison'] == comp) & (combined_data['Model'] == model)]['Objective Value']
        ci, mean_value = calculate_mean_and_ci(subset)
        
        # Append results to dataframe for plotting
        ci_data.append({'Comparison': comp, 'Model': model, 'Lower CI': ci[0], 'Upper CI': ci[1], 'Mean': mean_value})
        
        # Print the mean and confidence interval
        print(f"Comparison: {comp}, Model: {model}")
        print(f"  Average Objective Value: {mean_value:.2f}")
        print(f"  95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]\n")

ci_df = pd.DataFrame(ci_data)

# Plot box plots with overlaid confidence intervals
plt.figure(figsize=(14, 8))
palette = 'Set2'

# Create the box plot
sns.boxplot(
    x='Comparison',
    y='Objective Value',
    hue='Model',
    data=combined_data,
    palette=palette,
    showfliers=False  # Optional: Hide outliers to avoid clutter
)

# Overlay confidence intervals
for idx, row in ci_df.iterrows():
    # Get x position for the current Comparison and Model
    comparison_idx = sorted(ci_df['Comparison'].unique()).index(row['Comparison'])
    model_offset = -0.2 if row['Model'] == 'Optimization Model' else 0.2  # Adjust for hue categories
    x_pos = comparison_idx + model_offset
    
    # Plot error bar for the confidence interval
    plt.errorbar(
        x=x_pos, 
        y=row['Mean'], 
        yerr=[[row['Mean'] - row['Lower CI']], [row['Upper CI'] - row['Mean']]], 
        fmt='o', 
        capsize=5, 
        color='black'
    )

# Customize plot
plt.title('Box Plots with Confidence Intervals for Mean Objective Values')
plt.xlabel('Scenario (Comparison)')
plt.ylabel('Objective Value')
plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
