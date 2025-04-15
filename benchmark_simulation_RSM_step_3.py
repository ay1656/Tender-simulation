"""I am writing a simulation code using SimPy environment for tender scheduling for vaccine procurement. It must have a main function that calls upon the following logic:
Initial condition must be read for a group of antigens, these antigens are Polio, Diphtheria, Tetanus, Pertussis, Mumps, Measles, and Rubella. The initial conditions come from year 0 and should be read into the program as a data frame. These conditions are, for every antigen:  population of unvaccinated individuals in the market, the number of doses used of the antigen to immunize the population during the upcoming period that was committed in a previous tender year, the number of doses delivered in that year, the time-space to be covered by the supply of a resulting scheduled tender, the number of doses required per antigen to reach immunity for one individual, the current inventory of vaccines that cover the antigen,  a reorder point for every antigen that will be a product of the doses required to reach immunity multiplied by the expected birth cohort of the current year, the reservation price per vaccine, the cost per unvaccinated individual, the holding cost per dose of an antigen per period, a fixed set-up cost when a tender is scheduled. A production lead time that will be fixed for all vaccines, is 2 years. Finally, the provider capability for every vaccine that covers an antigen will take the integer value of the expected value of normal distribution with a mean and a standard deviation. There will be 3 providers for Polio, 4 providers for Diphtheria, tetanus and pertussis (one of the providers is common amongst all providers and 3 providers for Measles, Mumps and Rubella.
The following step that the main function must call functions that perform the following events related to demand every year, from year one to the final year of the simulation: Calculating the current inventory for every antigen, which will be equal to, the inventory from the previous period plus the sum of all partial deliveries to be received in the year minus the number of vaccines used. The following function should be the inventory of unvaccinated individuals per antigen each year, which will be the number of unvaccinated individuals from the previous year, plus the immunized individuals each year plus the birth cohort of the year which will be the result of a random Poisson variable with a lambda value of average births per year. The following function that the main function must call is the calculate demand function, which will be for every antigen, which will result from multiplying the number of unvaccinated individuals in the previous function multiplied by the number of doses required to reach immunity per antigen. It must later calculate the metrics unvaccinated cost per antigen, which will be the multiplication of the unvaccinated individuals per antigen in each year, multiplied by the fixed cost per unvaccinated individual that was in the initial conditions.
The following step is that the main function must call functions that perform tender scheduling, it must be performed individually per antigen. A tender will be scheduled if and only if the current inventory for an antigen is less or equal than its reorder point, there are no previous tender scheduled in the previous time periods extending back from the current year minus the time-space parameter plus one year and that the capacity of a provider is greater or equal to the current demand multiplied by the time-space parameter. A provider must be chosen and a variable must identify that this provider is selected
Once a tender is scheduled, doses are committed, the committed doses will be greater or equal to the demand multiplied by the time-space parameter. 
The next step that the main function will call is if a tender is scheduled and the committed doses are created the tender cost will be calculated as equal to the fixed set-up cost parameter. The partial deliveries will be calculated as the total doses committed divided by the time-space parameter which will be distributed equally over the time-space and will become the partial delivery used in the calculate demand function. The capabilities of the provider selected must be reduced by the committed doses, until the production lead time has passed.
 """
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt
from scipy.stats import poisson
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from itertools import combinations
from multiprocessing import Process
import os
import csv
import sys
import warnings
warnings.filterwarnings('ignore')



# Set a fixed seed for reproducibility 
seed = 1
np.random.seed(seed)

def read_initial_conditions(initial_conditions_csv):
    # Read initial conditions from a CSV file into a DataFrame
    initial_conditions = pd.read_csv(initial_conditions_csv, index_col=0)
    return initial_conditions
"""Reading Initial Conditions:
The read_initial_conditions function reads initial conditions from a CSV file into a Pandas DataFrame."""
    

#"""Initialization (__init__): Initializes the simulation with essential parameters like the simulation environment (env), initial conditions, the number of years to simulate (num_years), etc."""
class VaccineProcurementSimulation: 
    def __init__(self, env, initial_conditions, num_years):
        self.env = env
        self.initial_conditions = initial_conditions
        self.antigens = initial_conditions.index
        self.providers = self.initialize_providers()
        self.num_years=num_years
        self.tender_records = {antigen: {} for antigen in self.antigens}
        
#"""initialize_providers: Sets up providers' capabilities based on a normal distribution for different antigens."""
    def initialize_providers(self):
        # Initialize providers with capabilities based on normal distribution
        providers = {}
        for antigen in self.antigens:
            num_providers = self.initial_conditions.loc[antigen, 'num_providers']
            mean_capability = self.initial_conditions.loc[antigen, 'provider_mean_capability']
            std_dev_capability = self.initial_conditions.loc[antigen, 'provider_std_dev_capability']
            provider_prefix = 'Provider' + antigen.capitalize()  # This assumes the prefix is the same as the antigen name
            
            providers[antigen] = {
                f'{provider_prefix}{i+1}': int(np.random.normal(mean_capability, std_dev_capability))
                for i in range(0, num_providers)
            }
            # Add provider_prefix to the initial conditions DataFrame
            self.initial_conditions.loc[antigen, 'provider_prefix'] = provider_prefix

        return providers
    
    def update_provider_capabilities_after_lead_time(self, antigen, provider, committed_doses, production_lead_time):
        # Wait for the production lead time
        yield self.env.timeout(production_lead_time)

        # Replenish the provider's capabilities by adding the committed doses back
        self.providers[antigen][provider] += committed_doses
        #print("Replenished Providers Dictionary for Antigen:", antigen)
        #print(self.providers[antigen])
        
    
#"""process_demand_with_tender_scheduling: Simulates the procurement process for each antigen, checking inventory, calculating demand, and determining if a tender should be scheduled based on certain conditions."""
    def process_demand_with_tender_scheduling(self, production_lead_time, antigen, num_years):
        committed_doses_dict={}
        for current_year in range (0, num_years):
            # Calculate current inventory for every antigen
            # Calculate inventory of unvaccinated individuals per antigen each year
            previous_unvaccinated = self.initial_conditions.loc[antigen, 'unvaccinated_individuals']
            birth_cohort = np.random.poisson(self.initial_conditions.loc[antigen, 'average_births_per_year'])
            vaccines_used = self.initial_conditions.loc[antigen, 'vaccines_used']
            unvaccinated_individuals = previous_unvaccinated + birth_cohort + vaccines_used
            # Calculate demand for every antigen           
            doses_required = self.initial_conditions.loc[antigen, 'doses_required']
            demand = unvaccinated_individuals * doses_required 
            if current_year==0:
                previous_inventory = self.initial_conditions.loc[antigen, 'current_inventory']
                partial_deliveries = self.initial_conditions.loc[antigen, 'partial_deliveries']
                current_inventory = previous_inventory + partial_deliveries - vaccines_used
                # Calculate unvaccinated cost per antigen       
                unvaccinated_cost = unvaccinated_individuals * self.initial_conditions.loc[antigen, 'cost_per_unvaccinated_individual']
                holding_cost = current_inventory * self.initial_conditions.loc[antigen, 'holding_cost']
                #Update relevant variables
                self.initial_conditions.loc[antigen, f'current_inventory_year_{current_year}'] = current_inventory
                self.initial_conditions.loc[antigen, 'unvaccinated_individuals'] = unvaccinated_individuals
                self.initial_conditions.loc[antigen, 'demand'] = demand
                self.initial_conditions.loc[antigen, 'unvaccinated_cost'] = unvaccinated_cost
                self.initial_conditions.loc[antigen, f'holding_cost_year_{current_year}'] = holding_cost
                time_space_demand = self.initial_conditions.loc[antigen, 'demand']#*self.initial_conditions.loc[antigen, 'time_space']
                committed_doses = int(time_space_demand)
                self.initial_conditions.loc[antigen, f'needed_doses_year_{current_year}']=committed_doses*self.initial_conditions.loc[antigen, 'time_space']
                                        
                if self.is_tender_scheduled(antigen, current_year, committed_doses, current_inventory):
                    # Calculate purchase cost for each antigen               
                    purchase_cost = committed_doses * self.initial_conditions.loc[antigen, 'purchase_cost']
                    provider = self.choose_provider(antigen, committed_doses)                     
                    tender_cost = self.initial_conditions.loc[antigen, 'fixed_setup_cost']
                    # Calculate partial deliveries
                    partial_deliveries = committed_doses / self.initial_conditions.loc[antigen, 'time_space']
                    doses_used= int(partial_deliveries)
                    current_inventory += int(partial_deliveries-doses_used)
                    unvaccinated_individuals-=doses_used
                    # Update relevant variables
                    self.initial_conditions.loc[antigen, 'tender_cost'] = tender_cost
                    self.initial_conditions.loc[antigen, f'partial_deliveries_year_{current_year}'] = partial_deliveries
                    self.initial_conditions.loc[antigen, f'purchase_cost_year_{current_year}'] = purchase_cost
                    self.initial_conditions.loc[antigen, f'current_inventory_year_{current_year}']= int(current_inventory)
                    self.initial_conditions.loc[antigen, 'vaccines_used']=doses_used
                    self.initial_conditions.loc[antigen, 'unvaccinated_individuals']=unvaccinated_individuals
                    self.initial_conditions.loc[antigen, f'committed_doses_year_{current_year}']=committed_doses*self.initial_conditions.loc[antigen, 'time_space']
                    
                    if provider:
                        self.update_provider_capabilities(antigen, provider, committed_doses)
                        
                        # Schedule the update of provider capabilities after the lead time
                        self.env.process(self.update_provider_capabilities_after_lead_time(antigen, provider, committed_doses, production_lead_time))
                        
                    # Production lead time - wait before updating provider capabilities
                    yield self.env.timeout(production_lead_time)
                    
                    # Store committed doses for future distribution
                    committed_doses_dict[antigen] = committed_doses_dict.get(antigen, 0) + committed_doses
                
                else:
                    self.initial_conditions.loc[antigen, f'committed_doses_year_{current_year}']=0
                    self.initial_conditions.loc[antigen, f'partial_deliveries_year_{current_year}']=self.initial_conditions.loc[antigen, 'partial_deliveries']
                    self.initial_conditions.loc[antigen, f'vaccines_used_year_{current_year}']=self.initial_conditions.loc[antigen, 'vaccines_used']
                    self.initial_conditions.loc[antigen, f'current_inventory_year_{current_year}']= self.initial_conditions.loc[antigen, 'current_inventory']-(self.initial_conditions.loc[antigen, 'partial_deliveries']+self.initial_conditions.loc[antigen, 'vaccines_used'])
                    
                   

            else:
                self.initial_conditions.loc[antigen, 'current_inventory'] = current_inventory
                self.initial_conditions.loc[antigen, 'unvaccinated_individuals'] = unvaccinated_individuals
                self.initial_conditions.loc[antigen, 'demand'] = demand
                self.initial_conditions.loc[antigen, 'unvaccinated_cost'] = unvaccinated_cost
                self.initial_conditions.loc[antigen, f'holding_cost_year_{current_year}'] = holding_cost
                time_space_demand = self.initial_conditions.loc[antigen, 'demand']#*self.initial_conditions.loc[antigen, 'time_space']
                committed_doses = int(time_space_demand)
                self.initial_conditions.loc[antigen, f'needed_doses_{current_year}']=committed_doses*self.initial_conditions.loc[antigen, 'time_space']
                
                                        
                if self.is_tender_scheduled(antigen, current_year, committed_doses, current_inventory):
                    # Calculate purchase cost for each antigen               
                    purchase_cost = committed_doses * self.initial_conditions.loc[antigen, 'purchase_cost']
                    provider = self.choose_provider(antigen, committed_doses)                     
                    tender_cost = self.initial_conditions.loc[antigen, 'fixed_setup_cost']
                    # Calculate partial deliveries
                    partial_deliveries = committed_doses / self.initial_conditions.loc[antigen, 'time_space']
                    doses_used=int(partial_deliveries)
                    current_inventory += int(partial_deliveries-doses_used)
                    unvaccinated_individuals-=doses_used
                    # Update relevant variables
                    self.initial_conditions.loc[antigen, 'tender_cost'] = tender_cost
                    self.initial_conditions.loc[antigen, f'partial_deliveries_year_{current_year}'] = partial_deliveries
                    self.initial_conditions.loc[antigen, f'purchase_cost_year_{current_year}'] = purchase_cost
                    self.initial_conditions.loc[antigen, f'current_inventory_year_{current_year}']= int(current_inventory)
                    self.initial_conditions.loc[antigen, 'vaccines_used']=doses_used
                    self.initial_conditions.loc[antigen, 'unvaccinated_individuals']=unvaccinated_individuals
                    self.initial_conditions.loc[antigen, f'committed_doses_year_{current_year}']=committed_doses*self.initial_conditions.loc[antigen, 'time_space']
                   
                    if provider:
                        self.update_provider_capabilities(antigen, provider, committed_doses)
                        
                         # Schedule the update of provider capabilities after the lead time
                        self.env.process(self.update_provider_capabilities_after_lead_time(antigen, provider, committed_doses, production_lead_time))
                        
                    # Production lead time - wait before updating provider capabilities
                    yield self.env.timeout(production_lead_time)
                    
                    # Store committed doses for future distribution
                    committed_doses_dict[antigen] = committed_doses_dict.get(antigen, 0) + committed_doses
               
                else:
                    partial_deliveries = committed_doses / self.initial_conditions.loc[antigen, 'time_space']
                    doses_used=int(partial_deliveries)
                    current_inventory += int(partial_deliveries-doses_used)
                    self.initial_conditions.loc[antigen, f'committed_doses_year_{current_year}']=0
                    self.initial_conditions.loc[antigen, f'current_inventory_year_{current_year}']= int(current_inventory)
                    self.initial_conditions.loc[antigen, f'partial_deliveries_year_{current_year}'] = partial_deliveries
                    self.initial_conditions.loc[antigen, f'purchase_cost_year_{current_year}'] = 0
                    
            # Save committed doses as demand for the current year
            self.initial_conditions.loc[antigen, f'demand_year_{current_year}'] = committed_doses    
                        
        
        yield self.env.timeout(1)
                
                
        # Distribute accumulated committed doses across future years
        for antigen, committed_doses in committed_doses_dict.items():
            # Distribute committed doses across future years
            for future_year in range(current_year + 1, self.num_years):
                partial_deliveries_future = committed_doses / (self.initial_conditions.loc[antigen, 'time_space'])
                self.initial_conditions.loc[antigen, f'partial_deliveries_year_{future_year}'] += partial_deliveries_future    

#"""is_tender_scheduled: Checks conditions to decide whether a tender should be scheduled for a specific antigen in a particular year."""
    
    def is_tender_scheduled(self, antigen, current_year, committed_doses, current_inventory):    
        doses_required = self.initial_conditions.loc[antigen, 'doses_required']
        birth_cohort = np.random.poisson(self.initial_conditions.loc[antigen, 'average_births_per_year'])
        # Calculate the reorder point based on doses required per antigen multiplied by the birth cohort
        reorder_point = doses_required * birth_cohort*self.initial_conditions.loc[antigen, 'time_space']*self.initial_conditions.loc[antigen, 'factor']
        time_space = (self.initial_conditions.loc[antigen, 'time_space'])
        time_space=int(time_space)
        provider = self.choose_provider(antigen, committed_doses)
        #print(f"Providers Dictionary: {self.providers}")
        #print(f"Antigen: {antigen}")
        #print(f"Year:{current_year}")
        #print(f"Comitted Doses:{committed_doses}")
        if provider is None or provider not in self.providers.get(antigen, {}):
            return False 
        
        if current_year==0:
            has_previous_tender = self.has_previous_tender(antigen, current_year, time_space)
            inventory_check = self.initial_conditions.loc[antigen, 'current_inventory']  <= reorder_point
            provider_check = self.providers[antigen].get(provider, 0) >= committed_doses
        
        else:
            has_previous_tender = self.has_previous_tender(antigen, current_year, time_space)
            inventory_check = current_inventory <= reorder_point
            provider_check = self.providers[antigen].get(provider, 0) >= committed_doses
        
        conditions_met = inventory_check and provider_check and not has_previous_tender
        
        # Check if the provider_check fails
        if not provider_check:
            # Check combinations of providers
            for i in range(2, len(self.providers[antigen]) + 1):
                for combination in combinations(self.providers[antigen].items(), i):
                    total_capability = sum(capability for _, capability in combination)
                    if total_capability >= committed_doses and all(self.provider_covers_antigen(provider, antigen) for provider, _ in combination):
                        provider_check = True  # Return True when conditions are met for a combination of providers
                    else:
                        provider_check= False

               
        conditions_met=inventory_check and provider_check and not has_previous_tender
        
        if conditions_met:
            self.tender_records[antigen][current_year] = 1
            return True
        else:
            return False
        
    def save_initial_conditions_to_csv(self, results_complete='initial_conditions_output.csv'):
        self.initial_conditions.to_csv(results_complete)      

#"""has_previous_tender: Checks if a tender has been scheduled in the past based on the time space for a particular antigen."""
    def has_previous_tender(self, antigen, current_year, time_space):
        # Define the start and end years to check for previous tenders
        start_year = max(0, current_year )  # Year minus time_space or 0 (whichever is greater)
        end_year = max(0, (current_year + time_space) - 1)  # Year before the current year
        
       # Initialize the total tender count
        total_tender_count = 0
        
        if current_year == 0:
            total_tender_count += self.initial_conditions.loc[antigen, 'tender_scheduled']
        
        elif current_year>0 and current_year<time_space:
            # Determine if there were any tenders scheduled over the time frame from tender_records
            previous_years = range(max(0, current_year - time_space + 1), max(0, current_year))
            total_tender_count = self.initial_conditions.loc[antigen, 'tender_scheduled']+ sum(self.tender_records[antigen].get(prev_year, 0) for prev_year in previous_years)
        
        else:
            previous_years = range(max(0, current_year - time_space + 1), max(0, current_year))
            total_tender_count = + sum(self.tender_records[antigen].get(prev_year, 0) for prev_year in previous_years)
        
        print(self.tender_records)
        return total_tender_count > 0                   
        return sum_previous_tenders > 0 
    
#"""choose_provider: Selects a provider based on their capabilities to fulfill the committed doses for an antigen."""

    def choose_provider(self, antigen, committed_doses):
        antigen_providers = self.providers.get(antigen, {})
      
        for provider, capability in sorted(antigen_providers.items(), key=lambda x: x[1], reverse=True):
            
            if int(capability) >= int(committed_doses):
                if self.provider_covers_antigen(provider, antigen):                    
                    return provider
        #No single provider found, checking combinations
        for i in range(2, len(antigen_providers) + 1):
            
            for combination in combinations(antigen_providers.items(), i):
                total_capability = sum(capability for _, capability in combination)
                if total_capability >= committed_doses and all(
                    self.provider_covers_antigen(provider, antigen) for provider, _ in combination):
                          return [provider for provider, _ in combination][0]  # Return only the provider
       
        return None  
    
#"""provider_covers_antigen: Checks if a specific provider produces vaccines that cover a given antigen."""       
    def provider_covers_antigen(self, provider, antigen):
         # Check if the given provider produces vaccines that cover the specified antigen
            provider_prefix = self.initial_conditions.loc[antigen, "provider_prefix"]
            provider_expected_prefix = provider_prefix
            covers = provider.startswith(provider_expected_prefix)
            
            return covers

#"""update_provider_capabilities: Updates provider capabilities after a tender is scheduled."""
    def update_provider_capabilities(self, antigen, provider, committed_doses):
        # Update provider capabilities after a tender is scheduled
        # Reduce the capabilities of the selected provider
        self.providers[antigen][provider] -= committed_doses
        #print("Updated Providers Dictionary for Antigen:", antigen)
        #print(self.providers[antigen])
           
        
#"""run_one_year_simulation: Runs the simulation for one year."""
  
    def run_planning_horizon_simulation(self, production_lead_time, num_years):
        #Run the simulation       
        for antigen in self.antigens:
            self.env.process(self.process_demand_with_tender_scheduling(production_lead_time,antigen, num_years))
            self.env.run(until=self.env.now+num_years)  # Run 

#"""store_yearly_results: Stores the results of the simulation for each antigen for each year."""
    def store_yearly_results(self, total_objective_value):
        result_vector = []
        for current_year in range(0, self.num_years):
            for antigen in self.antigens:
                #Fetch the necessary values for the current antigen
                current_inventory =  self.initial_conditions.loc[antigen, f'current_inventory_year_{current_year}']
                committed_doses = self.initial_conditions.loc[antigen, f'committed_doses_year_{current_year}']
                partial_deliveries = self.initial_conditions.loc[antigen, f'partial_deliveries_year_{current_year}'] 
                # Check if a tender was scheduled
                tender_scheduled = self.tender_records[antigen].get(current_year, 0) == 1
                # Create a dictionary containing the fetched values
                antigen_result = {
                    'Antigen': antigen,
                    'Year': current_year,
                    'Yearly Inventory': current_inventory,
                    'Committed Doses': committed_doses,
                    'Tender Schedule': tender_scheduled
                }

                # Append the dictionary to the result vector
                result_vector.append(antigen_result)
          #Return the result_vector containing results for each antigen for each year
        return result_vector

#"""calculate_objective_function: Calculates an objective function based on various costs and parameters."""
    
    def calculate_objective_function(self, num_years):
        total_objective_value = 0
        
        for antigen in self.antigens:
            antigen_objective_value=0
            #Iterate over all years
            for current_year in range(0, num_years):
                tender_scheduled = self.tender_records[antigen].get(current_year, 0) == 1
                committed_doses = self.initial_conditions.loc[antigen, f'committed_doses_year_{current_year}']
                if tender_scheduled is True:
                    # Sum of all tender costs across all antigens, providers, and periods
                    tender_cost = self.initial_conditions.loc[antigen, 'tender_cost']
                    antigen_objective_value -= tender_cost
                    
                    # Difference between individual purchase cost of a dose and reservation price per dose
                    reservation_price = self.initial_conditions.loc[antigen, 'reservation_price']
                    cost_difference = (reservation_price) * committed_doses
                    antigen_objective_value+=cost_difference
            #Add the antigen's objective value to the total objective value
            total_objective_value+=antigen_objective_value
        
        #Print the total objective value        
        print(f'Total objective function value: {total_objective_value}')
        
        return total_objective_value


    def plot_tender_records(self):
        # Set up a figure and axis
        plt.figure(figsize=(10, 6))
    
        # Define the range of years to plot (you can adjust the year range as needed)
        max_year = max(max(years.keys()) for years in self.tender_records.values())  # Get max year across all antigens
        years_range = range(0, max_year + 1)  # Define years to plot
    
        for antigen, records in self.tender_records.items():
            # Initialize a list to hold 1s and 0s for each year
            tender_count = [1 if year in records else 0 for year in years_range]       
            # Plot the tender data for each antigen
            plt.plot(years_range, tender_count, label=f'{antigen}')
    
        # Customize the plot
        plt.xlabel('Year')
        plt.ylabel('Tender Scheduled (1 = Yes, 0 = No)')
        plt.title('Tender Records over Time')
        plt.legend()
        plt.grid(True)
        plt.yticks([0, 1], ['No Tender', 'Tender Scheduled'])  # Customize y-axis ticks
        plt.xticks(years_range)  # Set x-ticks for all years
        plt.tight_layout()  # Adjust layout for readability
        plt.show()


#""Main Function (main):
#Reads initial conditions from a CSV file.
#Sets up the simulation environment (simpy.Environment()).
#Initializes the simulation object (VaccineProcurementSimulation) and runs the simulation.
#Overall Flow:
#Read initial conditions.
#Initialize simulation parameters.
#Run the simulation for a specified number of years.
#Store and update results.
#Calculate objective functions and save results to CSV files."""
def main(num_replicates=int(input("Enter number of replicates: "))):
    global objective_df     
    file_paths = ['conditions_0000000._5_5.csv']
    num_years = int(input("Enter the number of years for the simulation: "))
    production_lead_time = int(input("Enter the production lead time: "))

    all_yearly_results = []
    all_objective_values = []

    for replicate in range(num_replicates):
        seed = replicate + 1
        np.random.seed(seed)
        print(f"Running replicate {seed}...")

        final_results_df = pd.DataFrame()

        for file_path in file_paths:
            initial_conditions = read_initial_conditions(file_path)
            env = simpy.Environment()
            simulation = VaccineProcurementSimulation(env, initial_conditions, num_years)
            
            simulation.run_planning_horizon_simulation(production_lead_time, num_years)
            total_objective_value = simulation.calculate_objective_function(num_years)
            yearly_results = simulation.store_yearly_results(total_objective_value)
            
            all_yearly_results.extend(yearly_results)
            all_objective_values.append(total_objective_value)
            
            final_results_df = pd.DataFrame(all_yearly_results)
            objective_df = pd.DataFrame(all_objective_values, columns=['Objective Value'])

        # Plotting tender records for each replicate
        simulation.plot_tender_records()  # Call the plotting function here
        
        final_results_filename = f'final_results_{num_years}_{seed}_years.csv'
        objective_results_filename = f'all_objective_results_{num_years}_{seed}_years_final.csv'
        final_results_df.to_csv(final_results_filename, index=False)
        objective_df.to_csv(objective_results_filename, index=False)

if __name__ == "__main__":
    main()

def main():   
    
    replicates = int(input("Confirm replicates again: "))
    objective_df['Combination'] = (objective_df.index // replicates) + 1

    # Define factor levels
    factor_a_levels = [1, 2, 3]
    factor_b_levels = [1, 2, 3]

    # Create a mapping of combination to factors
    combinations = [(a, b) for a in factor_a_levels for b in factor_b_levels]
    factor_mapping = {i + 1: combination for i, combination in enumerate(combinations)}

    # Assign factor levels to each combination
    objective_df['FactorA'] = objective_df['Combination'].map(lambda x: factor_mapping[x][0])
    objective_df['FactorB'] = objective_df['Combination'].map(lambda x: factor_mapping[x][1])

    # Group by 'FactorA' and 'FactorB' and compute the mean of 'Objective Value'
    averaged_df = objective_df.groupby(['FactorA', 'FactorB'])['Objective Value'].mean().reset_index()

    # Add squared terms for FactorA and FactorB
    averaged_df['FactorA_squared'] = averaged_df['FactorA'] ** 2
    averaged_df['FactorB_squared'] = averaged_df['FactorB'] ** 2

    # Assign blocking factors
    num_rows = len(averaged_df)
    block_size = num_rows // 3  # Assuming 3 blocks

    averaged_df['BlockFactor1'] = ['Block1'] * block_size + ['Block2'] * block_size + ['Block3'] * (num_rows - 2 * block_size)
    averaged_df['BlockFactor2'] = ['BlockA'] * block_size + ['BlockB'] * block_size + ['BlockC'] * (num_rows - 2 * block_size)

    # Convert blocking factors to categorical
    averaged_df['BlockFactor1'] = averaged_df['BlockFactor1'].astype('category')
    averaged_df['BlockFactor2'] = averaged_df['BlockFactor2'].astype('category')

    # Verify new columns
    print("Averaged DataFrame with Blocking Factors:")
    print(averaged_df.head())

    print("Columns in the DataFrame:")
    print(averaged_df.columns)

    # Rename columns to avoid spaces
    averaged_df.rename(columns={'Objective Value': 'Objective_Value'}, inplace=True)

    # Step 1: Fit the second-order model including quadratic terms
    second_order_formula = "Objective_Value ~ FactorA + FactorB + I(FactorA ** 2) + I(FactorB ** 2) + C(BlockFactor1) + C(BlockFactor2)"
    second_order_model = ols(second_order_formula, data=averaged_df).fit()
    print("\nSecond-Order Model Summary:")
    print(second_order_model.summary())

    # Step 2: Determine the path of steepest ascent
    # Extract the coefficients for the linear terms
    linear_terms = ['FactorA', 'FactorB']
    coefficients = second_order_model.params[linear_terms]

    # Compute the gradient for the linear terms
    gradient = coefficients.values

    # Normalize the gradient to get the direction vector
    direction_vector = gradient / np.linalg.norm(gradient)
    print("\nDirection vector (path of steepest ascent):", direction_vector)

    # Function to plot the response surface
    def response_surface_plot(model, x1_name, x2_name):
        # Generate a grid of values
        x1 = np.linspace(averaged_df[x1_name].min(), averaged_df[x1_name].max(), 100)
        x2 = np.linspace(averaged_df[x2_name].min(), averaged_df[x2_name].max(), 100)
        x1, x2 = np.meshgrid(x1, x2)

        # Create a DataFrame for predictions
        df = pd.DataFrame({
            x1_name: x1.ravel(),
            x2_name: x2.ravel(),
            'BlockFactor1': ['Block1'] * len(x1.ravel()),  # Use default or average values for blocks
            'BlockFactor2': ['BlockA'] * len(x1.ravel())
        })

        # Add squared terms
        df['FactorA_squared'] = df[x1_name] ** 2
        df['FactorB_squared'] = df[x2_name] ** 2

        # Ensure all required columns are present
        required_columns = [
            x1_name, x2_name, 'FactorA_squared', 'FactorB_squared',
            'BlockFactor1', 'BlockFactor2'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        # Predict and reshape
        y = model.predict(df)
        y = y.values.reshape(x1.shape)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1, x2, y, cmap='viridis', alpha=0.8)
        ax.set_xlabel(x1_name)
        ax.set_ylabel(x2_name)
        ax.set_zlabel('Objective Value')
        plt.title('Response Surface Plot')
        plt.show()

    # Call the function with the model and variable names
    response_surface_plot(second_order_model, 'FactorA', 'FactorB')

if __name__ == "__main__":
    main()