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
from scipy.stats import poisson
from itertools import combinations
import csv

# Set a fixed seed for reproducibility
seed = 42
np.random.seed(seed)

def read_initial_conditions(initial_conditions_csv): """Reading Initial Conditions:
The read_initial_conditions function reads initial conditions from a CSV file into a Pandas DataFrame."""
    # Read initial conditions from a CSV file into a DataFrame
    initial_conditions = pd.read_csv(initial_conditions_csv, index_col=0)
    return initial_conditions


class VaccineProcurementSimulation: """Initialization (__init__): Initializes the simulation with essential parameters like the simulation environment (env), initial conditions, the number of years to simulate (num_years), etc."""
    def __init__(self, env, initial_conditions, num_years):
        self.env = env
        self.initial_conditions = initial_conditions
        self.antigens = initial_conditions.index
        self.providers = self.initialize_providers()
        self.num_years=num_years

    def initialize_providers(self):"""initialize_providers: Sets up providers' capabilities based on a normal distribution for different antigens."""
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
        # Debug statement to print providers dictionary
        print("Providers Dictionary:", providers)
        return providers
    

    def process_demand_with_tender_scheduling(self, production_lead_time):"""process_demand_with_tender_scheduling: Simulates the procurement process for each antigen, checking inventory, calculating demand, and determining if a tender should be scheduled based on certain conditions."""
        for current_year in range(1):
            for  antigen in (self.antigens):
                # Calculate current inventory for every antigen
                previous_inventory = self.initial_conditions.loc[antigen, 'current_inventory']
                partial_deliveries = self.initial_conditions.loc[antigen, 'partial_deliveries']
                vaccines_used = self.initial_conditions.loc[antigen, 'vaccines_used']
                current_inventory = previous_inventory + partial_deliveries - vaccines_used
                # Calculate inventory of unvaccinated individuals per antigen each year
                previous_unvaccinated = self.initial_conditions.loc[antigen, 'unvaccinated_individuals']
                birth_cohort = np.random.poisson(self.initial_conditions.loc[antigen, 'average_births_per_year'])
                unvaccinated_individuals = previous_unvaccinated + birth_cohort + vaccines_used
                # Calculate demand for every antigen
                doses_required = self.initial_conditions.loc[antigen, 'doses_required']
                demand = unvaccinated_individuals * doses_required            # Calculate unvaccinated cost per antigen       
                unvaccinated_cost = unvaccinated_individuals * self.initial_conditions.loc[antigen, 'cost_per_unvaccinated_individual']
                holding_cost = current_inventory * self.initial_conditions.loc[antigen, 'holding_cost']
                # Update relevant variables
                self.initial_conditions.loc[antigen, 'current_inventory'] = current_inventory
                self.initial_conditions.loc[antigen, 'unvaccinated_individuals'] = unvaccinated_individuals
                self.initial_conditions.loc[antigen, 'demand'] = demand*self.initial_conditions.loc[antigen, 'time_space']
                self.initial_conditions.loc[antigen, 'unvaccinated_cost'] = unvaccinated_cost
                self.initial_conditions.loc[antigen, f'holding_cost_year'] = holding_cost
            
                time_space_demand = self.initial_conditions.loc[antigen, 'demand']
                committed_doses = time_space_demand
                print(f"Antigen: {antigen}, Committed Doses: {committed_doses}")  # Display committed doses
            
                             
                #time_space_demand = [self.initial_conditions.loc[antigen, 'demand'] 
                 #                    for i in range(current_year, current_year - self.initial_conditions.loc[antigen, 'time_space'], -1)]
                #committed_doses = sum(time_space_demand)
                print(f"Antigen: {antigen}, Committed Doses: {committed_doses}")  # Display committed doses  
                if self.is_tender_scheduled(antigen, current_year, committed_doses):
                    # Calculate purchase cost for each antigen
                    purchase_cost = committed_doses * self.initial_conditions.loc[antigen, 'purchase_cost']
                    provider = self.choose_provider(antigen, committed_doses)                     
                    tender_cost = self.initial_conditions.loc[antigen, 'fixed_setup_cost']
                    # Calculate partial deliveries
                    partial_deliveries = committed_doses / self.initial_conditions.loc[antigen, 'time_space']
                    # Update relevant variables
                    self.initial_conditions.loc[antigen, 'tender_cost'] = tender_cost
                    self.initial_conditions.loc[antigen, 'partial_deliveries'] = partial_deliveries
                    self.initial_conditions.loc[antigen, f'purchase_cost_year_{current_year}'] = purchase_cost
                    # Production lead time - wait before updating provider capabilities
                    yield self.env.timeout(production_lead_time)
                    # Update provider capabilities
                    self.update_provider_capabilities(antigen, provider, committed_doses)
            
            yield self.env.timeout(1)
    
    def is_tender_scheduled(self, antigen, year, committed_doses):"""is_tender_scheduled: Checks conditions to decide whether a tender should be scheduled for a specific antigen in a particular year."""
        doses_required = self.initial_conditions.loc[antigen, 'doses_required']
        birth_cohort = np.random.poisson(self.initial_conditions.loc[antigen, 'average_births_per_year'])
        # Calculate the reorder point based on doses required per antigen multiplied by the birth cohort
        reorder_point = doses_required * birth_cohort
        time_space = (self.initial_conditions.loc[antigen, 'time_space'])
        time_space=int(time_space)
        provider = self.choose_provider(antigen, committed_doses)
        # Check values before accessing self.providers[antigen][provider]
        print(f"Antigen: {antigen}, Provider: {provider}") 
        # Check conditions for tender scheduling
        conditions_met = ((self.initial_conditions.loc[antigen, 'current_inventory'] <= reorder_point) & (self.providers[antigen][provider] >= committed_doses) & (not self.has_previous_tender(antigen, year-1, time_space)))
        if conditions_met.all():
            print(f"Tender for {antigen} is scheduled in year {year} due to met conditions.")
        return conditions_met.all()   

    def has_previous_tender(self, antigen, year, time_space):"""has_previous_tender: Checks if a tender has been scheduled in the past based on the time space for a particular antigen."""
        tender_schedule = self.initial_conditions.loc[antigen, 'tender_scheduled']
        for i in range(1, min(year,time_space) + 1):
            index_to_check=year-i
            if index_to_check >= 0:
                print(f"Checking index: {index_to_check}, Value: {tender_schedule[index_to_check]}") 
                if tender_schedule[index_to_check] == 1:
                    return True
    
        return False

    def choose_provider(self, antigen, committed_doses):"""choose_provider: Selects a provider based on their capabilities to fulfill the committed doses for an antigen."""
        antigen_providers = self.providers.get(antigen, {})
        print(f"Antigen: {antigen}, Committed Doses: {committed_doses}")

        for provider, capability in sorted(antigen_providers.items(), key=lambda x: x[1], reverse=True):
            print(f"Provider: {provider}, Capability: {capability}")
            if int(capability) >= int(committed_doses):
                if self.provider_covers_antigen(provider, antigen):
                    print(f"Chosen Provider: {provider}")
                    return provider
                #else:
                 #   print(f"Provider {provider} does not cover {antigen}")
            #else:
             #   print(f"Provider {provider} does not meet committed doses")

        # Print available providers and their capabilities for debugging
        #print("Available Providers and Capabilities:")
        #for provider, capability in antigen_providers.items():
        #    print(f"{provider}: {capability}")

        #print(f"No suitable provider found for {antigen} and {committed_doses}")
        #return None

        #No single provider found, checking combinations
        for i in range(2, len(antigen_providers) + 1):
            print(f"Checking combinations of size {i} for {antigen}...")
            for combination in combinations(antigen_providers.items(), i):
                total_capability = sum(capability for _, capability in combination)
                if total_capability >= committed_doses and all(
                    self.provider_covers_antigen(provider, antigen) for provider, _ in combination):
                          return [provider for provider, _ in combination][0]  # Return only the provider

        print(f"Unable to find providers or combinations to fulfill committed doses for {antigen}")
        return None  
    
       
    def provider_covers_antigen(self, provider, antigen):"""provider_covers_antigen: Checks if a specific provider produces vaccines that cover a given antigen."""
         # Check if the given provider produces vaccines that cover the specified antigen
            provider_prefix = self.initial_conditions.loc[antigen, "provider_prefix"]
            print(f"Antigen: {antigen}, Provider Prefix: {provider_prefix}")
            provider_expected_prefix = provider_prefix
            print(f"Antigen: {antigen}, Provider: {provider}, Expected Prefix: {provider_expected_prefix}")
            covers = provider.startswith(provider_expected_prefix)
            print(f"Antigen: {antigen}, Provider Covers: {covers}")
            return covers

    def update_provider_capabilities(self, antigen, provider, committed_doses):"""update_provider_capabilities: Updates provider capabilities after a tender is scheduled."""
        # Update provider capabilities after a tender is scheduled
        # Reduce the capabilities of the selected provider
        self.providers[antigen][provider] -= committed_doses
    
    def run_one_year_simulation(self, production_lead_time):"""run_one_year_simulation: Runs the simulation for one year."""
        # Run the simulation for one year
        for antigen in self.antigens:
            self.env.process(self.process_demand_with_tender_scheduling(production_lead_time))
            self.env.run(until=self.env.now+1)  # Run for one year

        # Collect and store results for the first year
        self.store_yearly_results()

    def store_yearly_results(self):"""store_yearly_results: Stores the results of the simulation for each antigen for each year."""
        result_vector = []
        for year in range(self.num_years):
            for antigen in self.antigens:
                # Fetch the necessary values for the current antigen
                current_inventory = self.initial_conditions.loc[antigen, 'current_inventory']
                tender_scheduled = self.initial_conditions.loc[antigen, 'tender_scheduled']
                committed_doses = self.initial_conditions.loc[antigen, f'committed_doses_year_{year}']
                partial_deliveries = self.initial_conditions.loc[antigen, 'partial_deliveries'] 
                purchase_cost = self.initial_conditions.loc[antigen, f'purchase_cost_year_{year}']
                holding_cost = self.initial_conditions.loc[antigen, f'holding_cost_year_{year}']
                unvaccinated_cost = self.initial_conditions.loc[antigen, 'unvaccinated_cost']
                # Create a dictionary containing the fetched values
                antigen_result = {
                    'Antigen': antigen,
                    'Year': year,
                    'Current Inventory': current_inventory,
                    'Tender Scheduled': 'Yes' if tender_scheduled else 'No',  # Convert boolean to string representation
                    'Committed Doses': committed_doses,
                    'Purchase Cost': purchase_cost,
                    'Holding Cost': holding_cost,
                    'Unvaccinated Cost': unvaccinated_cost
                }

                # Append the dictionary to the result vector
                result_vector.append(antigen_result)
        # Return the result_vector containing results for each antigen for each year
        return result_vector

    def update_initial_conditions(self):"""update_initial_conditions: Updates initial conditions for subsequent years based on stored results."""
        # Update initial conditions for subsequent years using the results from the stored result_vector
        for result in result_vector:
            antigen = result['Antigen']
            year = result['Year']
            # Update initial conditions based on the stored results for each antigen and year
            self.initial_conditions.loc[antigen, 'current_inventory'] = result['Current Inventory']
            self.initial_conditions.loc[antigen, 'tender_scheduled'] = result['Tender Scheduled']
            self.initial_conditions.loc[antigen, f'committed_doses_year_{year}'] = result['Committed Doses']
            self.initial_conditions.loc[antigen, f'purchase_cost_year_{year}'] = result['Purchase Cost']
            self.initial_conditions.loc[antigen, f'holding_cost_year_{year}'] = result['Holding Cost']
            self.initial_conditions.loc[antigen, 'unvaccinated_cost'] = result['Unvaccinated Cost']

    def run_remaining_years_simulation(self, production_lead_time):"""run_remaining_years_simulation: Runs the simulation for the remaining years."""
        # Run the simulation for the remaining years using updated initial conditions
        for year in range(2, self.num_years + 1):
            self.update_initial_conditions()  # Update initial conditions for each year
            for antigen in self.antigens:
                self.env.process(self.process_demand_with_tender_scheduling(antigen, production_lead_time))
            self.env.run(until=year)  # Run for the specific year

   
    def run_simulation(self, production_lead_time):"""run_simulation: Main function to run the entire simulation for multiple years."""
        # Run simulation for one year initially, then subsequent years
        self.run_one_year_simulation(production_lead_time)
        self.run_remaining_years_simulation(production_lead_time)

    
    def calculate_objective_function(self):"""calculate_objective_function: Calculates an objective function based on various costs and parameters."""
        objective_value = 0
        
        for antigen in self.antigens:
            for year in range(1):
                # Sum of purchase cost across all antigens, providers, and periods
                purchase_cost = self.initial_conditions.loc[antigen, f'purchase_cost_year_{year}']
                objective_value += purchase_cost

                # Sum of all tender costs across all antigens, providers, and periods
                tender_cost = self.initial_conditions.loc[antigen, 'tender_cost']
                objective_value -= tender_cost

                # Difference between individual purchase cost of a dose and reservation price per dose
                reservation_price = self.initial_conditions.loc[antigen, 'reservation_price']
                committed_doses = self.initial_conditions.loc[antigen, f'demand_year_{year}'] / self.initial_conditions.loc[antigen, 'doses_required']
                cost_difference = (reservation_price - self.initial_conditions.loc[antigen, 'purchase_cost']) * committed_doses
                objective_value -= cost_difference
        
        return objective_value

    def save_tender_scheduling_results(self):"""save_tender_scheduling_results: Saves tender scheduling results to a CSV file."""
        # Save tender scheduling results per antigen over 15 years to a CSV file
        with open('tender_scheduling_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Antigen', 'Year', 'Committed Doses']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for antigen in self.antigens:
                for year in range(1):
                    committed_doses = self.initial_conditions.loc[antigen, f'committed_doses_year_{year}']
                    writer.writerow({'Antigen': antigen, 'Year': year, 'Committed Doses': committed_doses})

    def save_yearly_values(self):"""save_yearly_values: Saves yearly values for each antigen to a CSV file."""
        # Save yearly values for each antigen to a CSV file
        with open('yearly_values.csv', 'w', newline='') as csvfile:
            fieldnames = ['Antigen', 'Year', 'Current Inventory', 'Committed Doses', 'Purchase Cost', 'Holding Cost', 'Unvaccinated Cost', 'Objective Function']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for antigen in self.antigens:
                for year in range(1):
                    current_inventory = self.initial_conditions.loc[antigen, 'current_inventory']
                    committed_doses = self.initial_conditions.loc[antigen, f'committed_doses_year_{year}']
                    purchase_cost = self.initial_conditions.loc[antigen, f'purchase_cost_year_{year}']
                    holding_cost = self.initial_conditions.loc[antigen, f'holding_cost_year_{year}']
                    unvaccinated_cost = self.initial_conditions.loc[antigen, 'unvaccinated_cost']
                    objective_function = self.calculate_objective_function()
                    writer.writerow({'Antigen': antigen, 'Year': year, 'Current Inventory': current_inventory, 'Committed Doses': committed_doses,
                                     'Purchase Cost': purchase_cost, 'Holding Cost': holding_cost, 'Unvaccinated Cost': unvaccinated_cost,
                                     'Objective Function': objective_function}) 
    

def main():"""Main Function (main):
Reads initial conditions from a CSV file.
Sets up the simulation environment (simpy.Environment()).
Initializes the simulation object (VaccineProcurementSimulation) and runs the simulation.
Overall Flow:
Read initial conditions.
Initialize simulation parameters.
Run the simulation for a specified number of years.
Store and update results.
Calculate objective functions and save results to CSV files.
"""
    # Specify the file path for your initial conditions CSV file
    file_path = 'initial_conditions.csv'

    # Read initial conditions from the CSV file
    initial_conditions = read_initial_conditions(file_path)
    
    print(initial_conditions)
    
    env = simpy.Environment()
    num_years=15
    production_lead_time=2

    # Instantiate the simulation with the read initial conditions
    simulation = VaccineProcurementSimulation(env, initial_conditions, num_years)

    # Run the simulation with a production lead time of 2 time units
    simulation.run_simulation(production_lead_time)

if __name__ == "__main__":
    main()
    main()
