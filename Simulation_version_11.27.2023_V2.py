import numpy as np
import pandas as pd
import simpy
from scipy.stats import poisson
from itertools import combinations
import csv

# Set a fixed seed for reproducibility
seed = 42
np.random.seed(seed)

def read_initial_conditions(initial_conditions_csv):
    # Read initial conditions from a CSV file into a DataFrame
    initial_conditions = pd.read_csv(initial_conditions_csv, index_col=0)
    return initial_conditions

class VaccineProcurementSimulation:
    def __init__(self, env, initial_conditions, num_years):
        self.env = env
        self.initial_conditions = initial_conditions
        self.antigens = initial_conditions.index
        self.providers = self.initialize_providers()
        self.num_years=num_years

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
        # Debug statement to print providers dictionary
        print("Providers Dictionary:", providers)
        return providers
    

    def process_demand_with_tender_scheduling(self, production_lead_time):
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
    
    def is_tender_scheduled(self, antigen, year, committed_doses):
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

    def has_previous_tender(self, antigen, year, time_space):
        tender_schedule = self.initial_conditions.loc[antigen, 'tender_scheduled']
        for i in range(1, min(year,time_space) + 1):
            index_to_check=year-i
            if index_to_check >= 0:
                print(f"Checking index: {index_to_check}, Value: {tender_schedule[index_to_check]}") 
                if tender_schedule[index_to_check] == 1:
                    return True
    
        return False

    def choose_provider(self, antigen, committed_doses):
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
    
       
    def provider_covers_antigen(self, provider, antigen):
         # Check if the given provider produces vaccines that cover the specified antigen
            provider_prefix = self.initial_conditions.loc[antigen, "provider_prefix"]
            print(f"Antigen: {antigen}, Provider Prefix: {provider_prefix}")
            provider_expected_prefix = provider_prefix
            print(f"Antigen: {antigen}, Provider: {provider}, Expected Prefix: {provider_expected_prefix}")
            covers = provider.startswith(provider_expected_prefix)
            print(f"Antigen: {antigen}, Provider Covers: {covers}")
            return covers

    def update_provider_capabilities(self, antigen, provider, committed_doses):
        # Update provider capabilities after a tender is scheduled
        # Reduce the capabilities of the selected provider
        self.providers[antigen][provider] -= committed_doses
    
    def run_one_year_simulation(self, production_lead_time):
        # Run the simulation for one year
        for antigen in self.antigens:
            self.env.process(self.process_demand_with_tender_scheduling(production_lead_time))
            self.env.run(until=self.env.now+1)  # Run for one year

        # Collect and store results for the first year
        self.store_yearly_results()

    def store_yearly_results(self):
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

    def update_initial_conditions(self):
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

    def run_remaining_years_simulation(self, production_lead_time):
        # Run the simulation for the remaining years using updated initial conditions
        for year in range(2, self.num_years + 1):
            self.update_initial_conditions()  # Update initial conditions for each year
            for antigen in self.antigens:
                self.env.process(self.process_demand_with_tender_scheduling(antigen, production_lead_time))
            self.env.run(until=year)  # Run for the specific year

   
    def run_simulation(self, production_lead_time):
        # Run simulation for one year initially, then subsequent years
        self.run_one_year_simulation(production_lead_time)
        self.run_remaining_years_simulation(production_lead_time)

    
    def calculate_objective_function(self):
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

    def save_tender_scheduling_results(self):
        # Save tender scheduling results per antigen over 15 years to a CSV file
        with open('tender_scheduling_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Antigen', 'Year', 'Committed Doses']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for antigen in self.antigens:
                for year in range(1):
                    committed_doses = self.initial_conditions.loc[antigen, f'committed_doses_year_{year}']
                    writer.writerow({'Antigen': antigen, 'Year': year, 'Committed Doses': committed_doses})

    def save_yearly_values(self):
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
    

def main():
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