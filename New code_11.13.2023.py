import numpy as np
import pandas as pd
import simpy
import random
from scipy.stats import poisson

# Set a fixed seed for reproducibility
seed = 42
np.random.seed(seed)

def read_initial_conditions(initial_conditions.csv):
    # Read initial conditions from a CSV file into a DataFrame
    initial_conditions = pd.read_csv(initial_conditions.csv, index_col=0)
    return initial_conditions


class VaccineProcurementSimulation:
    def __init__(self, env, initial_conditions):
        self.env = env
        self.initial_conditions = initial_conditions
        self.antigens = initial_conditions.index
        self.providers = self.initialize_providers()

        # Add additional simulation environment variables and resources as needed

    def initialize_providers(self):
        # Initialize providers with capabilities based on normal distribution
        providers = {}
        for antigen in self.antigens:
            mean_capability = self.initial_conditions.loc[antigen, 'provider_mean_capability']
            std_dev_capability = self.initial_conditions.loc[antigen, 'provider_std_dev_capability']
            num_providers = self.initial_conditions.loc[antigen, 'num_providers']
            provider_prefix = antigen  # This assumes the prefix is the same as the antigen name
            
            providers[antigen] = {
                f'Provider{i+1}': int(np.random.normal(mean_capability, std_dev_capability))
                for i in range(num_providers)
            }
            # Add provider_prefix to the initial conditions DataFrame
            self.initial_conditions.loc[antigen, 'provider_prefix'] = provider_prefix
       
        return providers

    
    def process_demand(self, year):
         while True:
                for antigen in self.antigens:
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
                    demand = unvaccinated_individuals * doses_required
                    # Calculate unvaccinated cost per antigen
                    unvaccinated_cost = unvaccinated_individuals * self.initial_conditions.loc[antigen, 'cost_per_unvaccinated_individual']
                    # Update relevant variables
                    self.initial_conditions.loc[antigen, 'current_inventory'] = current_inventory
                    self.initial_conditions.loc[antigen, 'unvaccinated_individuals'] = unvaccinated_individuals
                    self.initial_conditions.loc[antigen, 'demand'] = demand
                    self.initial_conditions.loc[antigen, 'unvaccinated_cost'] = unvaccinated_cost
                    yield self.env.timeout(1)

    def tender_scheduling(self, year):
        while True:
            for antigen in self.antigens:
                if self.is_tender_scheduled(antigen, year):
                    provider = self.choose_provider(antigen)
                    committed_doses = self.initial_conditions.loc[antigen, 'demand'] * self.initial_conditions.loc[antigen, 'time_space']
                    tender_cost = self.initial_conditions.loc[antigen, 'fixed_setup_cost']

                    # Calculate partial deliveries
                    partial_deliveries = committed_doses / self.initial_conditions.loc[antigen, 'time_space']

                    # Update provider capabilities
                    self.update_provider_capabilities(provider, committed_doses)

                    # Update relevant variables
                    self.initial_conditions.loc[antigen, 'tender_cost'] = tender_cost
                    self.initial_conditions.loc[antigen, 'partial_deliveries'] = partial_deliveries

    def is_tender_scheduled(self, antigen, year):
        
        reorder_point = self.initial_conditions.loc[antigen, 'reorder_point']
        time_space = self.initial_conditions.loc[antigen, 'time_space']
        provider = self.choose_provider(antigen)

        # Check conditions for tender scheduling
        if (
            self.initial_conditions.loc[antigen, 'current_inventory'] <= reorder_point
            and not self.has_previous_tender(antigen, year, time_space)
            and self.providers[antigen][provider] >= self.initial_conditions.loc[antigen, 'demand'] * time_space
        ):
            return True
        return False

    def has_previous_tender(self, antigen, year, time_space):
        for i in range(1, time_space + 1):
            if self.initial_conditions.loc[antigen, 'tender_scheduled'][year - i]:
                return True
        return False

    def choose_provider(self, antigen, committed_doses):
        eligible_providers=[]
        # Iterate through providers for the given antigen
        for provider, capability in self.providers[antigen].items():
            # Check if the provider has enough capability to cover the committed doses
            if capability >= committed_doses:
                # Check if the provider produces vaccines that cover the antigen
            if self.provider_covers_antigen(provider, antigen):
                eligible_providers.append(provider)

        if not eligible_providers:
            # If no eligible providers are found, raise an exception or handle the situation accordingly
            raise ValueError("No eligible provider found for the given antigen and committed doses.")
        
        # Randomly choose a provider from the eligible ones
        chosen_provider = np.random.choice(eligible_providers)
        return chosen_provider

    def provider_covers_antigen(self, provider, antigen):
    # Check if the given provider produces vaccines that cover the specified antigen
    return provider.startswith(f'Provider{self.initial_conditions.loc[antigen, "provider_prefix"]}')

    def update_provider_capabilities(self, provider, committed_doses):
        # Update provider capabilities after a tender is scheduled
        # Reduce the capabilities of the selected provider
        # Placeholder for illustration, replace with actual implementation
        self.providers[antigen][provider] -= committed_doses
        pass

    def run_simulation(self, num_years):
        for antigen in self.antigens:
            # Start SimPy processes for each antigen
            self.env.process(self.process_demand(antigen))
            self.env.process(self.tender_scheduling(antigen))

        # Run the simulation for a specified number of years
        self.env.run(until=num_years)
def main():
    # Specify the file path for your initial conditions CSV file
    file_path = 'initial_conditions.csv'

    # Read initial conditions from the CSV file
    initial_conditions = read_initial_conditions(file_path)
    
    env=simpy.Environment()

    # Instantiate the simulation with the read initial conditions
    simulation = VaccineProcurementSimulation(initial_conditions)

    # Run the simulation
    simulation.run_simulation(num_years=15)

if __name__ == "__main__":
    main()
