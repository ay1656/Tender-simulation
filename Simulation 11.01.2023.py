import numpy as np
import pandas as pd
import simpy
import random
from scipy.stats import poisson

# Set a fixed seed for reproducibility
seed = 42
np.random.seed(seed)

# Read initial conditions from the "Data simulation" data frame
initial_conditions_df = pd.read_csv("Simulation_Data.csv")

# Define the simulation duration
simulation_duration = 15

# Create a dictionary to map antigens to the vaccines produced by manufacturers
antigen_to_vaccine_mapping = {
    "Polio": ["Provider 1", "Provider 2", "Provider 3"],
    "Diphtheria": ["Provider 3", "Provider 4", "Provider 5"],
    "Tetanus": ["Provider 3", "Provider 4", "Provider 5"],
    "Pertussis": ["Provider 3", "Provider 4", "Provider 5"],
    "Measles": ["Provider 6", "Provider 7", "Provider 8"],
    "Mumps": ["Provider 6", "Provider 7", "Provider 8"],
    "Rubella": ["Provider 6", "Provider 7", "Provider 8"]
}

class AntigenDemand:
    def __init__(self, antigen_data, simulation_duration, vaccine_mapping):
        self.antigen = antigen_data['Antigen']
        self.simulation_duration = simulation_duration
        self.seed=seed
        self.inventory = {year: 0 for year in range(simulation_duration + 1)}
        self.inventory[0] = antigen_data['Inventory']
        self.doses_required = antigen_data['DosesRequired']
        self.time_space = antigen_data['TimeSpace']
        self.average_birth_rate = antigen_data['AverageBirthRate']
        self.cost_per_unvaccinated = antigen_data['CostPerUnvaccinated']
        self.holding_cost_per_dose = antigen_data['HoldingCostPerDose']
        self.unvaccinated = {year: 0 for year in range(simulation_duration + 1)}
        self.unvaccinated[0] = antigen_data['UnvaccinatedPopulation']
        self.demand = {year: 0 for year in range(simulation_duration + 1)}
        self.partial_delivery = {year: 0 for year in range(simulation_duration + 1)}
        self.partial_delivery[0]= antigen_data['PartialDelivery']
        self.partial_deliveries = {year: 0 for year in range(1, simulation_duration + 1)}
        self.total_cost = {year: 0 for year in range(simulation_duration + 1)}
        self.reorder_point = {year: self.doses_required*self.average_birth_rate for year in range(1, simulation_duration + 1)}
        self.tender_schedule = {}
        self.tender_cost = antigen_data['TenderCost']
        self.vaccines_covering_antigen = antigen_data['VaccinesCoveringAntigen']
        self.scheduled_tenders = {}
        self.reservation_price={year:{} for year in range(1,self.simulation_duration+1)}
        self.reservation_price = antigen_data['ReservationPrice']  # Added reservation price
        self.production_lead_time=antigen_data['ProductionLeadTime']
        self.penalty=antigen_data['Penalty']
        self.purchase_costs = {year: {} for year in range(1, simulation_duration + 1)}
        self.expected_prices = {year: {} for year in range(1, simulation_duration + 1)}
        self.price_difference = {year: {antigen: 0 for antigen in antigens_to_simulate} for year in range(1, simulation_duration + 1)}
        self.vaccine_mapping=antigen_to_vaccine_mapping
        print(self.antigen)
        print('Inventory:',self.inventory)
        print('Unvaccinated:',self.unvaccinated)
        print('Partial delivery:',self.partial_delivery)
        print('Reorder point:', self.reorder_point)
        print('Tender cost:', self.tender_cost)

        

  
  
    def update_initial_inventory(self, year, partial_delivery, partial_deliveries):
        for year in range(1, simulation_duration):
            if year==1:
                self.used_doses= random.uniform(0.1, 0.9)*self.partial_delivery[0]
                self.inventory[year] = self.inventory[0] + self.partial_delivery[0]-self.used_doses            
            else:
                self.used_doses= random.uniform(0.1, 0.9)*self.partial_deliveries[year]
                self.inventory[year]= self.inventory[year-1]+ partial_deliveries[year]-self.used_doses
                    
    
    def update_unvaccinated(self, year, partial_delivery, partial_deliveries):
        average_births = self.average_birth_rate
        birth_cohort = poisson.rvs(average_births)
        if year == 1:
            self.used_doses= random.uniform(0.1, 0.9)*self.partial_delivery[0]
            immunizations=self.used_doses
            self.unvaccinated[1] = self.unvaccinated[0]+birth_cohort-immunizations
            print('Birth cohort:', birth_cohort)
            print('Used_doses:', immunizations)
        else:    
            self.used_doses= random.uniform(0.1, 0.9)*self.partial_deliveries[year]
            immunizations=self.used_doses
            self.unvaccinated[year] = self.unvaccinated[year - 1] + birth_cohort-immunizations
            print('Birth cohort:', birth_cohort)
            print('Used_doses:', immunizations)

        
    def calculate_demand(self, year):
        average_births = self.average_birth_rate
        birth_cohort = poisson.rvs(average_births)
        if year == 1:
            true_demand= self.unvaccinated[1] *self.doses_required
            self.demand[year] = int(true_demand)
            print('Demand:', true_demand)
        else:
            if year in self.unvaccinated:
                true_demand= self.unvaccinated[year-1]*self.doses_required
                self.demand[year] = int(true_demand)
                print('Demand:', true_demand)
            
        return true_demand
       
            
    def calculate_costs(self, year):
        if year == 1:
            unvaccinated_cost=self.unvaccinated[1]*self.cost_per_unvaccinated
        else:
            unvaccinated_cost = self.unvaccinated[year-1] * self.cost_per_unvaccinated

        if year == 1:
            holding_cost = self.inventory[1] * self.holding_cost_per_dose
        else:
            holding_cost = self.inventory[year - 1] * self.holding_cost_per_dose
        
        # Check if a tender is scheduled for the current year
        tender_cost = int(self.tender_cost) if self.scheduled_tenders.get(year, False) else 0

        self.total_cost[year] = unvaccinated_cost + holding_cost + tender_cost

        
    def schedule_tender(self, year):
     
        # Calculate the birth cohort based on the average birth rate
        average_births = self.average_birth_rate
        birth_cohort = poisson.rvs(average_births)
        
        # Retrieve relevant parameters
        time_space = self.time_space
        current_inventory = self.inventory[year]
        
        # Initialize variables
        reorder_point = self.reorder_point[year] * birth_cohort
        tenders_scheduled = False
        tender_year = None
        unvaccinated_cost = self.unvaccinated[year] * self.cost_per_unvaccinated if year > 0 else 0
        holding_cost = self.inventory[year] * self.holding_cost_per_dose if year > 0 else 0

        # Check if this year is scheduled for a tender
        if year not in self.scheduled_tenders:
            # Check if inventory is below reorder point for the current year
            if current_inventory <= reorder_point:
                # Check if there are no tenders scheduled in the years before it within the time-space interval
                no_scheduled_tenders = all(self.tender_schedule.get(year - ts, 0) == 0 for ts in range(1, int(time_space) + 1))
                if no_scheduled_tenders:
                    tenders_scheduled = True
                    tender_year = year + time_space
                    self.tender_schedule[tender_year] = 1
                    

        
        return tenders_scheduled, tender_year
    
        # Calculate costs based on whether tenders are scheduled
        
        if tenders_scheduled:
            self.total_cost[year] = unvaccinated_cost + holding_cost + self.tender_cost
        else:
            self.total_cost[year] = unvaccinated_cost + holding_cost
    
    def commit_vaccines(self, year, committed_doses):
        if self.scheduled_tenders.get(year, False):  # Check if a tender is scheduled for the current year
            # Sum the true demand from the current year until the current year plus time_space - 1
            committed_doses = sum(self.calculate_demand(year - ts) for ts in range(self.time_space))
            #partial_deliveries = {}
            for ts in range(1, self.time_space + 1):
                self.partial_deliveries[year-ts] = committed_doses/self.time_space
            print('Committed doses:', committed_doses)
            print('Partial deliveries:', self.partial_deliveries[year-ts])
            
        else:
            # If no tender is scheduled, committed doses are zero
            committed_doses = 0 
            partial_deliveries={year:0}
            print('Committed doses:', committed_doses)
        return committed_doses
    
        
        
    def calculate_purchase_costs(self, year):
        committed_doses=0
        for year in range(1, self.simulation_duration + 1):
            committed_doses = self.commit_vaccines(year, committed_doses)
            # Generate a random vaccine_cost using the random seed
            random.seed(seed)  # Set the random seed
            min_cost = 30  # Adjust the minimum and maximum cost values as needed
            max_cost = 70
            vaccine_cost = random.uniform(min_cost, max_cost)  
            self.purchase_costs[year][self.antigen] = committed_doses * vaccine_cost
        return self.purchase_costs[year][self.antigen]     

    def calculate_expected_prices(self, year):
        committed_doses=0
        for year in range(1, self.simulation_duration + 1):
            committed_doses = self.commit_vaccines(year, committed_doses)
            for antigen in antigens_to_simulate:
                reservation_price = self.reservation_price
                self.expected_prices[year][antigen] = committed_doses * reservation_price
        
        return self.expected_prices[year][self.antigen]
                  
    
    def calculate_price_difference(self,year):
        # Calculate the difference between expected price and purchase cost for each antigen
        self.price_difference = {year: {} for year in range(1, self.simulation_duration + 1)}
        for year in range(1, self.simulation_duration + 1):          
            expected_prices_year = self.calculate_expected_prices(year)
            purchase_costs_year = self.calculate_purchase_costs(year)
            for antigen_data in antigen_data_list:  # Loop through antigens
                expected_price = expected_prices_year
                purchase_cost = purchase_costs_year
            
            
                if expected_price is not None and purchase_cost is not None:
                    price_difference = expected_price - purchase_cost
                    self.price_difference[year]=int(price_difference)
                    print("Price Difference:", price_difference)
    
    def calculate_objective_function(self, year, committed_doses):
        total_purchase_costs = 0
        total_tender_costs = 0
        total_penalty = 0
        

        for year in range(1, self.simulation_duration + 1):
            for antigen in antigens_to_simulate:
                total_purchase_costs += self.purchase_costs[year][self.antigen]
                if self.scheduled_tenders.get(year, False):
                    total_tender_costs += self.tender_cost
                total_penalty += self.price_difference[year] * committed_doses * self.penalty

        # Calculate the overall objective function
        objective_function = total_purchase_costs - total_tender_costs - total_penalty
        return objective_function


             
                    
class Vaccine:
    def __init__(self, vaccine_mapping, simulation_duration):
        self.provider=vaccine_mapping
        self.simulation_duration = simulation_duration
        self.vaccine_providers = antigen_to_vaccine_mapping
        self.provider_capabilities = {}  # Use a dictionary to store capabilities
        self.committed_capacity = {vaccine: {year: 0 for year in range(1, simulation_duration + 1)} for vaccine in vaccine_mapping}
        
    def calculate_provider_capabilities(self, year):
        # Define the mean and standard deviation for capability
        mean_capability = 1000000
        std_dev_capability = 200
        
        capabilities = {}
        
        # Calculate provider capabilities for each vaccine for the given year
        for vaccine in self.vaccine_providers:
            capability = int(np.random.normal(mean_capability, std_dev_capability))
            capabilities[vaccine] = capability
        
        self.provider_capabilities[year] = capabilities  # Store capabilities for the given year
        
        return capabilities  
        print('Capability:', self.provider_capabilities[year])
    
    def commit_capacity(self, year, vaccine_name, capacity):
        # Update the committed capacity for the specified year and vaccine
        self.committed_capacity[vaccine_mapping][year]+= capacity

    def get_committed_capacity(self, year, vaccine_name):
        # Retrieve the committed capacity for the specified year and vaccine
        return self.committed_capacity[vaccine_name][year]
    

# Define a list of antigens to consider
antigens_to_simulate = ["Polio", "Diphtheria","Tetanus","Pertussis", "Measles", "Mumps", "Rubella"]

# Create a list of dictionaries, where each dictionary represents the data for an antigen
antigen_data_list = []

production_lead_time=2

for antigen in antigens_to_simulate:
    antigen_data = initial_conditions_df[initial_conditions_df['Antigen'] == antigen].iloc[0].to_dict()
    antigen_data['ProductionLeadTime']=production_lead_time
    antigen_data_list.append(antigen_data)

# Define a list of vaccines to consider
vaccines_to_simulate = ["VaccineA", "VaccineB", "VaccineC", "VaccineD", "VaccineE", "VaccineF"]

# Create a list to hold instances of AntigenDemand for each antigen
antigen_demand_objs = [AntigenDemand(antigen_data, simulation_duration, antigen_to_vaccine_mapping) for antigen_data in antigen_data_list]

# Create a list to hold instances of Vaccine for each vaccine
vaccine_objs = [Vaccine(antigen_to_vaccine_mapping, simulation_duration) for vaccine_name in vaccines_to_simulate]

def simulation(env, antigen_demand_objs, vaccine_objs, simulation_duration, committed_doses):
    current_year = 1  # initialize current year
    committed_doses = 0 #initialize committed doses.
    while current_year <= simulation_duration:
        
        # Calculate demand for all years up to and including the current year
        for antigen_obj in antigen_demand_objs:
            for year in range(1, simulation_duration+1):
                antigen_obj.calculate_demand(year)
                
        # Calculate costs and schedule tenders for the current year
        for antigen_obj in antigen_demand_objs:
            committed_doses = antigen_obj.commit_vaccines(current_year, committed_doses)  # Pass committed_doses
            partial_deliveries=antigen_obj.commit_vaccines(current_year, committed_doses)
            antigen_obj.calculate_costs(current_year)
            antigen_obj.update_unvaccinated(current_year, antigen_obj.partial_delivery, antigen_obj.partial_deliveries)
            antigen_obj.calculate_purchase_costs(year)  # Call this method
            antigen_obj.calculate_expected_prices(year)  # Call this method
            antigen_obj.calculate_price_difference(year)  # Now, you can calculate the price difference
            antigen_obj.update_initial_inventory(year, antigen_obj.partial_delivery, antigen_obj.partial_deliveries)
            antigen_obj.calculate_objective_function(year, committed_doses)
        
        #Calculate objective function
        for antigen_obj in antigen_demand_objs:
            for year in range(1, simulation_duration+1):
                objective_function_value = antigen_obj.calculate_objective_function(year, committed_doses)
                
        #capabilites={}
        #for vaccine_obj in vaccine_objs:
         #   capabilities = vaccine_obj.calculate_provider_capabilities(current_year)
          #  vaccine_obj.provider_capabilities[current_year] = capabilities
        
        while current_year<= simulation_duration:
            for antigen_obj in antigen_demand_objs:
                #Calculate the birth cohort based on the average birth rate
                average_births = antigen_obj.average_birth_rate
                birth_cohort = poisson.rvs(average_births)
                
                # Retrieve relevant parameters
                time_space = antigen_obj.time_space
                current_inventory = antigen_obj.inventory[year]
                
                # Initialize variables
                reorder_point = antigen_obj.reorder_point[year] * birth_cohort
                tenders_scheduled = False
                tender_year = None
                unvaccinated_cost = antigen_obj.unvaccinated[year] * antigen_obj.cost_per_unvaccinated if year > 0 else 0
                holding_cost = antigen_obj.inventory[year] * antigen_obj.holding_cost_per_dose if year > 0 else 0
            
          
    
                # Check if this year is scheduled for a tender
                if year not in antigen_obj.scheduled_tenders:
                    # Check if inventory is below reorder point for the current year
                    if current_inventory <= reorder_point:
                        # Check if there are no tenders scheduled in the years before it within the time-space interval
                        no_scheduled_tenders = all(antigen_obj.tender_schedule.get(year - ts, 0) == 0 for ts in range(1, int(time_space) + 1))
                        if no_scheduled_tenders:
                            tenders_scheduled = True
                            tender_year = year + time_space
                            antigen_obj.tender_schedule[tender_year] = 1
        
        return tenders_scheduled, tender_year
        
        #if tenders_scheduled:
         #   vaccines_produced = antigen_obj.vaccine_mapping[antigen_obj.antigen]
          #  for vaccine_obj in vaccine_objs:
           #     if vaccine_obj.name in vaccines_produced:      
            #        committed_capacity = calculate_committed_capacity()  # Calculate the committed capacity
             #       vaccine_obj.commit_capacity(current_year, vaccine_obj.name, committed_capacity)    

        
        
        
        # Advance the simulation time
        yield env.timeout(1)  # Move the time advancement here
        current_year += 1  # Increment the current year

# Run the simulation for 15 years
simulation_duration = 15
env = simpy.Environment()
committed_doses=0
env.process(simulation(env, antigen_demand_objs, vaccine_objs, simulation_duration, committed_doses))
env.run(until=simulation_duration)

# Define the collect_results function to organize simulation results into a DataFrame
def collect_results(antigen_demand_objs, vaccine_objs, committed_doses):
    results = []
    
    for antigen_obj in antigen_demand_objs:
        for year, demand in antigen_obj.demand.items():
            results.append({
                'Year': year,
                'Antigen': antigen_obj.antigen,
                'Inventory': antigen_obj.inventory,
                'Demand': demand,
                'TenderScheduled': antigen_obj.scheduled_tenders.get(year, 0),
                'CommittedDoses': antigen_obj.commit_vaccines(year, committed_doses),
                'Partial Delivery':antigen_obj.partial_deliveries,
            })
                      
    
    
    # Convert the scheduled_tenders dictionary to a DataFrame
    tender_results = []
    for antigen_obj in antigen_demand_objs:
        for year, scheduled in antigen_obj.scheduled_tenders.items():
            tender_results.append({
                'Year': year,
                'Antigen': antigen_obj.antigen,
                'Scheduled_Tender': scheduled
            })
            
    results_df = pd.DataFrame(results)
    tender_df = pd.DataFrame(tender_results)
   
    

    return results_df, tender_df

def collect_metrics(antigen_demand_objs, vaccine_objs, committed_doses):
    
    metrics = []
    
    for antigen_obj in antigen_demand_objs:
        for year, demand in antigen_obj.demand.items():
            metrics.append({
                'Year': year,
                'Antigen': antigen_obj.antigen,
                'Total_Cost': antigen_obj.total_cost,
                'Purchase Cost': antigen_obj.calculate_purchase_costs(year),
                'TenderCost': antigen_obj.tender_cost if antigen_obj.scheduled_tenders.get(year, False) else 0,
                'PriceDifference': antigen_obj.price_difference,
                'Objective Function': antigen_obj.calculate_objective_function(year, committed_doses)
            })
    
    metrics_df=pd.DataFrame(metrics)
    
    return metrics_df

def collect_resultsv(vaccine_objs):    
    resultsv = []           
    for vaccine_obj in vaccine_objs:
        for year, capabilities in vaccine_obj.provider_capabilities.items():
            for vaccine, capability in capabilities.items():
                resultsv.append({
                    'Year': year,
                    'Vaccine': vaccine_obj.name,
                    'Capability': capabilities
                    })
    resultsv_df=pd.DataFrame(resultsv)
    return resultsv_df

# Collect simulation results into DataFrames
results_df, tender_df= collect_results(antigen_demand_objs, vaccine_objs, committed_doses)
#Collect si ulation metrics into DataFrames
metrics_df=collect_metrics(antigen_demand_objs, vaccine_objs, committed_doses)
# Collect simulation results into DataFrames
resultsv_df=collect_resultsv(vaccine_objs)

# Save the results to CSV files
results_df.to_csv('simulation_results.csv', index=False)
resultsv_df.to_csv('simulation_resultsv.csv', index=False)
tender_df.to_csv('scheduled_tenders.csv', index=False)
metrics_df.to_csv('simulation_metrics.csv', index=False)