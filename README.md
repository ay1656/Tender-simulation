Vaccine Procurement and Tender Scheduling Simulation
Overview

The project develops a discrete-event simulation framework for vaccine procurement and tender scheduling using the SimPy simulation environment. The models represent the interaction between vaccine demand, population growth, inventory levels, procurement decisions, provider production capacity, tender commitments, partial deliveries, procurement costs, and production lead times. The simulation is structured around a group of vaccine antigens, including Polio, Diphtheria, Tetanus, Pertussis, Mumps, Measles, and Rubella, with the initial conditions for each antigen being provided through CSV files and loaded into Pandas DataFrames. These initial conditions establish the operational and economic parameters required for the simulation, including unvaccinated population, current inventory, vaccine usage, birth cohorts, doses required for immunity, provider capacity, reorder parameters, reservation prices, holding costs, unvaccinated costs, purchase costs, and tender set-up costs.

The two codes follow the same fundamental procurement logic but differ in how the simulation experiments and results are managed. The first code extends the simulation with statistical analysis and response-surface modelling, while the second code reorganizes the simulation execution to support parallel replications through multiple CPU processes. Consequently, both implementations share the same core vaccine procurement mechanism while providing different approaches for experimentation, replication, and analysis.

Simulation Structure

Both codes begin by reading the initial conditions from CSV files and using these data to initialize a VaccineProcurementSimulation object. The simulation identifies the antigens directly from the input DataFrame and establishes a SimPy environment for representing the passage of time. Provider capabilities are generated for each antigen using a normal distribution based on the specified mean and standard deviation, with the resulting values converted to integers. Provider names are constructed using antigen-specific prefixes, allowing the model to associate providers with the vaccines that they are able to supply.

The simulation then progresses through a yearly procurement process. For each antigen, the model generates a birth cohort using a Poisson distribution based on the average number of births per year. The resulting population is combined with the existing unvaccinated population and vaccine usage to update the number of individuals requiring immunization. Vaccine demand is subsequently calculated by multiplying the unvaccinated population by the number of doses required to achieve immunity. At the same time, the model records measures such as unvaccinated cost and holding cost, thereby connecting population demand with the economic consequences of procurement and inventory decisions.

Tender Scheduling

The central component of both codes is the tender scheduling mechanism. A tender is evaluated separately for each antigen and is dependent on inventory conditions, provider capacity, and previous procurement activity. The model calculates a reorder point using the required number of doses, the simulated birth cohort, the time-space parameter, and an additional factor. A tender can subsequently be recorded when the inventory condition is satisfied, an appropriate provider has sufficient capacity, and there is no previous tender within the relevant time-space period.

Once a tender is scheduled, the model determines the number of doses to be committed and calculates the associated purchase and tender costs. The committed supply is divided by the time-space parameter to obtain partial deliveries, allowing the procurement quantity to be distributed across future periods rather than being treated as an immediate single delivery. This mechanism enables the simulation to represent the relationship between procurement timing and vaccine availability over several years.

Provider selection is performed by first examining individual providers according to their available capacity. If a single provider is unable to satisfy the required commitment, the model also examines combinations of providers to determine whether their combined capability is sufficient. Providers are additionally checked against the antigen-specific provider prefix to ensure that the selected supplier is associated with the relevant vaccine.

Production Capacity and Lead Time

An important feature of the simulation is the representation of production lead time. When a tender is awarded, the committed quantity is deducted from the selected provider's available capability. A SimPy process is then created to wait for the specified production lead time before returning the committed quantity to the provider's available capacity. This approach represents the temporary restriction of production capacity caused by an existing procurement commitment and allows future tenders to be influenced by previously committed production.

The committed quantities are also distributed across subsequent years as partial deliveries. This allows inventory to evolve according to both vaccine consumption and incoming deliveries. The resulting yearly records contain information on inventory, committed doses, partial deliveries, and whether a tender was scheduled for each antigen and year.

Objective Function and Results

Both implementations calculate an objective value based on tender-related economic measures. For periods in which a tender is scheduled, the model subtracts the tender set-up cost and adds a value based on the reservation price and the number of committed doses. The objective values are accumulated across antigens and simulation years to provide an overall measure for each simulated procurement scenario.

The simulation results are organized into Pandas DataFrames containing the antigen, simulation year, yearly inventory, committed doses, and tender schedule. These results, together with the calculated objective values, are subsequently exported to CSV files, providing a structured output that can be used for further analysis.

Code 1 – Statistical and Response-Surface Analysis

The first implementation extends the core simulation by incorporating statistical modelling through Statsmodels. In addition to the SimPy, Pandas, NumPy, and SciPy components used for the procurement simulation, it imports ordinary least squares regression, ANOVA, and related statistical tools.

The main simulation is executed across a predefined collection of initial-condition files representing different experimental combinations. Multiple replicates can be performed, with a different random seed assigned to each replicate. The objective values generated from these experiments are subsequently organized according to three experimental factors. The code assigns combinations of Factor A, Factor B, and Factor C, calculates average objective values for each combination, and creates squared terms to support a second-order response model.

The statistical component then constructs a second-order regression model containing the three linear factors, their quadratic terms, pairwise interaction terms, and two blocking factors. This provides a response-surface representation of the relationship between the experimental factors and the objective value. The fitted model is also used to calculate a gradient based on the linear coefficients, producing a direction vector for the model's steepest-ascent analysis.

Finally, the code generates a three-dimensional response surface using the fitted regression model. A grid of Factor A, Factor B, and Factor C values is constructed, predictions are generated from the regression model, and the resulting response is represented as a three-dimensional surface. The first code therefore combines the procurement simulation with an additional experimental-design and statistical-analysis layer, allowing simulation outputs to be examined in relation to predefined experimental factors.

Code 2 – Parallel Simulation and Replication

The second implementation retains the same underlying vaccine procurement model but places greater emphasis on computational execution and replication. It introduces a dedicated run_simulation() function that receives a random seed, simulation horizon, production lead time, and collection of input files. The function executes the simulation for each supplied input file, calculates the objective value, collects yearly results, and returns the resulting DataFrames together with the seed used for the experiment.

The main function then uses Python's ProcessPoolExecutor to distribute independent simulation replications across available CPU cores. Each replicate receives its own seed, allowing multiple stochastic versions of the procurement system to be evaluated. Once individual simulations have completed, their yearly results and objective values are combined into accumulated DataFrames and written to CSV files. This structure is intended to make the repeated execution of the stochastic simulation more computationally efficient.

Unlike the first implementation, the second code does not contain the subsequent response-surface and regression-analysis section. Its primary focus is therefore on executing the procurement simulation repeatedly and collecting the resulting operational and objective-function outputs. The two implementations consequently represent complementary stages of the modelling process: one emphasizes experimental and statistical analysis, while the other emphasizes parallelized simulation execution.

Common Simulation Flow

The overall logic of both codes can be summarized as follows:

Initial Conditions → Provider Initialization → Population and Demand Generation → Inventory Update → Tender Evaluation → Provider Selection → Tender Commitment → Partial Deliveries → Production Lead-Time Capacity Update → Objective Calculation → Results Export

The initial-condition files provide the parameters for the vaccine procurement environment. The simulation then generates stochastic population demand and evaluates inventory conditions for each antigen. When the tender conditions are satisfied, procurement quantities are committed and allocated to a suitable provider. The provider's available capability is temporarily reduced while production is in progress, and the resulting vaccine supply is distributed through partial deliveries. At the end of the simulation, yearly operational results and objective values are collected for subsequent analysis.

Outputs

The simulation produces two principal forms of output. The first is a yearly results dataset, which records the procurement state for each antigen and year, including yearly inventory, committed doses, and tender scheduling decisions. The second is an objective-value dataset, which records the calculated economic objective for each simulation experiment or replicate. The codes save these outputs as CSV files using filenames based on the simulation horizon and replicate seed.

Relationship Between the Two Codes

The two codes should be viewed as closely related implementations of the same vaccine procurement simulation rather than as completely independent models. Their common foundation is the SimPy-based representation of vaccine demand, inventory, tender scheduling, provider capacity, partial deliveries, production lead time, and procurement economics. The principal difference is the analytical layer surrounding the simulation.

The first code extends the model toward experimental analysis by organizing objective values according to combinations of three factors and fitting a second-order response-surface model with interaction, quadratic, and blocking terms. The second code extends the model toward computational experimentation by separating individual simulation runs into a reusable function and distributing multiple replications across CPU cores. Together, these approaches provide a framework in which the procurement model can first be executed under multiple scenarios and then analyzed according to the experimental factors represented in the simulation inputs.

Implementation Notes

Overall, the project establishes a simulation-based framework for studying vaccine procurement decisions under stochastic demand, limited provider capacity, inventory constraints, tender timing rules, and production lead times. The two implementations build upon the same procurement logic while supporting different analytical objectives: repeated and parallel simulation in one implementation, and experimental-design, regression, and response-surface analysis in the other.
