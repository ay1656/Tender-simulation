# Tender-simulation
###There are three versions of my simulation code. 
The version from October 30th 2023 works only the condition that only 1 provider per antigen and no previous tenders are scheduled. These conditions are not realisitic, thus the code was modified.
The version from November 1st attempted to address the previous scenario with only 1 provider, but it has a logic error in the main function, hence it enters an infinite loop.
The third and most recent version, November 12th, addresses both previous errors. The code has been reduced to 140 lines. Reduced significantly from more than 430 lines in previous iterations. It has not been tested and will be reviewed on November 13th
with Dr. Proaño.
I have added a newer version of the code, with today's date 11.13.2023. I made some adjustments to the code that I will review with Dr. Proaño.
The initial conditions file needs to be created but this will be performed once the logic has been discussed and validated.
After review with Dr. Proaño, the code has been updated to a version for 11/27/2023 V2 and the initial conditions file has been created. The simulation for year one initially performs adequately scheduling tenders for 3 of 4 antigens that meet the specified conditions. However, this version should stop iterating through years and only run for the initial year as I wished to add logic to use the results and feed into the next years, but it is not performing this activity thus encountering an error. This error is the one I wish to debug with Dr. Proaño  ###
