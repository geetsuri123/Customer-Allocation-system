# Customer-Allocation-system

<br>

## Introduction
Service providers serve customers from several regional facilities. Haket et al. (2020)
investigate the customer allocation problem which aim to save time and money as well as
reducing carbon dioxide emission. This project will be based on the mentioned paper
and you are asked to work on the following tasks.
<br>
Haket, C., van der Rhee, B. and de Swart, J., 2020. Saving time and money and reducing carbon
dioxide emissions by efficiently allocating customers. INFORMS journal on applied analytics,
50(3), pp.153-165.

<br>

## Tasks
1. (20%) Describe in your own words the optimization problem formulation presented in
Section Method: Mathematical Model, including decision variables, constraints, the
objective function, and all required parameters. State any necessary assumptions.
2. (20%) Explain in detail the construction heuristic (GA) and the improvement heuristic
(FI) presented in Section Method: Heuristic Algorithms, including the motivations.
State any necessary assumptions and you are encouraged to use figures, diagrams, and
examples to explain the algorithms if needed.
3. (30%) You are asked to consult a new service provider Prodnav Ltd. operating in
Great Britain. They have 13 regional facilities whose information, including their
nominal capacities (number of customer visits), are presented in facilities.xlsx. The
customers are grouped by local authorities and average customer demands are
proportional to the population with the rate of 0.1%. The population data is available
https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/
populationestimates/datasets/populationestimatesforukenglandandwalesscotlandan
dnorthernireland. Finally, distances between customer locations and facilities can be
calculated using their latitude and longitude values. Geography information of local
authorities can be found here: https://geoportal.statistics.gov.uk/. Use a constructive
heuristic to propose a customer allocation solution to the company. Explain in detail
how you prepare data, implement the heuristic in Python, and briefly analyse the
resulting customer allocation solution.
4. (30%) Use results from the previous parts to solve the problem using the optimization
problem formulation as well as the proposed heuristics in Python. Explain in detail how
you prepare additional required data/parameters for the problem. Analyse the
solutions by varying relevant parameters and comment on how the proposed
formulation, heuristics, and data preparation/settings can be improved or
implemented differently? 
