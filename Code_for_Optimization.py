
#importing necessary libraries

import pandas as pd
import pulp as pl
import random as random
import matplotlib.pyplot as plt



#Importing necessary data
distance_v = pd.read_excel("Distance, Capacity and demand.xlsx", sheet_name = "Distance",index_col=0)

demand_v =  pd.read_excel("Distance, Capacity and demand.xlsx", sheet_name = "Demand", index_col = 0)

capacity_v = pd.read_excel("Distance, Capacity and demand.xlsx", sheet_name = "Capacity", index_col = 0)




# Def gaHuerestic function 
def gaHueristic(distance_matrix, demand_values, capacity_values):
    
    #creating a copy of capacity values
    
    capacity_c = capacity_values.copy()
    
    #Creating an empty allocation matrix(using the format of distance matrix)
    solution = pd.DataFrame(index = distance_matrix.index, columns= distance_matrix.columns)
    
    #Creating a copy of demand matrix to preserve original values in its copy.
    demand = pd.DataFrame(demand_values)
    
    #Greedy Adaptive Algorithm
    
    #Ordering the customers in descending order of their demand
    demand_values = demand_values.sort_values(by=['Demand'],ascending=False)
    
    #Making a while loop. Here, each iteration represents allocation of each selected customerq
    #setting i = 1, which will increment in while loop for setting different random seed process for different iteration
    i = 1
    while True:
        #setting a random seed for random process below
        random.seed(i)
        
        #Selecting a random number between 1 and the number of remaining customers
        number_of_rows = random.randint(1, len(demand_values))
        
        #Creating a list of the random number of largest remaining Customer. 
        restricted_candidate_list = demand_values.head(number_of_rows)
        
        #Randomly selecting a row index value (selecting a customer) from the restricted candidate list.
        random_number_for_topc = random.randint(0, len(restricted_candidate_list)-1)
        
        #Using the selected row value to select the customer row of the above series
        demand_of_selected_customer = restricted_candidate_list.iloc[random_number_for_topc,:]
        
        #Name of customer
        name_of_selected_customer = demand_of_selected_customer.name
        
        #Demand Value of the customer
        demand_value_of_selected_customer = demand_of_selected_customer[0]
        
        #Creating a series of the distances of the selected customer with each of the facility location, and sorting them in ascending order.
        ## arranged in Lowest Value to Highest value of distance
        distances_for_selected_customer = distance_matrix.loc[name_of_selected_customer,:].sort_values()
        
        #initializing initial number of unallocated customers
        initial_lenof_demand_values = len(demand_values)
        
       
        
        #Incrementing i
        i = i + 1
        
        #For the selected customer, looping over all facilities in the distance series in ascending order of their distance 
        for facility in distances_for_selected_customer.index:
            
            # if the available capacity of the facility is mroe or equal to demand value of the customer
            if capacity_c.loc[facility, "Capacity (1000 visits)"] * 1000 >= demand_value_of_selected_customer:
                
                #Allocate the customer to this facility
                solution.loc[name_of_selected_customer, facility] = 1
                
                #Delete the customer from the original demand list
                demand_values.drop(name_of_selected_customer, inplace = True)
                
                #Reduce the capacity of the facility by the newly allocated customer's demand value
                capacity_c.loc[facility, "Capacity (1000 visits)"] -= demand_value_of_selected_customer/1000
                
                #break the for loop of going through the next facilities
                break
            
        
        #After breaking the loop, check if the original demand list is equal to the demand list currently. If yes, then customer is not allocated
        if initial_lenof_demand_values == len(demand_values):
            
            #Since, customer is not allocated, just delete from the demand list, and do not do anything.
            demand_values.drop(name_of_selected_customer, inplace = True)
            
            # If Length of demand values are 0 now, then we have to break the loop of customers, as all customers are done checking for.
            if len(demand_values) == 0:
                break
            
        else:
            # If Length of demand values are 0 now, then we have to break the loop of customers, as all customers are done checking for.
            if len(demand_values) == 0:
                break
        
        
    
        
    #Once all custoemrs are done, replace the na values with 1 in solution matrix. All the other values are 1 because of above for loop.
    solution.fillna(0, inplace = True)
    
    #Analysis
    
    #Creating allocated total distance matrix
    matrix = distance_matrix * solution
    
    #Calculating the total distance that each customer has to travel in one visit.
    matrix = matrix.sum(axis=1)
    
    
    #Multiplying customers demand with the distance in 1 visit to calculation total allocation costs
    allocation_costs = matrix.sort_index() * demand.sort_index().loc[:,"Demand"]
    
    #returning a list of solution matrix, as well as allocation cost of each customer]
    return [solution, allocation_costs]
    
#Using the above function to find solution for the given dataset
solution_list = gaHueristic(distance_v, demand_v, capacity_v)

#subsetting allocation matrix
allocation_matrix = solution_list[0]

#subsetting allocation_cost
allocation_cost = solution_list[1]



#printing the result
print(f'''The total distance value achieved by GA construction Heuristic is {allocation_cost.sum()}. Below is the allocation matrix: 
      {allocation_matrix}''')
      
#creating histogram to breifly analyze the solution
histogram = plt.hist(allocation_cost, bins=20, color='c', edgecolor = 'k', alpha = 0.65 )
plt.axvline(allocation_cost.mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel("Allocation Cost")
plt.ylabel("Number of Customers")
plt.title("Distribution of Allocation Costs of Different customers after applying GA")
plt.text(13000,100,f'Mean: {round(allocation_cost.mean())}',rotation=0)
plt.show()

#creating an empty capacity relaxation dataframe
capacity_relaxation_matrix = pd.DataFrame(columns = ["Capacity_increment","Total_Distance_Value"])

#Filling values in the capacity relaxation table
#initializing an index
index = 0
for capacity_increase in range(0, 55, 10):
    
    #transforming capacity_v table that was imported by adding each value with capacty increment
    capacity = capacity_v* (100 + capacity_increase)/100
    
    #calculating total allocation cost
    totalcost = gaHueristic(distance_v, demand_v, capacity)[1].sum()
    
    #adding this value in the appropriate row
    capacity_relaxation_matrix.loc[index,"Total_Distance_Value"] = totalcost
    
    #also adding the percentage of capacity increment  done in the same row
    capacity_relaxation_matrix.loc[index,"Capacity_increment"] = f"{capacity_increase}%"
    
    #moving to next row index
    index += 1
    

#printing the capacity relaxation table for the construction heursitic
print(f'''Additionally, the capacity relaxation table for Greedy Adaptive Construction Heuristic is as follows: 
      {capacity_relaxation_matrix}''')

#---------------------------------------------------------------------------------------------------------

#Optimization Model

#defining a function
def linear_programming_OM(a_ij, T, distance_matrix, demand_values, capacity_values):
    
    #creating x_ij matrix with the same format as a_ij matrix
    x_ij = pd.DataFrame(index=a_ij.index, columns=a_ij.columns)
    
    
    #filling the x_ij matrix with the corresponding pulp variables
    for i in range(len(x_ij)):
        for j in range(len(x_ij.columns)):
            x = pl.LpVariable(f"x_{x_ij.index[i]}_{x_ij.columns[j]}", cat = pl.LpBinary)
            x_ij.iloc[i,j] = x
    
    
    #Creating model
    model = pl.LpProblem("Reallocation", pl.LpMinimize)
    
    
    #Adding the Objective Function to the model
    model += pl.lpSum([demand_values.iloc[i]*distance_matrix.iloc[i,j]*x_ij.iloc[i,j] for j in range(len(capacity_values)) for i in range(len(demand_values))])
    
    
    #Adding constraints to the model
    for i in range(len(demand_values)):
        
        #one customer allocated to one facility
        model += pl.lpSum([x_ij.iloc[i,j] for j in range(0,len(capacity_values))]) == 1 
        
        
        #Variable that defines wether reallocation happened
        R = 1- pl.lpSum([a_ij.iloc[i,j]*x_ij.iloc[i,j] for j in range(len(capacity_values))])
        
        # reallocation savings should exceed reallocation cost, if reallocation is happening
        model += pl.lpSum([demand_values.iloc[i]*distance_matrix.iloc[i,j]*(a_ij.iloc[i,j] - x_ij.iloc[i,j]) 
                           for j in range(0,len(capacity_values))]) >= T*R
    
    #Total demand in each facility should not exceed its capacity
    for j in range(0,len(capacity_values)):
        model += pl.lpSum([demand_values.iloc[i]*x_ij.iloc[i,j] for i in range(0,len(demand_values))]) <= capacity_values.iloc[j] * 1000
            
    
    #solving the optimization model
    model.solve() 
    
    #printing the status
    print("Status:", pl.LpStatus[model.status])
    
    # calculating the objective function value, which is the total allocation cost.
    total_allocation_cost = pl.value(model.objective)
    
    #initializing a copy of x_ij. 
    x_ij_values = x_ij.copy()
    
    #further fill it with the values of the binary variables, which means Creating allocation matrix for the optimization solution
    for i in range(len(demand_values)):
        for j in range(len(capacity_values)):
            x_ij_values.iloc[i,j] = x_ij.iloc[i,j].varValue
            
            
    #Creating a table of customers, wherein the value is 1 if the corresponding customer was reallocated, and 0 if otherwise.
    
    reallocation_details = pd.DataFrame(index = x_ij.index, columns = ["Reallocation_happened?"])
    
    for row_index in range(len(a_ij.index)):
        if (x_ij_values.iloc[row_index,:] * a_ij.iloc[row_index,:]).sum() == 1:
            reallocation_details.iloc[row_index, 0] = 0
        else:
            reallocation_details.iloc[row_index, 0] = 1
    
    
    # Returning the allocation matrix from the optimization model, reallocation details table, as well as total allocation cost.
    return [x_ij_values, reallocation_details, total_allocation_cost]


#Applying the function for a reallocation cost set to 0. 
solution_OM = linear_programming_OM(allocation_matrix, 0, distance_v, demand_v, capacity_v)

#retreiving the total allocation cost, and printing it on screen.

print(f"The total allocation cost, or the value of objective function, after applying the Optimization Model, is {solution_OM[2]}")


#Seeing how many customers were reallocated and printing the result on the screen
print(f"Moreover, the number of customers which were reallocated are {solution_OM[1].sum()[0]}")

print("Now, we will try doing first improvement and see if the solution original solution improves.")

#----------------------------------------------------------------------------------------

#First Improvement


def first_improvement_heuristic(allocation_matrix, reallocation_cost, distance_matrix, demand_values, original_capacity_values):
    
    #number of FI iterations
    number_of_real_improvement_cycles = 0

    # Creating a list of customers
    customers = list(allocation_matrix.index)
    
    #setting up current allocation_matrix
    current_allocation_matrix = allocation_matrix.copy()
    
    #calculating remaining capacity based on current allocation
    workload = (demand_values["Demand"].sort_index().dot(current_allocation_matrix.sort_index()))/1000
    
    
    remaining_capacity_of_facilities = (original_capacity_values.iloc[:,0]).sort_index() - (workload.sort_index())
    
   
    
    while True:
        #Randomly shuffling customers
        random.shuffle(customers)
        
        # Initializing new allocation_matrix which will be updated during first improvement
        new_allocation_matrix = current_allocation_matrix.copy()
        
        # Iterating over all the customers in the solution matrix in random order
        for customer in customers:
            
            #Finding out the current allocation cost
            current_customer_allocation_det = current_allocation_matrix.loc[customer,:]
            
            #name of the currently allocated facility
            currently_allocated_facility = (current_customer_allocation_det[current_customer_allocation_det == 1]).index[0]
            
            #cost of the currently allocated facility: current allocation cost
            current_allocation_cost = demand_values.loc[customer, "Demand"] * distance_matrix.loc[customer, currently_allocated_facility]
            
            #calculating minimum allocation cost
            minimum_allocation_cost = min(distance_matrix.loc[customer,:].sort_index() * demand_values.loc[customer,"Demand"])
            
            if current_allocation_cost > minimum_allocation_cost:
                
                # Iterating over all the facilities in the solution matrix for that customer by their index
                for facility in new_allocation_matrix.columns:
                    
                    #checking the cost
                    cost = demand_values.loc[customer, "Demand"] * distance_matrix.loc[customer, facility]
                    
                    # if total cost after reallocation is less than the current allocation cost
                    if cost + reallocation_cost < current_allocation_cost:
                        
                        #if there is available space in the facility
                        if remaining_capacity_of_facilities.loc[facility] * 1000 >= demand_values.loc[customer,"Demand"]:
                            
                            #allocating the customer
                            new_allocation_matrix.loc[customer, currently_allocated_facility] = 0
                            new_allocation_matrix.loc[customer, facility] = 1
                            
                            # Adjusting the capacity of the facilities
                            remaining_capacity_of_facilities[currently_allocated_facility] += demand_values.loc[customer, "Demand"] / 1000
                            
                            remaining_capacity_of_facilities[facility] -= demand_values.loc[customer, "Demand"] / 1000
                            # Breaking out of the inner loop to move on to the next customer
                            break
                            
    
        #Stopping criteria
        if new_allocation_matrix.equals(current_allocation_matrix):
            break
        
        #if stopping criteria is not met, it means there was an improvement
        else:
            
            #set the updated allocation matrix as the current allocation matrix
            current_allocation_matrix = new_allocation_matrix.copy()
            
            # There was an improvement. Increment the number of real improvement cycles by 1
            number_of_real_improvement_cycles += 1
            
        
    #Once out of the while loop, it means the first improvement exercise is done.
    
    # Calculating the allocation cost for each customer
    new_allocation_cost = (new_allocation_matrix.sort_index().sort_index(axis = 1)*distance_matrix.sort_index().sort_index(axis = 1)).sum(axis = 1) * demand_values.iloc[:,0].sort_index()

    # Returning the updated solution matrix, allocation cost, and number of real improvement cycles it went through
    return [new_allocation_matrix, new_allocation_cost, number_of_real_improvement_cycles]


#Getting First Improvement Solution
solution_FI = first_improvement_heuristic(allocation_matrix, 0, distance_v, demand_v, capacity_v)

print(f"The total allocation cost, or the total allocation cost, after applying first improvement, is {solution_FI[1].sum()}. Moreover, the number of real improvement cycles were {solution_FI[2]}")




#-----------------------------------------------------------------------------------


#Performing sensitivity analysis by Varying the parameters like capacity and reallocation cost to see wether total allocation costs change
print("Now, we will perform sensitivity analysis, to see wether changing reallocation cost or capacity changes the total allocation costs for both FI and OM")

#Creating an empty capacity relaxation table
capacity_variation_table = pd.DataFrame(columns = ["Increment",
                                                   "Total Allocation Cost after incrementing Capacity for OM", 
                                                   "% improvement for OM", 
                                                   "Total Allocation Cost after incrementing Capacity for FI", 
                                                   "Number of FI improvement iterations",
                                                   "% improvement for FI"])

#Creating an empty reallocation cost sensitivity analysis table
reallocation_cost_variation_table = pd.DataFrame(columns = ["Increment", 
                                                            "Total Allocation Cost after incrementing Reallocation Cost (Initially set to 100k) for OM",
                                                            "% improvement for OM",
                                                            "Total Allocation Cost after incrementing Reallocation Cost (Initially set to 100k) for FI",
                                                            "Number of FI improvement iterations",
                                                            "% improvement for FI"])


#Setting reallocation cost value to 100k for reallocation_cost_variation.
reallocation_cost_v = 100000


#initializing an index
index = 0
#initializing list of increases
increases = list(range(0, 105, 10))

increases.append(float('inf'))

for increase in increases:
    
    
    if increase == float('inf'):
        #transforming capacity_v table that was imported by adding each value with increment
        capacity = capacity_v* (100 + 100000000000000)/100
        
        #transforming reallocation cost by adding it with increment
        reallocation_cost = reallocation_cost_v*(100 + 1000000000000000)/100
    
    else:
        #transforming capacity_v table that was imported by adding each value with increment
        capacity = capacity_v* (100 + increase)/100
        
        #transforming reallocation cost by adding it with increment
        reallocation_cost = reallocation_cost_v*(100 + increase)/100
    
    

    #calculating total allocation cost for each of the cases.
    

    #Total allocation cost by increasing capacity for OM
    totalcost_c_OM = linear_programming_OM(allocation_matrix, 0, distance_v, demand_v, capacity)[2]
    
    #Total allocation cost by increasing reallocation cost for OM
    totalcost_r_OM = linear_programming_OM(allocation_matrix, reallocation_cost, distance_v, demand_v, capacity_v)[2]
    
    #Total allocation cost and number of FI improvement iterations by increasing capacity for FI
    ##Applying FI
    result_FI_c = first_improvement_heuristic(allocation_matrix, 0, distance_v, demand_v, capacity)
    ##cost
    totalcost_c_FI = result_FI_c[1].sum()
    ##number of FI improvement iterations
    noofiter_c_FI = result_FI_c[2]
    

    #Total allocation cost by increasing reallocation cost for FI
    
    ##Applying FI
    result_FI_r = first_improvement_heuristic(allocation_matrix, reallocation_cost, distance_v, demand_v, capacity_v)
    
    ##cost
    totalcost_r_FI = result_FI_r[1].sum()
    
    ##Number of FI improvement iterations
    noofiter_r_FI = result_FI_r[2]

    #adding values to capacity_variation table
    capacity_variation_table.loc[index,"Increment"] = f"{increase}%"
    
    capacity_variation_table.loc[index, "Total Allocation Cost after incrementing Capacity for OM"] = totalcost_c_OM
    
    capacity_variation_table.loc[index, "Total Allocation Cost after incrementing Capacity for FI"] = totalcost_c_FI
    
    capacity_variation_table.loc[index, "Number of FI improvement iterations"] = noofiter_c_FI
    
    if index == 0:
        initial_allocation_cost_c_OM = totalcost_c_OM
        initial_allocation_cost_c_FI = totalcost_c_FI
    
    elif index != 0:
        capacity_variation_table.loc[index, "% improvement for OM"] = round(((initial_allocation_cost_c_OM - totalcost_c_OM)/initial_allocation_cost_c_OM) * 100)
        capacity_variation_table.loc[index, "% improvement for FI"] = round(((initial_allocation_cost_c_FI - totalcost_c_FI)/initial_allocation_cost_c_FI) * 100)
    
    
    #adding values to reallocation cost variation table
    reallocation_cost_variation_table.loc[index,"Increment"] = f"{increase}%"
    
    reallocation_cost_variation_table.loc[index,"Total Allocation Cost after incrementing Reallocation Cost (Initially set to 100k) for OM"] = totalcost_r_OM
    
    reallocation_cost_variation_table.loc[index,"Total Allocation Cost after incrementing Reallocation Cost (Initially set to 100k) for FI"] = totalcost_r_FI
    
    reallocation_cost_variation_table.loc[index,"Number of FI improvement iterations"] = noofiter_r_FI
    
    if index == 0:
        initial_allocation_cost_r_OM = totalcost_r_OM
        initial_allocation_cost_r_FI = totalcost_r_FI
    
    elif index != 0:
        reallocation_cost_variation_table.loc[index, "% improvement for OM"] = round(((initial_allocation_cost_r_OM - totalcost_r_OM)/initial_allocation_cost_r_OM)*100)
        reallocation_cost_variation_table.loc[index, "% improvement for FI"] = round(((initial_allocation_cost_r_FI - totalcost_r_FI)/initial_allocation_cost_r_FI) * 100)
        
    #moving to next row index
    index += 1

#Printing the sensitivity analysis result
print(f'''Below is the capacity relaxation table for both FI and OM, for the same initial allocation matrix generated by GA. Note that here, the reallocation cost was set to 0:
      
      {capacity_variation_table}''')

print(f'''Below is the sensitivity analysis table for reallocation cost, for both FI and OM, for the same initial allocation matrix generated by GA. Note that here, the reallocation cost was initially set to 100k:
      
      {reallocation_cost_variation_table}''')


#Creating visualization to present the key sensitivity analsysis result

##Creating Stepwise Line graph for Capacity relaxation table
cleaned_cr_table = capacity_variation_table.fillna(0)
x = capacity_variation_table.loc[:,"Increment"]
x.drop(0, inplace = True)
x = x.append(pd.Series([""], index = [12]))


y_OM = cleaned_cr_table.loc[:, "% improvement for OM"]
y_FI = cleaned_cr_table.loc[:, "% improvement for FI"]

plt.fill_between(x, y_OM, step="pre", alpha=0.4)
plt.fill_between(x, y_FI, step="pre", alpha=0.4)

plt.plot(x, y_OM, drawstyle="steps")
plt.plot(x, y_FI, drawstyle="steps")

plt.ylabel("% Improvement from Initial Value")
plt.xlabel("Capacity relaxation")

plt.title("Comparison of the improvement % between OM and FI for different capacity relaxations")
plt.legend(["OM", "FI"])
plt.show()




      
        

        



 
    
    
    







            
            
            
    
    
    


