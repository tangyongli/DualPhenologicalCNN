import pandas as pd 
 
# reading csv file from url  
data = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv") 
 
# creating position and label variables
position = 2
label = 'Name'
     
# calling .at[] method
output = data.at[position, label]
 
# display
print(output)