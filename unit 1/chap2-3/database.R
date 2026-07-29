# reading database
#library(RODBC)
library(dplyr)

library(DBI)
library(RSQLite)

# Creates a new SQLite database (if it doesn't exist)
con <- dbConnect(RSQLite::SQLite(), "new_database.db")

# Create a table
dbExecute(con, "CREATE TABLE users (id INTEGER, name TEXT)")

# Insert data
dbExecute(con, "INSERT INTO users VALUES (1, 'Alice')")
dbExecute(con, "INSERT INTO users VALUES (2, 'Bob')")

# Query data
users_data <- dbGetQuery(con, "SELECT * FROM users")
print(users_data)

# Close the connection
dbDisconnect(con)
#-----------------------------------------------
  
#install.packages("dplyr")
#install.packages("dbplyr")

library(dplyr)
library(DBI)
library(RSQLite)

# Connect to SQLite
con <- dbConnect(RSQLite::SQLite(), "example.db")

# Copy a data frame into SQLite
mtcars_db <- copy_to(con, mtcars, "mtcars_table", temporary = FALSE)

# Query using dplyr syntax
results <- tbl(con, "mtcars_table") %>%
  filter(mpg > 20) %>%
  select(mpg, cyl, hp) %>%
  collect()
results<-results %>% collect()
print(results)

# Close connection
dbDisconnect(con)  
