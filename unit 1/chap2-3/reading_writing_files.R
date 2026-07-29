#sample csv files
# https://support.staffbase.com/hc/en-us/articles/360007108391-CSV-File-Examples

 
link.bb <- "https://support.staffbase.com/hc/en-us/article_attachments/360009197031/username.csv"
#load(link.bb)
BrainBody_t <- read.table(file = link.bb, header = TRUE, sep = ";", stringsAsFactors = FALSE)
BrainBody_c <- read.csv(link.bb) #,  header = TRUE, sep = ";", stringsAsFactors = FALSE)
BrainBody_c2<-read.csv2(link.bb)
str(BrainBody_t)
View(BrainBody_t)
View(BrainBody_c)
View(BrainBody_c2)
getwd()
setwd("/home/anil/Documents/data science using r/prog")
list.files()
------------------------------------------
print(1)

unique(airquality$Month)
duplicated(airquality)

?write.table
?write.csv

library(readxl)
data<-read_excel("/home/anil/Downloads/Subject Attendance Report GNITS ERP.xlsx")
print(data)


a=1:70
tail(a)
tail(a,n=-55)
head(a,n=-55)
