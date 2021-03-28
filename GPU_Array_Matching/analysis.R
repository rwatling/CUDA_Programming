setwd("/home/rwatling/Academics/MTU/masters/Programming/CUDA_Programming/GPU_Array_Matching")

#packages required
packages = c("gbutils", "ggplot2", "gghighlig")

## Now load or install&load all
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

file1 <- "data/change_arr_size_small.csv"
file2 <- "data/change_arr_size_avg.csv"
file3 <- "data/change_arr_size_lg.csv"
file4 <- "data/change_num_arr_small.csv"
file5 <- "data/change_num_arr_avg.csv"
file6 <- "data/change_num_arr_lg.csv"
all_files <- c(file1, file2, file3, file4, file5, file6)

### Change Array Size: file1-3###
allPerfDf <- read.csv(all_files[2])
allPerfDf <- allPerfDf[which(allPerfDf$time < 10000000),]

# For speedup
sharedChangeDf <-  allPerfDf[which(allPerfDf$type == 1),]
globalChangeDf <-  allPerfDf[which(allPerfDf$type == 0),]

# For plots
allPerfDf[which(allPerfDf$type == 1),]$type = "shared"
allPerfDf[which(allPerfDf$type == 0),]$type = "global"

# Plot 
plot.new()
ggplot(data = allPerfDf) +
  geom_smooth(mapping = aes(x = array_size, y = time, group = type, color=type)) +
  geom_point(mapping = aes(x = array_size, y = time, color=type)) +
  ggtitle("Time vs. Array Size") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme_minimal()

speedup = mean(globalChangeDf$time) / mean(sharedChangeDf$time)
speedup

### Change Array Number: file4-6###
allPerfDf <- read.csv(all_files[4])
allPerfDf <- allPerfDf[which(allPerfDf$time < 10000000),]

# For speedup
sharedChangeDf <-  allPerfDf[which(allPerfDf$type == 1),]
globalChangeDf <-  allPerfDf[which(allPerfDf$type == 0),]

# For plots
allPerfDf[which(allPerfDf$type == 1),]$type = "shared"
allPerfDf[which(allPerfDf$type == 0),]$type = "global"

# Plot 
plot.new()
ggplot(data = allPerfDf) +
  geom_smooth(mapping = aes(x = number_of_arrays, y = time, group = type, color=type)) +
  geom_point(shape = 20, mapping = aes(x = number_of_arrays, y = time, color=type)) +
  ggtitle("Time vs. Array Number") +
  xlab("Array Number") +
  ylab("Time (ms)") +
  theme_minimal()

speedup = mean(globalChangeDf$time) / mean(sharedChangeDf$time)
speedup
