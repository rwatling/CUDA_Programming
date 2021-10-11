setwd("/home/rwatling/Academics/MTU/masters/Programming/CUDA_Programming/GPU_Array_Matching")

#packages required
packages = c("gbutils", "ggplot2")

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

file1 <- "data/change_t_a4.csv"
file2 <- "data/change_t_a8.csv"
file3 <- "data/change_t_a16.csv"
file3 <- "data/change_t_a20.csv"
file4 <- "data/change_num_arr_small.csv"
file5 <- "data/change_num_arr_avg.csv"
file6 <- "data/change_num_arr_lg.csv"
all_files <- c(file1, file2, file3, file4, file5, file6)

### Change Array Size: file1-4###
allPerfDf <- read.csv(all_files[3])

# For speedup
shuffleChangeDf <-  allPerfDf[which(allPerfDf$type == 1),]
shareChangeDf <-  allPerfDf[which(allPerfDf$type == 0),]

# For plots
allPerfDf[which(allPerfDf$type == 1),]$type = "shuffle"
allPerfDf[which(allPerfDf$type == 0),]$type = "shared"

# Plot
plot.new()
ggplot(data = allPerfDf) +
  geom_smooth(mapping = aes(x = array_size, y = time, group = type, color=type)) +
  # geom_point(mapping = aes(x = array_size, y = time, color=type)) +
  ggtitle("Time vs. Number of Threads") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme_minimal()

plot.new()
ggplot(data = allPerfDf) +
  geom_smooth(mapping = aes(x = array_size, y = time, group = "shared", color=type)) +
  ggtitle("Time vs. Array Size (Shared Only)") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme_minimal()

speedup = mean(sharedChangeDf$time) / mean(shuffleChangeDf$time)
speedup
