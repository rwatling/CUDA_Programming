setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/GPU_Array_Matching_v2/analysis")

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
file3 <- "data/change_t_a12.csv"
file4 <- "data/change_t_a16.csv"
all_files <- c(file1, file2, file3, file4)

### Change Array Size: file1-4###
allPerfDf <- read.csv(all_files[3]) # Change this

# For speedup
shuffleChangeDf <-  allPerfDf[which(allPerfDf$type == 1),]
shareChangeDf <-  allPerfDf[which(allPerfDf$type == 0),]

# For plots
allPerfDf[which(allPerfDf$type == 1),]$type = "shuffle"
allPerfDf[which(allPerfDf$type == 0),]$type = "shared"

# Speedup information
speedup = mean(shareChangeDf$time) / mean(shuffleChangeDf$time)
speedup_text = paste("Average speedup ", speedup)

# Plot
png(file=gsub(".csv", ".png", all_files[3])) # Change this
plot.new()
ggplot(data = allPerfDf) +
  geom_smooth(mapping = aes(x = number_of_arrays, y = time, group = type, color=type)) +
  ggtitle("Time vs. Number of Threads") +
  xlab(paste("Number of Threads\n", speedup_text)) +
  ylab("Time (ms)") +
  theme_minimal()
dev.off()
