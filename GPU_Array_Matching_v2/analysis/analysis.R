setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/GPU_Array_Matching_v2/analysis")

#packages required
packages = c("gbutils", "ggplot2", "ggthemes")

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
file5 <- "data/change_t_a24.csv"
all_files <- c(file1, file2, file3, file4, file5)

### Change Number of Threads ###
allPerfDf1 <- read.csv(all_files[1])
allPerfDf2 <- read.csv(all_files[2])
allPerfDf3 <- read.csv(all_files[3])
allPerfDf4 <- read.csv(all_files[4])
allPerfDf5 <- read.csv(all_files[5])

# For plots
allPerfDf1[which(allPerfDf1$type == 1),]$type = "shuffle"
allPerfDf1[which(allPerfDf1$type == 0),]$type = "shared"

allPerfDf2[which(allPerfDf2$type == 1),]$type = "shuffle"
allPerfDf2[which(allPerfDf2$type == 0),]$type = "shared"

allPerfDf3[which(allPerfDf3$type == 1),]$type = "shuffle"
allPerfDf3[which(allPerfDf3$type == 0),]$type = "shared"

allPerfDf4[which(allPerfDf4$type == 1),]$type = "shuffle"
allPerfDf4[which(allPerfDf4$type == 0),]$type = "shared"

allPerfDf5[which(allPerfDf5$type == 1),]$type = "shuffle"
allPerfDf5[which(allPerfDf5$type == 0),]$type = "shared"

combined <- rbind(allPerfDf1, allPerfDf2, allPerfDf3, allPerfDf4, allPerfDf5)

# Plot
#png(file=gsub(".csv", ".png", all_files[file_num])) # Change this
plot.new()
ggplot() +
  geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size))) +
  ggtitle("Time vs. Number of Threads") +
  ylab("Time (ms)") +
  theme_minimal() +
  scale_color_colorblind()
#dev.off()

scale_color_

### Plot speedup bar graphs ###
# For speedup
shuffleChangeDf1 <-  allPerfDf[which(allPerfDf$type == 1),]
shareChangeDf1 <-  allPerfDf[which(allPerfDf$type == 0),]

# Speedup information
speedup = mean(shareChangeDf$time) / mean(shuffleChangeDf$time)
speedup_text = paste("Average speedup ", speedup)

# create a dataset
specie <- c(rep("sorgho" , 3) , rep("poacee" , 3) , rep("banana" , 3) , rep("triticum" , 3) )
condition <- rep(c("normal" , "stress" , "Nitrogen") , 4)
value <- abs(rnorm(12 , 0 , 15))
data <- data.frame(specie,condition,value)

# Grouped
ggplot(data, aes(fill=condition, y=value, x=specie)) + 
  geom_bar(position="dodge", stat="identity")