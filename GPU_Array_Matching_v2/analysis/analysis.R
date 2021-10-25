setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/GPU_Array_Matching_v2/analysis")

#packages required
packages = c("gbutils", "ggplot2", "ggthemes", "grid", "gridExtra", "gtable")

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

# Filter data
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

# Combine 1-5 for line graph
combined <- rbind(allPerfDf1, allPerfDf2, allPerfDf3, allPerfDf4, allPerfDf5)

# Plot
png("shared_vs_shuffle_line.png")
plot.new()
ggplot() +
  geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size))) +
  ggtitle("Time vs. Number of Threads") +
  ylab("Time (ms)") +
  theme_minimal() +
  scale_color_brewer("Type | Arrays Size", palette = "Paired")
dev.off()

### Plot grouped bar graphs ###
png("shared_vs_shufle_bar.png")
plot.new()
ggplot(combined, aes(fill=type, y=time, x=array_size)) +
  geom_bar(position="dodge",  width=2, stat="identity") +
  ylab("Time (ms)") +
  xlab("Array Size") +
  guides(fill=guide_legend("Type"))+
  theme_minimal()
dev.off()

### Speedup information ###
shuffleChangeDf1 <-  allPerfDf1[which(allPerfDf1$type == "shuffle"),]
shareChangeDf1 <-  allPerfDf1[which(allPerfDf1$type == "shared"),]

shuffleChangeDf2 <-  allPerfDf2[which(allPerfDf2$type == "shuffle"),]
shareChangeDf2 <-  allPerfDf2[which(allPerfDf2$type == "shared"),]

shuffleChangeDf3 <-  allPerfDf3[which(allPerfDf3$type == "shuffle"),]
shareChangeDf3 <-  allPerfDf3[which(allPerfDf3$type == "shared"),]

shuffleChangeDf4 <-  allPerfDf1[which(allPerfDf4$type == "shuffle"),]
shareChangeDf4 <-  allPerfDf1[which(allPerfDf4$type == "shared"),]

shuffleChangeDf5 <-  allPerfDf1[which(allPerfDf5$type == "shuffle"),]
shareChangeDf5 <-  allPerfDf1[which(allPerfDf5$type == "shared"),]

# Mean
meanTimeDf1 <- mean(shareChangeDf1$time) / mean(shuffleChangeDf1$time)
meanTimeDf2 <- mean(shareChangeDf2$time) / mean(shuffleChangeDf2$time)
meanTimeDf3 <- mean(shareChangeDf3$time) / mean(shuffleChangeDf3$time)
meanTimeDf4 <- mean(shareChangeDf4$time) / mean(shuffleChangeDf4$time)
meanTimeDf5 <- mean(shareChangeDf5$time) / mean(shuffleChangeDf5$time)

# Median
medianTimeDf1 <- median(shareChangeDf1$time) / median(shuffleChangeDf1$time)
medianTimeDf2 <- median(shareChangeDf2$time) / median(shuffleChangeDf2$time)
medianTimeDf3 <- median(shareChangeDf3$time) / median(shuffleChangeDf3$time)
medianTimeDf4 <- median(shareChangeDf4$time) / median(shuffleChangeDf4$time)
medianTimeDf5 <- median(shareChangeDf5$time) / median(shuffleChangeDf5$time)

# Assemble data frame
mean_speedup <- c(meanTimeDf1, meanTimeDf2, meanTimeDf3, meanTimeDf4, meanTimeDf5)
median_speedup <- c(medianTimeDf1, medianTimeDf2, medianTimeDf3, medianTimeDf4, medianTimeDf5)
size <- c(4, 8, 12, 16, 24)
speedupDf <- data.frame(size, mean_speedup, median_speedup)
speedupDf <- format(speedupDf, digits = 3)
colnames(speedupDf) <- c("Array Size", "Mean Speedup", "Median Speedup")
rownames(speedupDf) <- NULL

png("shuffle_shared_table.png")
plot.new()
g <- tableGrob(speedupDf, rows = NULL, theme = ttheme_minimal())
g <- gtable_add_grob(g,
                     grobs = rectGrob(gp = gpar(fill = NA, lwd = 2)),
                     t = 2, b = nrow(g), l = 1, r = ncol(g))
g <- gtable_add_grob(g,
                     grobs = rectGrob(gp = gpar(fill = NA, lwd = 2)),
                     t = 1, l = 1, r = ncol(g))
grid.draw(g)
dev.off()

### Register Information ###
array_sizes <- c(4, 8, 12, 16, 24)
static_shuffle_regs <- c(31, 40, 48, 54, 80)
static_share_regs <- c(22, 32, 48, 64, 56)
new_shuffle_regs <- c(29, 29, 30, 30, 30)
new_share_regs <- c(30, 30, 27, 27, 27)
shuffleRegsDf <- data.frame(array_sizes, static_shuffle_regs, new_shuffle_regs)
sharedRegsDf <- data.frame(array_sizes, static_share_regs, new_share_regs)

colnames(shuffleRegsDf) <- c("Array Size", "Static Shuffle Regs", "New Keyword Shuffle Regs")
colnames(sharedRegsDf) <- c("Array Size", "Static Shm Regs", "New Keyword Shm Regs")

png("shuffle_regs.png")
plot.new()
g <- tableGrob(shuffleRegsDf, rows = NULL, theme = ttheme_minimal())
g <- gtable_add_grob(g,
                     grobs = rectGrob(gp = gpar(fill = NA, lwd = 2)),
                     t = 2, b = nrow(g), l = 1, r = ncol(g))
g <- gtable_add_grob(g,
                     grobs = rectGrob(gp = gpar(fill = NA, lwd = 2)),
                     t = 1, l = 1, r = ncol(g))
grid.draw(g)
dev.off()

png("shared_regs.png")
plot.new()
g <- tableGrob(sharedRegsDf, rows = NULL, theme = ttheme_minimal())
g <- gtable_add_grob(g,
                     grobs = rectGrob(gp = gpar(fill = NA, lwd = 2)),
                     t = 2, b = nrow(g), l = 1, r = ncol(g))
g <- gtable_add_grob(g,
                     grobs = rectGrob(gp = gpar(fill = NA, lwd = 2)),
                     t = 1, l = 1, r = ncol(g))
grid.draw(g)
dev.off()
