setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/GPU_Array_Matching_v2/analysis")

#packages required
packages = c("gbutils", "ggplot2", "ggthemes", "grid", "gridExtra", "gtable", "remotes")


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


file1 <- "data/fall2021-mid/change_t_arr4.csv"
file2 <- "data/fall2021-mid/change_t_arr8.csv"
file3 <- "data/fall2021-mid/change_t_arr12.csv"
file4 <- "data/fall2021-mid/change_t_arr16.csv"
file5 <- "data/fall2021-mid/change_t_arr24.csv"

arr4Df <- read.csv(file1)
arr8Df <- read.csv(file2)
arr12Df <- read.csv(file3)
arr16Df <- read.csv(file4)
arr24Df <- read.csv(file5)

### Communication Analysis ###
# Filter data
temp1 <- arr4Df[which(arr4Df$type == "Nested Shfl"),]
temp2 <- arr4Df[which(arr4Df$type == "Nested Shm"),]
naive4 <- rbind(temp1, temp2)

temp1 <- arr8Df[which(arr8Df$type == "Nested Shfl"),]
temp2 <- arr8Df[which(arr8Df$type == "Nested Shm"),]
naive8 <- rbind(temp1, temp2)

temp1 <- arr12Df[which(arr12Df$type == "Nested Shfl"),]
temp2 <- arr12Df[which(arr12Df$type == "Nested Shm"),]
naive12 <- rbind(temp1, temp2)

temp1 <- arr16Df[which(arr16Df$type == "Nested Shfl"),]
temp2 <- arr16Df[which(arr16Df$type == "Nested Shm"),]
naive16 <- rbind(temp1, temp2)

temp1 <- arr24Df[which(arr24Df$type == "Nested Shfl"),]
temp2 <- arr24Df[which(arr24Df$type == "Nested Shm"),]
naive24 <- rbind(temp1, temp2)

# Combine
combined <- rbind(naive4, naive8, naive12, naive16, naive24)
combined <- combined[which(combined$number_of_arrays == 1024),]
combined[, c(3)] <- sapply(combined[,c(3)], as.character)
combined <- aggregate(combined[, 4], list(combined$type, combined$array_size), mean)

# Plot
png("shared_vs_shuffle_bar.png")
plot.new()
ggplot(combined, aes(x=Group.2, y=x, fill=Group.1, label=Group.2)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = TRUE, color="black") +
  scale_x_discrete(limits=c("4", "8", "12", "16", "24")) +
  scale_fill_manual("Communication Type", labels=c("Shuffle", "Shared Mem"), values=c("black", "white")) +
  ggtitle("Shared vs Shuffle Communication (T=1024)") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme(legend.position=c(0.3, 0.8),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

# Speedup at 1024 Threads
speedupDf = naive24[which(naive24$number_of_arrays == 1024),]
speedup = mean(speedupDf[which(speedupDf$type == "Nested Shm"),]$time)/mean(speedupDf[which(speedupDf$type == "Nested Shfl"),]$time)
print(speedup - 1.0)

### Match Analysis ###
## Unroll Shfl vs Nested Shfl
# Filter data
temp1 <- arr4Df[which(arr4Df$type == "Nested Shfl"),]
temp2 <- arr4Df[which(arr4Df$type == "Shfl Unroll"),]
unroll4 <- rbind(temp1, temp2)

temp1 <- arr8Df[which(arr8Df$type == "Nested Shfl"),]
temp2 <- arr8Df[which(arr8Df$type == "Shfl Unroll"),]
unroll8 <- rbind(temp1, temp2)

temp1 <- arr12Df[which(arr12Df$type == "Nested Shfl"),]
temp2 <- arr12Df[which(arr12Df$type == "Shfl Unroll"),]
unroll12 <- rbind(temp1, temp2)

temp1 <- arr16Df[which(arr16Df$type == "Nested Shfl"),]
temp2 <- arr16Df[which(arr16Df$type == "Shfl Unroll"),]
unroll16 <- rbind(temp1, temp2)

temp1 <- arr24Df[which(arr24Df$type == "Nested Shfl"),]
temp2 <- arr24Df[which(arr24Df$type == "Shfl Unroll"),]
unroll24 <- rbind(temp1, temp2)

# Combine 1-5 for line graph
combined <- rbind(unroll4, unroll8, unroll12, unroll16, unroll24)

# Combine
combined <- rbind(unroll4, unroll8, unroll12, unroll16, unroll24)
combined <- combined[which(combined$number_of_arrays == 1024),]
combined[, c(3)] <- sapply(combined[,c(3)], as.character)
combined <- aggregate(combined[, 4], list(combined$type, combined$array_size), mean)

# Plot
png("nested_v_unroll_bar.png")
plot.new()
ggplot(combined, aes(x=Group.2, y=x, fill=Group.1, label=Group.2)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = TRUE, color="black") +
  scale_x_discrete(limits=c("4", "8", "12", "16", "24")) +
  scale_fill_manual("Communication Type", labels=c("Nested Match", "Unrolled Match"), values=c("black", "white")) +
  ggtitle("Nested Loop Match vs Unroll Loop Match (T=1024)") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme(legend.position=c(0.3, 0.8),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

# Plot
#png("nested_shfl_v_shfl_unroll.png")
#plot.new()
#ggplot() +
  #geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  #ggtitle("Time vs. Number of Threads") +
  #ylab("Time (ms)") +
  #xlab("Number of Threads") +
  #theme_minimal() +
  #scale_color_brewer("Type | Arrays Size", palette = "Paired")
#dev.off()

# Speedup at 1024 Threads
speedupDf = unroll24[which(unroll24$number_of_arrays == 1024),]
speedup = mean(speedupDf[which(speedupDf$type == "Nested Shfl"),]$time)/mean(speedupDf[which(speedupDf$type == "Shfl Unroll"),]$time)
print(speedup - 1.0)

## Hash vs Shfl
# Filter data
temp1 <- arr4Df[which(arr4Df$type == "Nested Shfl"),]
temp2 <- arr4Df[which(arr4Df$type == "Shfl Hash"),]
hash4 <- rbind(temp1, temp2)

temp1 <- arr8Df[which(arr8Df$type == "Nested Shfl"),]
temp2 <- arr8Df[which(arr8Df$type == "Shfl Hash"),]
hash8 <- rbind(temp1, temp2)

temp1 <- arr12Df[which(arr12Df$type == "Nested Shfl"),]
temp2 <- arr12Df[which(arr12Df$type == "Shfl Hash"),]
hash12 <- rbind(temp1, temp2)

temp1 <- arr16Df[which(arr16Df$type == "Nested Shfl"),]
temp2 <- arr16Df[which(arr16Df$type == "Shfl Hash"),]
hash16 <- rbind(temp1, temp2)

temp1 <- arr24Df[which(arr24Df$type == "Nested Shfl"),]
temp2 <- arr24Df[which(arr24Df$type == "Shfl Hash"),]
hash24 <- rbind(temp1, temp2)

# Combine 1-5 for line graph
combined <- rbind(hash4, hash8, hash12, hash16, hash24)

# Combine
combined <- rbind(hash4, hash8, hash12, hash16, hash24)
combined <- combined[which(combined$number_of_arrays == 1024),]
combined[, c(3)] <- sapply(combined[,c(3)], as.character)
combined <- aggregate(combined[, 4], list(combined$type, combined$array_size), mean)

# Plot
png("nested_v_hash_bar.png")
plot.new()
ggplot(combined, aes(x=Group.2, y=x, fill=Group.1, label=Group.2)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = TRUE, color="black") +
  scale_x_discrete(limits=c("4", "8", "12", "16", "24")) +
  # theme_minimal() +
  scale_fill_manual("Communication Type", labels=c("Nested", "Hash Table"), values=c("black", "white")) +
  ggtitle("Nested Loop vs Hash Table Communication (T=1024)") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme(legend.position=c(0.3, 0.8),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

# Plot
#png("nested_shfl_v_shfl_hash.png")
#plot.new()
#ggplot() +
  #geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  #ggtitle("Time vs. Number of Threads") +
  #ylab("Time (ms)") +
  #xlab("Number of Threads") +
  #theme_minimal() +
  #scale_color_brewer("Type | Arrays Size", palette = "Paired")
#dev.off()

# Speedup at 1024 Threads
speedupDf = hash24[which(hash24$number_of_arrays == 1024),]
speedup = mean(speedupDf[which(speedupDf$type == "Nested Shfl"),]$time)/mean(speedupDf[which(speedupDf$type == "Shfl Hash"),]$time)
print(speedup - 1.0)

## Sort and Search vs Shfl
# Filter data
temp1 <- arr4Df[which(arr4Df$type == "Nested Shfl"),]
temp2 <- arr4Df[which(arr4Df$type == "Shfl Sort Search"),]
ss4 <- rbind(temp1, temp2)

temp1 <- arr8Df[which(arr8Df$type == "Nested Shfl"),]
temp2 <- arr8Df[which(arr8Df$type == "Shfl Sort Search"),]
ss8 <- rbind(temp1, temp2)

temp1 <- arr12Df[which(arr12Df$type == "Nested Shfl"),]
temp2 <- arr12Df[which(arr12Df$type == "Shfl Sort Search"),]
ss12 <- rbind(temp1, temp2)

temp1 <- arr16Df[which(arr16Df$type == "Nested Shfl"),]
temp2 <- arr16Df[which(arr16Df$type == "Shfl Sort Search"),]
ss16 <- rbind(temp1, temp2)

temp1 <- arr24Df[which(arr24Df$type == "Nested Shfl"),]
temp2 <- arr24Df[which(arr24Df$type == "Shfl Sort Search"),]
ss24 <- rbind(temp1, temp2)

# Combine
combined <- rbind(ss4,ss8,ss12,ss16,ss24)
combined <- combined[which(combined$number_of_arrays == 1024),]
combined[, c(3)] <- sapply(combined[,c(3)], as.character)
combined <- aggregate(combined[, 4], list(combined$type, combined$array_size), mean)

# Plot
png("nested_vs_sorted_bar.png")
plot.new()
ggplot(combined, aes(x=Group.2, y=x, fill=Group.1, label=Group.2)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = TRUE, color="black") +
  scale_x_discrete(limits=c("4", "8", "12", "16", "24")) +
  #theme_minimal() +
  scale_fill_manual("Communication Type", labels=c("Nested", "Sorted"), values=c("black", "white")) +
  ggtitle("Nested vs Sorted Match (T=1024)") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme(legend.position=c(0.3, 0.8),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()


# Plot
#png("nested_shfl_v_sort_search.png")
#plot.new()
#ggplot() +
  #geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  #ggtitle("Time vs. Number of Threads") +
  #ylab("Time (ms)") +
  #xlab("Number of Threads") +
  #theme_minimal() +
  #scale_color_brewer("Type | Arrays Size", palette = "Paired")
#dev.off()

# Speedup at 1024 Threads
speedupDf = ss24[which(ss24$number_of_arrays == 1024),]
speedup = mean(speedupDf[which(speedupDf$type == "Nested Shfl"),]$time)/mean(speedupDf[which(speedupDf$type == "Shfl Sort Search"),]$time)
print(speedup - 1.0)



### HASH ANALYSIS ###
sm_hash_a4 <- read.csv("data/original_hash/sm_hash_array_size4.csv")
sm_hash_a8 <- read.csv("data/original_hash/sm_hash_array_size8.csv")
sm_hash_a12 <- read.csv("data/original_hash/sm_hash_array_size12.csv")
sm_hash_a16 <- read.csv("data/original_hash/sm_hash_array_size16.csv")
sm_hash_a24 <- read.csv("data/original_hash/sm_hash_array_size24.csv")

lg_hash_a4 <- read.csv("data/original_hash/lg_hash_array_size4.csv")
lg_hash_a8 <- read.csv("data/original_hash/lg_hash_array_size8.csv")
lg_hash_a12 <- read.csv("data/original_hash/lg_hash_array_size12.csv")
lg_hash_a16 <- read.csv("data/original_hash/lg_hash_array_size16.csv")
lg_hash_a24 <- read.csv("data/original_hash/lg_hash_array_size24.csv")

### Base shuffle vs small hash ###
# Filter data
temp1 <- sm_hash_a4[which(sm_hash_a4$type == 2),]
temp2 <- sm_hash_a4[which(sm_hash_a4$type == 4),]
sm_hash_v_shfl4 <- rbind(temp1, temp2)
sm_hash_v_shfl4[which(sm_hash_v_shfl4 $type == 2),]$type = "shfl"
sm_hash_v_shfl4 [which(sm_hash_v_shfl4 $type == 4),]$type = "hash"

temp1 <- sm_hash_a8[which(sm_hash_a8$type == 2),]
temp2 <- sm_hash_a8[which(sm_hash_a8$type == 4),]
sm_hash_v_shfl8 <- rbind(temp1, temp2)
sm_hash_v_shfl8[which(sm_hash_v_shfl8$type == 2),]$type = "shfl"
sm_hash_v_shfl8 [which(sm_hash_v_shfl8$type == 4),]$type = "hash"

temp1 <- sm_hash_a12[which(sm_hash_a12$type == 2),]
temp2 <- sm_hash_a12[which(sm_hash_a12$type == 4),]
sm_hash_v_shfl12 <- rbind(temp1, temp2)
sm_hash_v_shfl12[which(sm_hash_v_shfl12$type == 2),]$type = "shfl"
sm_hash_v_shfl12[which(sm_hash_v_shfl12$type == 4),]$type = "hash"

temp1 <- sm_hash_a16[which(sm_hash_a16$type == 2),]
temp2 <- sm_hash_a16[which(sm_hash_a16$type == 4),]
sm_hash_v_shfl16 <- rbind(temp1, temp2)
sm_hash_v_shfl16[which(sm_hash_v_shfl16$type == 2),]$type = "shfl"
sm_hash_v_shfl16[which(sm_hash_v_shfl16$type == 4),]$type = "hash"

temp1 <- sm_hash_a24[which(sm_hash_a24$type == 2),]
temp2 <- sm_hash_a24[which(sm_hash_a24$type == 4),]
sm_hash_v_shfl24 <- rbind(temp1, temp2)
sm_hash_v_shfl24[which(sm_hash_v_shfl24$type == 2),]$type = "shfl"
sm_hash_v_shfl24 [which(sm_hash_v_shfl24$type == 4),]$type = "hash"

combined <- rbind(sm_hash_v_shfl4, sm_hash_v_shfl8, sm_hash_v_shfl12, sm_hash_v_shfl16, sm_hash_v_shfl24)

# Small hash vs large hash #
temp1 <- sm_hash_a4[which(sm_hash_a4$type == 4),]
temp2 <- lg_hash_a4[which(lg_hash_a4$type == 4),]
temp1[which(temp1$type == 4),]$type = "small"
temp2[which(temp2$type == 4),]$type = "large"
sm_hash_v_lg_hash4 <- rbind(temp1, temp2)

temp1 <- sm_hash_a8[which(sm_hash_a8$type == 4),]
temp2 <- lg_hash_a8[which(lg_hash_a8$type == 4),]
temp1[which(temp1$type == 4),]$type = "small"
temp2[which(temp2$type == 4),]$type = "large"
sm_hash_v_lg_hash8 <- rbind(temp1, temp2)

temp1 <- sm_hash_a16[which(sm_hash_a16$type == 4),]
temp2 <- lg_hash_a16[which(lg_hash_a16$type == 4),]
temp1[which(temp1$type == 4),]$type = "small"
temp2[which(temp2$type == 4),]$type = "large"
sm_hash_v_lg_hash16 <- rbind(temp1, temp2)

temp1 <- sm_hash_a12[which(sm_hash_a12$type == 4),]
temp2 <- lg_hash_a12[which(lg_hash_a12$type == 4),]
temp1[which(temp1$type == 4),]$type = "small"
temp2[which(temp2$type == 4),]$type = "large"
sm_hash_v_lg_hash12 <- rbind(temp1, temp2)

temp1 <- sm_hash_a24[which(sm_hash_a24$type == 4),]
temp2 <- lg_hash_a24[which(lg_hash_a24$type == 4),]
temp1[which(temp1$type == 4),]$type = "small"
temp2[which(temp2$type == 4),]$type = "large"
sm_hash_v_lg_hash24 <- rbind(temp1, temp2)

# Combined
combined <- rbind(sm_hash_v_lg_hash4, sm_hash_v_lg_hash8, sm_hash_v_lg_hash12, sm_hash_v_lg_hash16, sm_hash_v_lg_hash24)
combined <- combined[which(combined$number_of_arrays == 1024),]
combined[, c(3)] <- sapply(combined[,c(3)], as.character)
combined <- aggregate(combined[, 4], list(combined$type, combined$array_size), mean)

# Plot
png("small_vs_large_hash.png")
plot.new()
ggplot(combined, aes(x=Group.2, y=x, fill=Group.1, label=Group.2)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = TRUE, color="black") +
  scale_x_discrete(limits=c("4", "8", "12", "16", "24")) +
  #theme_minimal() +
  scale_fill_manual("Hash Type", labels=c("Large", "Small"), values=c("black", "white")) +
  ggtitle("Relative Hash Size Comparison (T=1024)") +
  xlab("Array Size") +
  ylab("Time (ms)") +
  theme(legend.position=c(0.3, 0.8),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

