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

file1 <- "data/hash/sm_hash_array_size4.csv"
file2 <- "data/hash/sm_hash_array_size8.csv"
file3 <- "data/hash/sm_hash_array_size12.csv"
file4 <- "data/hash/sm_hash_array_size16.csv"
file5 <- "data/hash/sm_hash_array_size24.csv"

### Communication Analysis ###
naiveArr4Df <- read.csv(file1)
naiveArr8Df <- read.csv(file2)
naiveArr12Df <- read.csv(file3)
naiveArr16Df <- read.csv(file4)
naiveArr24Df <- read.csv(file5)

# Filter data
temp1 <- naiveArr4Df[which(naiveArr4Df$type == 2),]
temp2 <- naiveArr4Df[which(naiveArr4Df$type == 1),]
naive4 <- rbind(temp1, temp2)

temp1 <- naiveArr8Df[which(naiveArr8Df$type == 2),]
temp2 <- naiveArr8Df[which(naiveArr8Df$type == 1),]
naive8 <- rbind(temp1, temp2)

temp1 <- naiveArr12Df[which(naiveArr12Df$type == 2),]
temp2 <- naiveArr12Df[which(naiveArr12Df$type == 1),]
naive12 <- rbind(temp1, temp2)

temp1 <- naiveArr16Df[which(naiveArr16Df$type == 2),]
temp2 <- naiveArr16Df[which(naiveArr16Df$type == 1),]
naive16 <- rbind(temp1, temp2)

temp1 <- naiveArr24Df[which(naiveArr24Df$type == 2),]
temp2 <- naiveArr24Df[which(naiveArr24Df$type == 1),]
naive24 <- rbind(temp1, temp2)

# Combine 1-5 for line graph
combined <- rbind(naive4, naive8, naive12, naive16, naive24)

# Plot
png("shared_vs_shuffle_line.png")
plot.new()
ggplot() +
  geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  ggtitle("Time vs. Number of Threads") +
  ylab("Time (ms)") +
  xlab("Number of Arrays") +
  theme_minimal() +
  scale_color_brewer("Type | Arrays Size", palette = "Paired")
dev.off()

# Speedup at 1024 Threads

### Match Analysis ###
## Nested loop vs Shfl
# Line graph
## Unrolled loop vs Shfl
# Line graph
## Sort and Search vs Shfl
# Line graph


### HASH ANALYSIS ###
sm_hash_a4 <- read.csv("data/hash/sm_hash_array_size4.csv")
sm_hash_a8 <- read.csv("data/hash/sm_hash_array_size8.csv")
sm_hash_a12 <- read.csv("data/hash/sm_hash_array_size12.csv")
sm_hash_a16 <- read.csv("data/hash/sm_hash_array_size16.csv")
sm_hash_a24 <- read.csv("data/hash/sm_hash_array_size24.csv")

lg_hash_a4 <- read.csv("data/hash/lg_hash_array_size4.csv")
lg_hash_a8 <- read.csv("data/hash/lg_hash_array_size8.csv")
lg_hash_a12 <- read.csv("data/hash/lg_hash_array_size12.csv")
lg_hash_a16 <- read.csv("data/hash/lg_hash_array_size16.csv")
lg_hash_a24 <- read.csv("data/hash/lg_hash_array_size24.csv")

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

# Plot
png("small_hash_line.png")
plot.new()
ggplot() +
  geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  ggtitle("Time vs. Number of Threads") +
  ylab("Time (ms)") +
  xlab("Number of Arrays") +
  theme_minimal() +
  scale_color_brewer("Type | Arrays Size", palette = "Paired")
dev.off()

# Base shuffle vs large hash #
temp1 <- lg_hash_a4[which(lg_hash_a4$type == 2),]
temp2 <- lg_hash_a4[which(lg_hash_a4$type == 4),]
lg_hash_v_shfl4 <- rbind(temp1, temp2)
lg_hash_v_shfl4[which(lg_hash_v_shfl4 $type == 2),]$type = "shfl"
lg_hash_v_shfl4 [which(lg_hash_v_shfl4 $type == 4),]$type = "hash"

temp1 <- lg_hash_a8[which(lg_hash_a8$type == 2),]
temp2 <- lg_hash_a8[which(lg_hash_a8$type == 4),]
lg_hash_v_shfl8 <- rbind(temp1, temp2)
lg_hash_v_shfl8[which(lg_hash_v_shfl8$type == 2),]$type = "shfl"
lg_hash_v_shfl8 [which(lg_hash_v_shfl8$type == 4),]$type = "hash"

temp1 <- lg_hash_a12[which(lg_hash_a12$type == 2),]
temp2 <- lg_hash_a12[which(lg_hash_a12$type == 4),]
lg_hash_v_shfl12 <- rbind(temp1, temp2)
lg_hash_v_shfl12[which(lg_hash_v_shfl12$type == 2),]$type = "shfl"
lg_hash_v_shfl12[which(lg_hash_v_shfl12$type == 4),]$type = "hash"

temp1 <- lg_hash_a16[which(lg_hash_a16$type == 2),]
temp2 <- lg_hash_a16[which(lg_hash_a16$type == 4),]
lg_hash_v_shfl16 <- rbind(temp1, temp2)
lg_hash_v_shfl16[which(lg_hash_v_shfl16$type == 2),]$type = "shfl"
lg_hash_v_shfl16[which(lg_hash_v_shfl16$type == 4),]$type = "hash"

temp1 <- lg_hash_a24[which(lg_hash_a24$type == 2),]
temp2 <- lg_hash_a24[which(lg_hash_a24$type == 4),]
lg_hash_v_shfl24 <- rbind(temp1, temp2)
lg_hash_v_shfl24[which(lg_hash_v_shfl24$type == 2),]$type = "shfl"
lg_hash_v_shfl24 [which(lg_hash_v_shfl24$type == 4),]$type = "hash"

combined <- rbind(lg_hash_v_shfl4, lg_hash_v_shfl8, lg_hash_v_shfl12, lg_hash_v_shfl16, lg_hash_v_shfl24)

# Plot
png("large_hash_line.png")
plot.new()
ggplot() +
  geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  ggtitle("Time vs. Number of Threads") +
  ylab("Time (ms)") +
  xlab("Number of Arrays") +
  theme_minimal() +
  scale_color_brewer("Type | Arrays Size", palette = "Paired")
dev.off()

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

combined <- rbind(sm_hash_v_lg_hash4, sm_hash_v_lg_hash8, sm_hash_v_lg_hash12, sm_hash_v_lg_hash16, sm_hash_v_lg_hash24)

# Plot
png("small_vs_large_hash.png")
plot.new()
ggplot() +
  geom_smooth(data = combined, mapping = aes(x = number_of_arrays, y = time, group=interaction(type, array_size), color=interaction(type, array_size)), se=F) +
  ggtitle("Time vs. Number of Threads") +
  ylab("Time (ms)") +
  xlab("Number of Arrays") +
  theme_minimal() +
  scale_color_brewer("Type | Arrays Size", palette = "Paired")
dev.off()