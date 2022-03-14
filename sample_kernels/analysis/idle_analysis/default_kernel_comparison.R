library(ggplot2)

setwd("~/CUDA_Programming/sample_kernels/analysis/default_kernel_comparison")
options(scipen = 32)

# Default kernels
# Matrix Multiply
matrixMul_df <- read.csv("matrixMul_default.csv")
matrixMul_df$type <- "matrixMul_compute"
matrixMul_df_startstop <- read.csv("start_stop_matrixMul_compute.csv")
matrixMul_df_startstop$timestamp <- matrixMul_df_startstop$timestamp - min(matrixMul_df$timestamp)
matrixMul_df$timestamp <- matrixMul_df$timestamp - min(matrixMul_df$timestamp)

# Vector Add
vectorAdd_df <- read.csv("vectorAdd_default.csv")
vectorAdd_df$type <- "vectorAdd_compute"
vectorAdd_df_startstop <- read.csv("start_stop_vectorAdd_compute.csv")
vectorAdd_df_startstop$timestamp <- vectorAdd_df_startstop$timestamp - min(vectorAdd_df$timestamp)
vectorAdd_df$timestamp <- vectorAdd_df$timestamp - min(vectorAdd_df$timestamp)

# CUFFT
simpleCUFFT_df <- read.csv("simpleCUFFT_default.csv")
simpleCUFFT_df$type <- "simpleCUFFT_hybrid"
simpleCUFFT_df_startstop <- read.csv("start_stop_simpleCUFFT_hybrid.csv")
simpleCUFFT_df_startstop$timestamp <- simpleCUFFT_df_startstop$timestamp - min(simpleCUFFT_df$timestamp)
simpleCUFFT_df$timestamp <- simpleCUFFT_df$timestamp - min(simpleCUFFT_df$timestamp)

# Coalescing
coalescing_df <- read.csv("coalescing_default.csv")
coalescing_df$type <- "coalescing_memory"
coalescing_df_startstop <- read.csv("start_stop_coalescing_memory.csv")
coalescing_df_startstop$timestamp <- coalescing_df_startstop$timestamp - min(coalescing_df$timestamp)
coalescing_df$timestamp <- coalescing_df$timestamp - min(coalescing_df$timestamp)

# Transpose
transpose_df <- read.csv("transpose_default.csv")
transpose_df$type <- "transpose_memory"
transpose_df_startstop <- read.csv("start_stop_transpose_memory.csv")
transpose_df_startstop$timestamp <- transpose_df_startstop$timestamp - min(transpose_df$timestamp)
transpose_df_startstop$type <- "transpose_memory"
transpose_df$timestamp <- transpose_df$timestamp - min(transpose_df$timestamp)

# Word count
wordcount_df <- read.csv("wordcount_default.csv")
wordcount_df$type <- "wordcount_memory"
wordcount_df_startstop <- read.csv("start_stop_wordcount_memory.csv")
wordcount_df_startstop$timestamp <- wordcount_df_startstop$timestamp - min(wordcount_df$timestamp)
wordcount_df$timestamp <- wordcount_df$timestamp - min(wordcount_df$timestamp)

# For color scheme
cols <- c("coalescing_memory" = "steelblue", "matrixMul_compute" = "red", 
         "transpose_memory" = "steelblue2", "vectorAdd_compute" = "red3", 
         "wordcount_memory" = "steelblue3", "simpleCUFFT_hybrid" = "green")

# Matrix Multiply Figure
png("matrixMul_default.png")
plot.new()
ggplot(matrixMul_df, aes(x = timestamp, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  geom_point(aes(x = matrixMul_df_startstop[1, 2], y = matrixMul_df_startstop[1,3], color = matrixMul_df_startstop[1,1])) +
  geom_point(aes(x = matrixMul_df_startstop[2, 2], y = matrixMul_df_startstop[2,3], color = matrixMul_df_startstop[2,1])) +
  geom_point(aes(x = matrixMul_df_startstop[3, 2], y = matrixMul_df_startstop[3,3], color = matrixMul_df_startstop[3,1])) +
  geom_point(aes(x = matrixMul_df_startstop[4, 2], y = matrixMul_df_startstop[4,3], color = matrixMul_df_startstop[4,1])) +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Matrix Multiply (Compute) Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

# Vector Add Figure
png("vectorAdd_default.png")
plot.new()
ggplot(vectorAdd_df, aes(x = timestamp, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  geom_point(aes(x = vectorAdd_df_startstop[1, 2], y = vectorAdd_df_startstop[1,3], color = vectorAdd_df_startstop[1,1])) +
  geom_point(aes(x = vectorAdd_df_startstop[2, 2], y = vectorAdd_df_startstop[2,3], color = vectorAdd_df_startstop[2,1])) +
  geom_point(aes(x = vectorAdd_df_startstop[3, 2], y = vectorAdd_df_startstop[3,3], color = vectorAdd_df_startstop[3,1])) +
  geom_point(aes(x = vectorAdd_df_startstop[4, 2], y = vectorAdd_df_startstop[4,3], color = vectorAdd_df_startstop[4,1])) +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Vector Add (Compute) Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

# simpleCUFFT Figure
png("simpleCUFFT_default.png")
plot.new()
ggplot(simpleCUFFT_df, aes(x = timestamp, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  geom_point(aes(x = simpleCUFFT_df_startstop[1, 2], y = simpleCUFFT_df_startstop[1,3], color = simpleCUFFT_df_startstop[1,1])) +
  geom_point(aes(x = simpleCUFFT_df_startstop[2, 2], y = simpleCUFFT_df_startstop[2,3], color = simpleCUFFT_df_startstop[2,1])) +
  geom_point(aes(x = simpleCUFFT_df_startstop[3, 2], y = simpleCUFFT_df_startstop[3,3], color = simpleCUFFT_df_startstop[3,1])) +
  geom_point(aes(x = simpleCUFFT_df_startstop[4, 2], y = simpleCUFFT_df_startstop[4,3], color = simpleCUFFT_df_startstop[4,1])) +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Simple CUDA FFT (Hybrid) Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

# coalescing memory Figure
png("coalescing_default.png")
plot.new()
ggplot(coalescing_df, aes(x = timestamp, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  geom_point(aes(x = coalescing_df_startstop[1, 2], y = coalescing_df_startstop[1,3], color = coalescing_df_startstop[1,1])) +
  geom_point(aes(x = coalescing_df_startstop[2, 2], y = coalescing_df_startstop[2,3], color = coalescing_df_startstop[2,1])) +
  geom_point(aes(x = coalescing_df_startstop[3, 2], y = coalescing_df_startstop[3,3], color = coalescing_df_startstop[3,1])) +
  geom_point(aes(x = coalescing_df_startstop[4, 2], y = coalescing_df_startstop[4,3], color = coalescing_df_startstop[4,1])) +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Coalescing (Memory) Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

# transpose Figure
png("transpose_default.png")
plot.new()
ggplot(transpose_df, aes(x = timestamp, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  geom_point(aes(x = transpose_df_startstop[1, 2], y = transpose_df_startstop[1,3], color = transpose_df_startstop[1,1])) +
  geom_point(aes(x = transpose_df_startstop[2, 2], y = transpose_df_startstop[2,3], color = transpose_df_startstop[2,1])) +
  geom_point(aes(x = transpose_df_startstop[3, 2], y = transpose_df_startstop[3,3], color = transpose_df_startstop[3,1])) +
  geom_point(aes(x = transpose_df_startstop[4, 2], y = transpose_df_startstop[4,3], color = transpose_df_startstop[4,1])) +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Transpose (Memory) Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

# wordCount Figure
png("wordCount_default.png")
plot.new()
ggplot(wordcount_df, aes(x = timestamp, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  geom_point(aes(x = wordcount_df_startstop[1, 2], y = wordcount_df_startstop[1,3], color = wordcount_df_startstop[1,1])) +
  geom_point(aes(x = wordcount_df_startstop[2, 2], y = wordcount_df_startstop[2,3], color = wordcount_df_startstop[2,1])) +
  geom_point(aes(x = wordcount_df_startstop[3, 2], y = wordcount_df_startstop[3,3], color = wordcount_df_startstop[3,1])) +
  geom_point(aes(x = wordcount_df_startstop[4, 2], y = wordcount_df_startstop[4,3], color = wordcount_df_startstop[4,1])) +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Word Count (Memory) Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

combined <- rbind(coalescing_df, transpose_df, wordcount_df, simpleCUFFT_df, matrixMul_df, vectorAdd_df)

png("default_all.png")
plot.new()
ggplot(combined, aes(x = timestep, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("All Sample Kernels Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position=c(0.8, 0.8),
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
dev.off()

print("Matrix Multiply")
print(matrixMul_df_startstop[3,2]-matrixMul_df_startstop[2,2])
print("Vector Add")
print(vectorAdd_df_startstop[3,2]-vectorAdd_df_startstop[2,2])
print("CUFFT")
print(simpleCUFFT_df_startstop[3,2]-simpleCUFFT_df_startstop[2,2])
print("Coalescing")
print(coalescing_df_startstop[3,2]-coalescing_df_startstop[2,2])
print("Transpose")
print(transpose_df_startstop[3,2]-transpose_df_startstop[2,2])
print("Wordcount")
print(wordcount_df_startstop[3,2]-wordcount_df_startstop[2,2])
