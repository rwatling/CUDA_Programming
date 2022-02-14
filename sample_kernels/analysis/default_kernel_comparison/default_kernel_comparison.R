library(ggplot2)

setwd("~/CUDA_Programming/sample_kernels/analysis/default_kernel_comparison")
options(scipen = 32)

coalescing_df <- read.csv("coalescing_default.csv")
coalescing_df$type <- "coalescing_memory"
transpose_df <- read.csv("transpose_default.csv")
transpose_df$type <- "transpose_memory"
wordcount_df <- read.csv("wordcount_default.csv")
wordcount_df$type <- "wordcount_memory"

simpleCUFFT_df <- read.csv("simpleCUFFT_default.csv")
simpleCUFFT_df$type <- "simpleCUFFT_hybrid"

matrixMul_df <- read.csv("matrixMul_default.csv")
matrixMul_df$type <- "matrixMul_compute"
vectorAdd_df <- read.csv("vectorAdd_default.csv")
vectorAdd_df$type <- "vectorAdd_compute"

coalescing_df_startstop <- read.csv("start_stop_coalescing_memory.csv")
transpose_df_startstop <- read.csv("start_stop_transpose_memory.csv")
transpose_df_startstop$type <- "transpose_memory"
wordcount_df_startstop <- read.csv("start_stop_wordcount_memory.csv")
simpleCUFFT_df_startstop <- read.csv("start_stop_simpleCUFFT_hybrid.csv")
matrixMul_df_startstop <- read.csv("start_stop_matrixMul_compute.csv")
vectorAdd_df_startstop <- read.csv("start_stop_vectorAdd_compute.csv")

combined <- rbind(coalescing_df, transpose_df, wordcount_df, simpleCUFFT_df, matrixMul_df, vectorAdd_df)


cols <- c("coalescing_memory" = "steelblue", "matrixMul_compute" = "red", 
         "transpose_memory" = "steelblue2", "vectorAdd_compute" = "red3", 
         "wordcount_memory" = "steelblue3", "simpleCUFFT_hybrid" = "green")

plot.new()
ggplot(combined, aes(x = timestep, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  # Coalescing start stop
  geom_point(aes(x = coalescing_df_startstop[1, 2], y = coalescing_df_startstop[1,3], color = coalescing_df_startstop[1,1])) +
  geom_point(aes(x = coalescing_df_startstop[2, 2], y = coalescing_df_startstop[2,3], color = coalescing_df_startstop[2,1])) +
  # Transpose start stop
  geom_point(aes(x = transpose_df_startstop[1, 2], y = transpose_df_startstop[1,3], color = transpose_df_startstop[1,1])) +
  geom_point(aes(x = transpose_df_startstop[2, 2], y = transpose_df_startstop[2,3], color = transpose_df_startstop[2,1])) +
  # wordcount start stop
  geom_point(aes(x = wordcount_df_startstop[1, 2], y = wordcount_df_startstop[1,3], color = wordcount_df_startstop[1,1])) +
  geom_point(aes(x = wordcount_df_startstop[2, 2], y = wordcount_df_startstop[2,3], color = wordcount_df_startstop[2,1])) +
  # simpleCUFFT start stop
  geom_point(aes(x = simpleCUFFT_df_startstop[1, 2], y = simpleCUFFT_df_startstop[1,3], color = simpleCUFFT_df_startstop[1,1])) +
  geom_point(aes(x = simpleCUFFT_df_startstop[2, 2], y = simpleCUFFT_df_startstop[2,3], color = simpleCUFFT_df_startstop[2,1])) +
  # matrixMul start stop
  geom_point(aes(x = matrixMul_df_startstop[1, 2], y = matrixMul_df_startstop[1,3], color = matrixMul_df_startstop[1,1])) +
  geom_point(aes(x = matrixMul_df_startstop[2, 2], y = matrixMul_df_startstop[2,3], color = matrixMul_df_startstop[2,1])) +
  # vectorAdd start stop
  geom_point(aes(x = vectorAdd_df_startstop[1, 2], y = vectorAdd_df_startstop[1,3], color = vectorAdd_df_startstop[1,1])) +
  geom_point(aes(x = vectorAdd_df_startstop[2, 2], y = vectorAdd_df_startstop[2,3], color = vectorAdd_df_startstop[2,1])) +
  ylab("Power (mW)") +
  xlab("Timestep") +
  ggtitle("Sample Kernels Power Consumption")+
  scale_color_manual(values = cols) +
  theme(legend.position=c(0.8, 0.2),
        plot.title = element_text(hjust = 0.5 , size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),)

