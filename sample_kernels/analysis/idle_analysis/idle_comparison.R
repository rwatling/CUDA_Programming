library(ggplot2)

setwd("~/CUDA_Programming/sample_kernels/analysis/default_kernel_comparison")
options(scipen = 32)

### Default kernels ###
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

### IDLE: Vector Add ###
setwd("~/CUDA_Programming/sample_kernels/analysis/idle_analysis")
vectorAdd_idle32_df <- read.csv("./vector_add/vectorAdd_idle32.csv")
vectorAdd_idle32_start_stop <- read.csv("./vector_add/start_stop_idle32_vectorAdd_compute.csv")
vectorAdd_idle32_start_stop$timestamp <- vectorAdd_idle32_start_stop$timestamp - min(vectorAdd_idle32_df$timestamp)
vectorAdd_idle32_df$timestamp <- vectorAdd_idle32_df$timestamp - min(vectorAdd_idle32_df$timestamp)

vectorAdd_idle64_df <- read.csv("./vector_add/vectorAdd_idle64.csv")
vectorAdd_idle64_start_stop <- read.csv("./vector_add/start_stop_idle64_vectorAdd_compute.csv")
vectorAdd_idle64_start_stop$timestamp <- vectorAdd_idle64_start_stop$timestamp - min(vectorAdd_idle64_df$timestamp)
vectorAdd_idle64_df$timestamp <- vectorAdd_idle64_df$timestamp - min(vectorAdd_idle64_df$timestamp)

vectorAdd_idle128_df <- read.csv("./vector_add/vectorAdd_idle128.csv")
vectorAdd_idle128_start_stop <- read.csv("./vector_add/start_stop_idle128_vectorAdd_compute.csv")
vectorAdd_idle128_start_stop$timestamp <- vectorAdd_idle128_start_stop$timestamp - min(vectorAdd_idle128_df$timestamp)
vectorAdd_idle128_df$timestamp <- vectorAdd_idle128_df$timestamp - min(vectorAdd_idle128_df$timestamp)

vectorAdd_idle256_df <- read.csv("./vector_add/vectorAdd_idle256.csv")
vectorAdd_idle256_start_stop <- read.csv("./vector_add/start_stop_idle256_vectorAdd_compute.csv")
vectorAdd_idle256_start_stop$timestamp <- vectorAdd_idle256_start_stop$timestamp - min(vectorAdd_idle256_df$timestamp)
vectorAdd_idle256_df$timestamp <- vectorAdd_idle256_df$timestamp - min(vectorAdd_idle256_df$timestamp)

vectorAdd_idle512_df <- read.csv("./vector_add/vectorAdd_idle512.csv")
vectorAdd_idle512_start_stop <- read.csv("./vector_add/start_stop_idle512_vectorAdd_compute.csv")
vectorAdd_idle512_start_stop$timestamp <- vectorAdd_idle512_start_stop$timestamp - min(vectorAdd_idle512_df$timestamp)
vectorAdd_idle512_df$timestamp <- vectorAdd_idle512_df$timestamp - min(vectorAdd_idle512_df$timestamp)

vectorAdd_all_df <- rbind(vectorAdd_idle32_df, vectorAdd_idle64_df, vectorAdd_idle128_df, vectorAdd_idle256_df, vectorAdd_idle512_df)


png("vector_idle.png")
ggplot(vectorAdd_all_df, aes(x = timestep, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Vector Add Idle Threads Comparison")+
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

### IDLE: CUFFT ###
setwd("~/CUDA_Programming/sample_kernels/analysis/idle_analysis")
simpleCUFFT_idle32_df <- read.csv("./simple_cufft/simpleCUFFT_idle32.csv")
simpleCUFFT_idle32_start_stop <- read.csv("./simple_cufft/start_stop_idle32_simpleCUFFT_hybrid.csv")
simpleCUFFT_idle32_start_stop$timestamp <- simpleCUFFT_idle32_start_stop$timestamp - min(simpleCUFFT_idle32_df$timestamp)
simpleCUFFT_idle32_df$timestamp <- simpleCUFFT_idle32_df$timestamp - min(simpleCUFFT_idle32_df$timestamp)

simpleCUFFT_idle64_df <- read.csv("./simple_cufft/simpleCUFFT_idle64.csv")
simpleCUFFT_idle64_start_stop <- read.csv("./simple_cufft/start_stop_idle64_simpleCUFFT_hybrid.csv")
simpleCUFFT_idle64_start_stop$timestamp <- simpleCUFFT_idle64_start_stop$timestamp - min(simpleCUFFT_idle64_df$timestamp)
simpleCUFFT_idle64_df$timestamp <- simpleCUFFT_idle64_df$timestamp - min(simpleCUFFT_idle64_df$timestamp)

simpleCUFFT_idle128_df <- read.csv("./simple_cufft/simpleCUFFT_idle128.csv")
simpleCUFFT_idle128_start_stop <- read.csv("./simple_cufft/start_stop_idle128_simpleCUFFT_hybrid.csv")
simpleCUFFT_idle128_start_stop$timestamp <- simpleCUFFT_idle128_start_stop$timestamp - min(simpleCUFFT_idle128_df$timestamp)
simpleCUFFT_idle128_df$timestamp <- simpleCUFFT_idle128_df$timestamp - min(simpleCUFFT_idle128_df$timestamp)

simpleCUFFT_idle256_df <- read.csv("./simple_cufft/simpleCUFFT_idle256.csv")
simpleCUFFT_idle256_start_stop <- read.csv("./simple_cufft/start_stop_idle256_simpleCUFFT_hybrid.csv")
simpleCUFFT_idle256_start_stop$timestamp <- simpleCUFFT_idle256_start_stop$timestamp - min(simpleCUFFT_idle256_df$timestamp)
simpleCUFFT_idle256_df$timestamp <- simpleCUFFT_idle256_df$timestamp - min(simpleCUFFT_idle256_df$timestamp)

simpleCUFFT_idle512_df <- read.csv("./simple_cufft/simpleCUFFT_idle512.csv")
simpleCUFFT_idle512_start_stop <- read.csv("./simple_cufft/start_stop_idle512_simpleCUFFT_hybrid.csv")
simpleCUFFT_idle512_start_stop$timestamp <- simpleCUFFT_idle512_start_stop$timestamp - min(simpleCUFFT_idle512_df$timestamp)
simpleCUFFT_idle512_df$timestamp <- simpleCUFFT_idle512_df$timestamp - min(simpleCUFFT_idle512_df$timestamp)

simpleCUFFT_all_df <- rbind(simpleCUFFT_idle32_df, simpleCUFFT_idle64_df, simpleCUFFT_idle128_df, simpleCUFFT_idle256_df, simpleCUFFT_idle512_df)

png("cufft_idle.png")
ggplot(simpleCUFFT_all_df, aes(x = timestep, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Simple CUFFT Idle Threads Comparison")+
  theme(legend.position=c(0.4, 0.3),
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

### IDLE: coalescing ###
setwd("~/CUDA_Programming/sample_kernels/analysis/idle_analysis")
coalescing_idle32_df <- read.csv("./coalescing/coalescing_idle32.csv")
coalescing_idle32_start_stop <- read.csv("./coalescing/start_stop_idle32_coalescing_memory.csv")
coalescing_idle32_start_stop$timestamp <- coalescing_idle32_start_stop$timestamp - min(coalescing_idle32_df$timestamp)
coalescing_idle32_df$timestamp <- coalescing_idle32_df$timestamp - min(coalescing_idle32_df$timestamp)

coalescing_idle64_df <- read.csv("./coalescing/coalescing_idle64.csv")
coalescing_idle64_start_stop <- read.csv("./coalescing/start_stop_idle64_coalescing_memory.csv")
coalescing_idle64_start_stop$timestamp <- coalescing_idle64_start_stop$timestamp - min(coalescing_idle64_df$timestamp)
coalescing_idle64_df$timestamp <- coalescing_idle64_df$timestamp - min(coalescing_idle64_df$timestamp)

coalescing_idle128_df <- read.csv("./coalescing/coalescing_idle128.csv")
coalescing_idle128_start_stop <- read.csv("./coalescing/start_stop_idle128_coalescing_memory.csv")
coalescing_idle128_start_stop$timestamp <- coalescing_idle128_start_stop$timestamp - min(coalescing_idle128_df$timestamp)
coalescing_idle128_df$timestamp <- coalescing_idle128_df$timestamp - min(coalescing_idle128_df$timestamp)

coalescing_idle256_df <- read.csv("./coalescing/coalescing_idle256.csv")
coalescing_idle256_start_stop <- read.csv("./coalescing/start_stop_idle256_coalescing_memory.csv")
coalescing_idle256_start_stop$timestamp <- coalescing_idle256_start_stop$timestamp - min(coalescing_idle256_df$timestamp)
coalescing_idle256_df$timestamp <- coalescing_idle256_df$timestamp - min(coalescing_idle256_df$timestamp)

coalescing_idle512_df <- read.csv("./coalescing/coalescing_idle512.csv")
coalescing_idle512_start_stop <- read.csv("./coalescing/start_stop_idle512_coalescing_memory.csv")
coalescing_idle512_start_stop$timestamp <- coalescing_idle512_start_stop$timestamp - min(coalescing_idle512_df$timestamp)
coalescing_idle512_df$timestamp <- coalescing_idle512_df$timestamp - min(coalescing_idle512_df$timestamp)

coalescing_all_df <- rbind(coalescing_idle32_df, coalescing_idle64_df, coalescing_idle128_df, coalescing_idle256_df, coalescing_idle512_df)

png("coalescing_idle.png")
ggplot(coalescing_all_df, aes(x = timestep, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Coalescing Idle Threads Comparison")+
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

### IDLE: Word Count ###
setwd("~/CUDA_Programming/sample_kernels/analysis/idle_analysis")
wordcount_idle32_df <- read.csv("./word_count/wordcount_idle32.csv")
wordcount_idle32_start_stop <- read.csv("./word_count/start_stop_idle32_wordcount_memory.csv")
wordcount_idle32_start_stop$timestamp <- wordcount_idle32_start_stop$timestamp - min(wordcount_idle32_df$timestamp)
wordcount_idle32_df$timestamp <- wordcount_idle32_df$timestamp - min(wordcount_idle32_df$timestamp)

wordcount_idle64_df <- read.csv("./word_count/wordcount_idle64.csv")
wordcount_idle64_start_stop <- read.csv("./word_count/start_stop_idle64_wordcount_memory.csv")
wordcount_idle64_start_stop$timestamp <- wordcount_idle64_start_stop$timestamp - min(wordcount_idle64_df$timestamp)
wordcount_idle64_df$timestamp <- wordcount_idle64_df$timestamp - min(wordcount_idle64_df$timestamp)

wordcount_idle128_df <- read.csv("./word_count/wordcount_idle128.csv")
wordcount_idle128_start_stop <- read.csv("./word_count/start_stop_idle128_wordcount_memory.csv")
wordcount_idle128_start_stop$timestamp <- wordcount_idle128_start_stop$timestamp - min(wordcount_idle128_df$timestamp)
wordcount_idle128_df$timestamp <- wordcount_idle128_df$timestamp - min(wordcount_idle128_df$timestamp)

wordcount_idle256_df <- read.csv("./word_count/wordcount_idle256.csv")
wordcount_idle256_start_stop <- read.csv("./word_count/start_stop_idle256_wordcount_memory.csv")
wordcount_idle256_start_stop$timestamp <- wordcount_idle256_start_stop$timestamp - min(wordcount_idle256_df$timestamp)
wordcount_idle256_df$timestamp <- wordcount_idle256_df$timestamp - min(wordcount_idle256_df$timestamp)

wordcount_idle512_df <- read.csv("./word_count/wordcount_idle512.csv")
wordcount_idle512_start_stop <- read.csv("./word_count/start_stop_idle512_wordcount_memory.csv")
wordcount_idle512_start_stop$timestamp <- wordcount_idle512_start_stop$timestamp - min(wordcount_idle512_df$timestamp)
wordcount_idle512_df$timestamp <- wordcount_idle512_df$timestamp - min(wordcount_idle512_df$timestamp)

wordcount_all_df <- rbind(wordcount_idle32_df, wordcount_idle64_df, wordcount_idle128_df, wordcount_idle256_df, wordcount_idle512_df)

png("word_idle.png")
ggplot(wordcount_all_df, aes(x = timestep, y = power_draw_mW, group = type, color = type)) +
  geom_line() +
  ylab("Power (mW)") +
  xlab("Timestamp") +
  ggtitle("Word Count Idle Threads Comparison")+
  theme(legend.position=c(0.4, 0.3),
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
