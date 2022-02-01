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

##### Power consumption matching #####
setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/GPU_Array_Matching_v2/analysis")
power1df = read.csv("./data/hardware_stats_shm_nested.csv")
power2df = read.csv("./data/hardware_stats_shfl_nested.csv")
power3df = read.csv("./data/hardware_stats_shfl_unroll.csv")
power4df = read.csv("./data/hardware_stats_shfl_unroll2.csv")
power5df = read.csv("./data/hardware_stats_shfl_hash.csv")
power6df = read.csv("./data/hardware_stats_shfl_bs.csv")

combined = rbind(power1df, power2df, power3df, power4df, power5df, power6df)

png("../../sample_kernels/analysis/match_kernels_power.png")
plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 2) +
  xlab("Time") +
  ylab("Power (mW)") +
  ggtitle("Matching Kernels Power Consumption")+
  scale_x_discrete() +
  theme(legend.position=c(0.7, 0.4),
        plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

png("../../sample_kernels/analysis/simple_kernel_analysis/match_kernels_temp.png")
plot.new()
ggplot(combined, aes(x = timestep, y=temperature_gpu, group = type, color = type)) +
  geom_smooth(show.legend = TRUE) +
  xlab("Time") +
  ylab("Temperature (C)") +
  ggtitle("Matching Kernels Time vs Temperature")+
  scale_x_discrete( ) +
  theme(legend.position=c(0.8, 0.2),
        plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

##### Power consumption samples #####
setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/sample_kernels/analysis/simple_kernel_analysis")
power7df = read.csv("./cuSolverDn_LinearSolver_hardware_stats.csv")
power8df = read.csv("./simpleCUBLAS_hardware_stats.csv")
power9df = read.csv("./simpleCUBLAS_LU_hardware_stats.csv")
powerAdf = read.csv("./simpleCUFFT_hardware_stats.csv")
powerBdf = read.csv("./coalescing_hardware_stats.csv")
powerCdf = read.csv("./transpose_hardware_stats.csv")
powerDdf = read.csv("./word_count_hardware_stats.csv")

combined = rbind(power7df, power8df, power9df,powerAdf, powerBdf, powerCdf, powerDdf)

png("sample_kernels_power.png")
plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 2) +
  xlab("Time") +
  ylab("Power (mW)") +
  ggtitle("Sample Kernels Power Consumption")+
  scale_x_discrete() +
  theme(legend.position=c(0.8, 0.2),
        plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

png("sample_kernels_temp.png")
plot.new()
ggplot(combined, aes(x = timestep, y=temperature_gpu, group = type, color = type)) +
  geom_smooth(show.legend = TRUE, size = 2) +
  xlab("Time") +
  ylab("Temperature (C)") +
  ggtitle("Sample Kernels Time vs Temperature")+
  scale_x_discrete() +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

##### All power consumption #####
combined = rbind(power1df, power2df, power3df, power4df, power5df, power6df, power7df, power8df, power9df,powerAdf, powerBdf, powerCdf, powerDdf)
combined = combined[which(combined$timestep < 1500),]

png("all_kernels_power.png")
plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 2) +
  xlab("Time") +
  ylab("Power (mW)") +
  ggtitle("All Kernels Power Consumption")+
  scale_x_discrete( ) +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

# Clock frequency for each type
png("linearSolver_clock.png")
plot.new()
ggplot(power7df, aes(x = timestep)) +
  geom_line(aes(y=mem_clock_freq_mhz, color = "Memory Clock Freq")) +
  geom_line(aes(y=sm_clock_freq_mhz, color = "SM Clock Freq")) +
  labs(x = "Time",
       y = "Clock Freq Mhz",
       color = "Legend") +
  ggtitle("Linear Solver Clock Rate")+
  scale_color_manual(values = c("Memory Clock Freq" = "red", "SM Clock Freq" = "blue")) +
  theme(legend.position=c(0.8, 0.4),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14), aspect.ratio = 1/2)
dev.off()

##### MEDIAN CLOCK RATES (Note that they are constant) #####
messages = c()
messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                   power1df$type[1], 
                   median(power1df$mem_clock_freq_mhz), 
                   median(power1df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power1df$type[1], 
                                    median(power1df$mem_clock_freq_mhz), 
                                    median(power1df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power2df$type[1], 
                                    median(power2df$mem_clock_freq_mhz), 
                                    median(power2df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power3df$type[1], 
                                    median(power3df$mem_clock_freq_mhz), 
                                    median(power3df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power4df$type[1], 
                                    median(power4df$mem_clock_freq_mhz), 
                                    median(power4df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power5df$type[1], 
                                    median(power5df$mem_clock_freq_mhz), 
                                    median(power5df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power6df$type[1], 
                                    median(power6df$mem_clock_freq_mhz), 
                                    median(power6df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power7df$type[1], 
                                    median(power7df$mem_clock_freq_mhz), 
                                    median(power7df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power8df$type[1], 
                                    median(power8df$mem_clock_freq_mhz), 
                                    median(power8df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    power9df$type[1], 
                                    median(power9df$mem_clock_freq_mhz), 
                                    median(power9df$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    powerAdf$type[1], 
                                    median(powerAdf$mem_clock_freq_mhz), 
                                    median(powerAdf$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    powerBdf$type[1], 
                                    median(powerBdf$mem_clock_freq_mhz), 
                                    median(powerBdf$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    powerCdf$type[1], 
                                    median(powerCdf$mem_clock_freq_mhz), 
                                    median(powerCdf$sm_clock_freq_mhz)))

messages = append(messages, sprintf("%s MemClockFreq: %f SMClockFreq: %f", 
                                    powerDdf$type[1], 
                                    median(powerDdf$mem_clock_freq_mhz), 
                                    median(powerDdf$sm_clock_freq_mhz)))

write.table(messages, file = "./median_clock_rates", sep = "\n", row.names = FALSE, col.names = FALSE, quote = FALSE)

##### MAXIMUM POWER CONSUMPTION #####
messages = c()
messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power1df$type[1], 
                                    max(power1df$power_draw_mW), 
                                    max(power1df$power_limit_mW)))
maxPowerDf1 <- c(power1df$type[1], max(power1df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power2df$type[1], 
                                    max(power2df$power_draw_mW), 
                                    max(power2df$power_limit_mW)))
maxPowerDf2 <- c(power2df$type[1], max(power2df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power1df$type[1], 
                                    max(power3df$power_draw_mW), 
                                    max(power3df$power_limit_mW)))
maxPowerDf3 <- c(power3df$type[1], max(power3df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power4df$type[1], 
                                    max(power4df$power_draw_mW), 
                                    max(power4df$power_limit_mW)))
maxPowerDf4 <- c(power4df$type[1], max(power4df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power5df$type[1], 
                                    max(power5df$power_draw_mW), 
                                    max(power5df$power_limit_mW)))
maxPowerDf5 <- c(power5df$type[1], max(power5df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power6df$type[1], 
                                    max(power6df$power_draw_mW), 
                                    max(power6df$power_limit_mW)))
maxPowerDf6 <- c(power6df$type[1], max(power6df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power7df$type[1], 
                                    max(power7df$power_draw_mW), 
                                    max(power7df$power_limit_mW)))
maxPowerDf7 <- c(power7df$type[1], max(power7df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power8df$type[1], 
                                    max(power8df$power_draw_mW), 
                                    max(power8df$power_limit_mW)))
maxPowerDf8 <- c(power8df$type[1], max(power8df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    power9df$type[1], 
                                    max(power9df$power_draw_mW), 
                                    max(power9df$power_limit_mW)))
maxPowerDf9 <- c(power9df$type[1], max(power9df$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    powerAdf$type[1], 
                                    max(powerAdf$power_draw_mW), 
                                    max(powerAdf$power_limit_mW)))
maxPowerDfA <- c(powerAdf$type[1], max(powerAdf$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    powerBdf$type[1], 
                                    max(powerBdf$power_draw_mW), 
                                    max(powerBdf$power_limit_mW)))
maxPowerDfB <- c(powerBdf$type[1], max(powerBdf$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    powerCdf$type[1], 
                                    max(powerCdf$power_draw_mW), 
                                    max(powerCdf$power_limit_mW)))
maxPowerDfC <- c(powerCdf$type[1], max(powerCdf$power_draw_mW))

messages = append(messages, sprintf("%s Max Power Consumption: %d (mW) Max Power Limit: %d (mW)", 
                                    powerDdf$type[1], 
                                    max(powerDdf$power_draw_mW), 
                                    max(powerDdf$power_limit_mW)))
maxPowerDfD <- c(powerDdf$type[1], max(powerDdf$power_draw_mW))

# Text File
write.table(messages, file = "./max_power", sep = "\n", row.names = FALSE, col.names = FALSE, quote = FALSE)

##### PLOT MAXIMUM POWER #####
combined <- t(data.frame(maxPowerDf1, maxPowerDf2, maxPowerDf3, maxPowerDf4, 
                       maxPowerDf5, maxPowerDf6, maxPowerDf7, maxPowerDf8, 
                       maxPowerDf9, maxPowerDfA, maxPowerDfB, maxPowerDfC, 
                       maxPowerDfD))
row.names(combined) <- NULL
colnames(combined) <- c("type", "power")
combined <- data.frame(combined)

png("maximum_power_bar.png")
plot.new()
ggplot(combined, aes(x=type, y=power)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = FALSE, color="black", fill = "white") +
  labs(x = "Benchmark",
       y = "Max Power (mW)") +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14),
        aspect.ratio = 1/2)
dev.off()