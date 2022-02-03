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

##### Power consumption samples #####
setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/sample_kernels/analysis/extended_kernel_analysis")
power7df = read.csv("./cuSolverDn_LinearSolver_hardware_stats.csv")
power8df = read.csv("./simpleCUBLAS_hardware_stats.csv")
power9df = read.csv("./simpleCUBLAS_LU_hardware_stats.csv")
powerAdf = read.csv("./simpleCUFFT_hardware_stats.csv")
powerBdf = read.csv("./coalescing_hardware_stats.csv")
powerCdf = read.csv("./transpose_hardware_stats.csv")
powerDdf = read.csv("./word_count_hardware_stats.csv")

combined = rbind(power8df, power9df,powerAdf, powerBdf, powerCdf, powerDdf)

png("extended_kernels_power.png")
plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 1) +
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

png("linear_solver_power.png")
plot.new()
ggplot(power7df, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 1) +
  xlab("Time") +
  ylab("Power (mW)") +
  ggtitle("Linear Solver Power Consumption")+
  scale_x_discrete() +
  theme(legend.position=c(0.5, 0.2),
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

png("extended_kernels_temp.png")
plot.new()
ggplot(combined, aes(x = timestep, y=temperature_gpu, group = type, color = type)) +
  geom_smooth(show.legend = TRUE, size = 1) +
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

png("linear_solver_temp.png")
plot.new()
ggplot(power7df, aes(x = timestep, y=temperature_gpu, group = type, color = type)) +
  geom_smooth(show.legend = TRUE, size = 1) +
  xlab("Time") +
  ylab("Temperature (C)") +
  ggtitle("Linear Solver Time vs Temperature")+
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

# Clock frequency
png("extended_clock_freq.png")
plot.new()
ggplot(combined, aes(x = timestep, y=sm_clock_freq_mhz, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 1) +
  xlab("Time") +
  ylab("SM Clock MHz") +
  ggtitle("Sample Kernels Clock Rates")+
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

png("linear_solver_clock.png")
plot.new()
ggplot(power7df, aes(x = timestep, y=sm_clock_freq_mhz, group = type, color = type)) +
  geom_line(show.legend = TRUE, size = 1) +
  xlab("Time") +
  ylab("SM Clock MHz") +
  ggtitle("Linear Solver Clock Rates")+
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

##### MAXIMUM POWER CONSUMPTION #####
messages = c()

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
combined <- t(data.frame(maxPowerDf7, maxPowerDf8, 
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
