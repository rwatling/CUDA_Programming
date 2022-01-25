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

# Power consumption matching
power1df = read.csv("./data/hardware_stats_shm_nested.csv")
power2df = read.csv("./data/hardware_stats_shfl_nested.csv")
power3df = read.csv("./data/hardware_stats_shfl_unroll.csv")
power4df = read.csv("./data/hardware_stats_shfl_unroll2.csv")
power5df = read.csv("./data/hardware_stats_shfl_hash.csv")
power6df = read.csv("./data/hardware_stats_shfl_bs.csv")

combined = rbind(power1df, power2df, power3df, power4df, power5df, power6df)

plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE) +
  xlab("Time") +
  ylab("Power (mW)") +
  ggtitle("Matching Problem Power Consumption")+
  scale_x_discrete(labels = NULL, breaks = NULL) +
  theme(legend.position=c(0.8, 0.3),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))

setwd("/home/rwatling/Academics/mtu/masters/programming/CUDA_Programming/sample_kernels/analysis")

# Power consumption
power7df = read.csv("../compute/cuSolverDn_LinearSolver/hardware_stats.csv")
power8df = read.csv("../compute/simpleCUBLAS/hardware_stats.csv")
power9df = read.csv("../compute/simpleCUBLAS_LU/hardware_stats.csv")
powerAdf = read.csv("../hybrid/simpleCUFFT/hardware_stats.csv")
powerBdf = read.csv("../memory/coalescing-global/hardware_stats.csv")
powerCdf = read.csv("../memory/transpose/hardware_stats.csv")
powerDdf = read.csv("../memory/word_count/hardware_stats.csv")

combined = rbind(power7df, power8df, powerAdf, powerBdf, powerCdf, powerDdf)

plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_mW, group = type, color = type)) +
  geom_line(show.legend = TRUE) +
  xlab("Time") +
  ylab("Power (mW)") +
  ggtitle("Matching Problem Power Consumption")+
  scale_x_discrete(labels = NULL, breaks = NULL) +
  theme(legend.position=c(0.8, 0.3),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
