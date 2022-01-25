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

# Power consumption
power1df = read.csv("./data/hardware_stats_shm_nested.csv")
power2df = read.csv("./data/hardware_stats_shfl_nested.csv")
power3df = read.csv("./data/hardware_stats_shfl_unroll.csv")
power4df = read.csv("./data/hardware_stats_shfl_unroll2.csv")
power5df = read.csv("./data/hardware_stats_shfl_hash.csv")
power6df = read.csv("./data/hardware_stats_shfl_bs.csv")

combined = rbind(power1df, power2df, power3df, power4df, power5df, power6df)

plot.new()
ggplot(combined, aes(x = timestep, y=power_draw_w, group = type, color = type)) +
  geom_line(show.legend = TRUE) +
  xlab("Time") +
  ylab("Power (mW)") +
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