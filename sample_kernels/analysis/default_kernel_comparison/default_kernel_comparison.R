library(ggplot2)

setwd("~/CUDA_Programming/sample_kernels/analysis/default_kernel_comparison")
options(scipen = 64)

coalescing_df <- read.csv("coalescing_default.csv")

plot.new()
ggplot() +
  geom_line(coalescing_df, aes(x = timestep, y = power_draw_mW)) +
  labs(x = "Time",
       y = "Max Power (mW)") +
  ggtitle("Coalescing") +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        axis.text.x = element_blank(),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
