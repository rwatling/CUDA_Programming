# Note: Power in mW and time is in nanoseconds
energyUsage <- function(full_file, start_stop_file) {
  options(scipen=32)
  start_stop_df <- read.csv(start_stop_file)
  full_df <- read.csv(full_file)
  
  kern_start <- start_stop_df[2,2]
  kern_end <- start_stop_df[3,2]
  
  full_df <- full_df[which(full_df$timestamp >= kern_start),]
  full_df <- full_df[which(full_df$timestamp <= kern_end),]
  
  total_energy <- 0
  total_time <- 0
  temp_energy <- 0
  
  # Integrate power via the trapezoidal method
  for (i in 1:(nrow(full_df) - 1)) {
    # area under curve at timestep i
    b1 <- full_df[i, 5]
    b2 <- full_df[i+1, 5]
    h <- full_df[i+1, 2] - full_df[i, 2]
    
    # (1/2) (base1 + base2) * height
    temp_energy <- 0.5 * (b1 + b2) * h
    total_energy <- total_energy + temp_energy
    total_time <- total_time + h
  }
  
  # Units is mW * nanoseconds which we convery to mW * s which is mJ
  convertToSeconds <- 1000000000
  total_energy_mJ <- total_energy / convertToSeconds
  
  retList <- c(total_energy_mJ, (total_time / convertToSeconds))
  return(retList)
}