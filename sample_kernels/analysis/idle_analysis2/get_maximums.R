setwd("~/Academics/mtu/masters/programming/CUDA_Programming/sample_kernels/analysis/idle_analysis2")
options(scipen = 32)

# Note: Power in mW and time is in nanoseconds
energyUsage <- function(full_file, start_stop_file) {
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

dirs = c("coalescing", "matrixMul", "simpleCUFFT", "vectorAdd", "transpose", "word_count")

for (dir in dirs){

  s_bench <- sprintf("Benchmark: %s", dir)
  print(s_bench)

  bench_dirs <- list.files(dir)
  
  for (dir2 in bench_dirs) {
    
    s_config <- sprintf("Configuration: %s", dir2)
    print(s_config)
    
    full_dir2 <- paste0("./",dir,"/",dir2)
    
    file1 <- list.files(full_dir2, pattern="r1.csv")
    file2 <- list.files(full_dir2, pattern="r2.csv")
    file3 <- list.files(full_dir2, pattern="r3.csv")
    file4 <- list.files(full_dir2, pattern="r4.csv")
    file5 <- list.files(full_dir2, pattern="r5.csv")
    
    file1 <- paste0(full_dir2, "/", file1)
    file2 <- paste0(full_dir2, "/", file2)
    file3 <- paste0(full_dir2, "/", file3)
    file4 <- paste0(full_dir2, "/", file4)
    file5 <- paste0(full_dir2, "/", file5)
    dir2_files <- c(file1, file2, file3, file4, file5)
    
    df1 <- read.csv(file1)
    df2 <- read.csv(file2)
    df3 <- read.csv(file3)
    df4 <- read.csv(file4)
    df5 <- read.csv(file5)
    
    avg = mean(max(df1$power_draw_mW),max(df2$power_draw_mW), max(df3$power_draw_mW), max(df4$power_draw_mW), max(df5$power_draw_mW))
    s_avg = sprintf("Average maximum power (mW) = %f", avg)
    
    start_stop_files <- list.files(full_dir2, pattern="start_stop")
    avg_total_energy <- 0
    avg_total_time_s <- 0
    
    for (k in 1:length(start_stop_files)) {
      start_stop_file <- paste0(full_dir2, "/", start_stop_files[k])
      usage <- energyUsage(dir2_files[k], start_stop_file)
      avg_total_energy <- avg_total_energy + usage[1]
      avg_total_time_s <- avg_total_time_s + usage[2]
    } 
    
    avg_total_energy <- avg_total_energy / k
    avg_total_time_s <- avg_total_time_s / k
    
    s_total_energy = sprintf("Average total energy (mJ) = %f", avg_total_energy)
    s_total_time = sprintf("Average total time (s) = %f", avg_total_time_s)

    print(s_avg)
    print(s_total_energy)
    print(s_total_time)
    print("------------------------------------")
  }
}
