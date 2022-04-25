setwd("~/Academics/mtu/masters/programming/CUDA_Programming/sample_kernels/analysis/thread_block_analysis/")

source("../energyUsage.R")

dirs <- list.files("./")
dirs = dirs[-4]

for (i in 1:length(dirs)) {
  path <- paste0("./", dirs[i], "/")
  
  my_files = list.files(path)
  
  for (j in 1:(length(my_files)%/%2)) {
    full <- paste0(path, my_files[j])
    start_stop <- paste0(path, my_files[(length(my_files)%/%2)+j])
    print(my_files[j])
    print(energyUsage(full, start_stop))
    print(energyUsage(start_stop, full))
  }
  
  readline(prompt="Press [enter] to continue")
}
