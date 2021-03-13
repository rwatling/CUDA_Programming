setwd("/home/rwatling/Academics/MTU/masters/Programming/CUDA_Programming/GPU_Array_Matching")

changeArrSizeDf <- read.csv("change_arr_size.csv")
sharedChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 1),]
globalChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 0),]

plot(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Size", pch =16)
points(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue", pch =18)

lines(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue")
lines(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, col="red")

