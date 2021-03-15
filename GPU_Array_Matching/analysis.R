setwd("/home/rwatling/Academics/MTU/masters/Programming/CUDA_Programming/GPU_Array_Matching")

file1 <- "data/change_arr_size_small.csv"
file2 <- "data/change_arr_size_avg.csv"
file3 <- "data/change_arr_size_lg.csv"
file4 <- "data/change_num_arr_small.csv"
file5 <- "data/change_num_arr_avg.csv"
file6 <- "data/change_num_arr_lg.csv"

### Change Array Size: file1-3###
changeArrSizeDf <- read.csv(file3)
sharedChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 1),]
globalChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 0),]

plot.new()
plot(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Size", pch =19,
     ylim=c(min(sharedChangeArrSizeDf$time), max(globalChangeArrSizeDf$time) + 0.005))
points(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue", pch =19)

lines(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue")
lines(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, col="red")

fit <- lm(globalChangeArrSizeDf$time ~ globalChangeArrSizeDf$array_size)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrSizeDf$time / sharedChangeArrSizeDf$time)

### Change number of arrays: files4-6 ###
changeArrNumDf <- read.csv(file4)
sharedChangeArrNumDf <- changeArrNumDf[which(changeArrNumDf$type == 1),]
globalChangeArrNumDf <- changeArrNumDf[which(changeArrNumDf$type == 0),]

plot.new()
plot(globalChangeArrNumDf$number_of_arrays, globalChangeArrNumDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Number", pch =19,
     ylim=c(min(sharedChangeArrNumDf$time), max(globalChangeArrNumDf$time) + 0.005))
points(sharedChangeArrNumDf$number_of_arrays, sharedChangeArrNumDf$time, col="blue", pch =19)

lines(sharedChangeArrNumDf$number_of_arrays, sharedChangeArrNumDf$time, col="blue")
lines(globalChangeArrNumDf$number_of_arrays, globalChangeArrNumDf$time, col="red")

fit <- lm(globalChangeArrNumDf$time ~ globalChangeArrNumDf$number_of_arrays)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrNumDf$time / sharedChangeArrNumDf$time)

