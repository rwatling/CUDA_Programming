setwd("/home/rwatling/Academics/MTU/masters/Programming/CUDA_Programming/GPU_Array_Matching")

file1 <- "data/change_arr_size_small.csv"
file2 <- "data/change_arr_size_avg.csv"
file3 <- "data/change_arr_size_lg.csv"
file4 <- "data/change_num_arr_small.csv"
file5 <- "data/change_num_arr_avg.csv"
file6 <- "data/change_num_arr_lg.csv"

### File 1: Small number of arrays, change array size ###
changeArrSizeDf <- read.csv(file1)
sharedChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 1),]
globalChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 0),]

plot(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Size", pch =19)
points(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue", pch =19)

lines(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue")
lines(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, col="red")

fit <- lm(globalChangeArrSizeDf$time ~ globalChangeArrSizeDf$array_size)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrSizeDf$time / sharedChangeArrSizeDf$time)

### File 2: Average number of arrays, change array size###
changeArrSizeDf <- read.csv(file2)
sharedChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 1),]
globalChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 0),]

plot(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Size", pch =19)
points(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue", pch =19)

lines(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue")
lines(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, col="red")

fit <- lm(globalChangeArrSizeDf$time ~ globalChangeArrSizeDf$array_size)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrSizeDf$time / sharedChangeArrSizeDf$time)

### File 3: Large number of arrays, change array size###
changeArrSizeDf <- read.csv(file3)
sharedChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 1),]
globalChangeArrSizeDf <- changeArrSizeDf[which(changeArrSizeDf$type == 0),]

plot(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Size", pch =19)
points(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue", pch =19)

lines(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue")
lines(globalChangeArrSizeDf$array_size, globalChangeArrSizeDf$time, col="red")

fit <- lm(globalChangeArrSizeDf$time ~ globalChangeArrSizeDf$array_size)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrSizeDf$time / sharedChangeArrSizeDf$time)

### File 4: Small array size, change number of arrays ###
changeArrNumDf <- read.csv(file4)
sharedChangeArrNumDf <- changeArrSizeDf[which(changeArrNumDf$type == 1),]
globalChangeArrNumDf <- changeArrSizeDf[which(changeArrNumDf$type == 0),]

plot(globalChangeArrNumDf$array_size, globalChangeArrNumDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Number", pch =19)
points(sharedChangeArrNumDf$array_size, sharedChangeArrNumDf$time, col="blue", pch =19)

lines(sharedChangeArrNumDf$array_size, sharedChangeArrNumDf$time, col="blue")
lines(globalChangeArrNumDf$array_size, globalChangeArrNumDf$time, col="red")

fit <- lm(globalChangeArrNumDf$time ~ globalChangeArrNumDf$array_num)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrNumDf$time / sharedChangeArrNumDf$time)

### File 5: Average array size, change number of arrays ###
changeArrNumDf <- read.csv(file5)
sharedChangeArrNumDf <- changeArrSizeDf[which(changeArrNumDf$type == 1),]
globalChangeArrNumDf <- changeArrSizeDf[which(changeArrNumDf$type == 0),]

plot(globalChangeArrNumDf$array_size, globalChangeArrNumDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Number", pch =19)
points(sharedChangeArrNumDf$array_size, sharedChangeArrNumDf$time, col="blue", pch =19)

lines(sharedChangeArrNumDf$array_size, sharedChangeArrNumDf$time, col="blue")
lines(globalChangeArrNumDf$array_size, globalChangeArrNumDf$time, col="red")

fit <- lm(globalChangeArrNumDf$time ~ globalChangeArrNumDf$array_num)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrNumDf$time / sharedChangeArrNumDf$time)

### File 6: Large array size, change number of arrays ###
changeArrNumDf <- read.csv(file6)
sharedChangeArrNumDf <- changeArrSizeDf[which(changeArrNumDf$type == 1),]
globalChangeArrNumDf <- changeArrSizeDf[which(changeArrNumDf$type == 0),]

plot(globalChangeArrNumDf$array_size, globalChangeArrNumDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Number", pch =19)
points(sharedChangeArrNumDf$array_size, sharedChangeArrNumDf$time, col="blue", pch =19)

lines(sharedChangeArrNumDf$array_size, sharedChangeArrNumDf$time, col="blue")
lines(globalChangeArrNumDf$array_size, globalChangeArrNumDf$time, col="red")

fit <- lm(globalChangeArrNumDf$time ~ globalChangeArrNumDf$array_num)
abline(fit)

legend("topleft", legend=c("Global Mem", "Shared Mem", "Global Fit"),col=c("red", "blue", "black"), pch = c(19, 19, 19))

#speed up
mean(globalChangeArrNumDf$time / sharedChangeArrNumDf$time)

