setwd("/home/rwatling/Academics/MTU/masters/Programming/CUDA_Programming/GPU_Array_Matching")

#packages required
packages = c("gbutils")

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
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Size", pch =20)
points(sharedChangeArrSizeDf$array_size, sharedChangeArrSizeDf$time, col="blue", pch =20)

fit1 <- lm(globalChangeArrSizeDf$time ~ globalChangeArrSizeDf$array_size)
abline(fit1, col = "red", lwd=2)

fit2 <- lm(sharedChangeArrSizeDf$time ~ sharedChangeArrSizeDf$array_size)
abline(fit2, col = "blue", lwd=2)

legend("topleft", legend=c("Global Mem", "Shared Mem"),col=c("red", "blue"), pch = c(20, 20))

#speed up
speedup <- paste("Global Time / Shared Time =", toString(mean(globalChangeArrSizeDf$time / sharedChangeArrSizeDf$time)))
mtext(speedup, side = 3)

### Change number of arrays: files 4-6 ###
changeArrNumDf <- read.csv(file6)
sharedChangeArrNumDf <- changeArrNumDf[which(changeArrNumDf$type == 1),]
globalChangeArrNumDf <- changeArrNumDf[which(changeArrNumDf$type == 0),]

plot.new()
plot(globalChangeArrNumDf$number_of_arrays, globalChangeArrNumDf$time, 
     col="red", xlab="Array Size", ylab = "Time", main = "Time vs. Array Number", pch =20)
points(sharedChangeArrNumDf$number_of_arrays, sharedChangeArrNumDf$time, col="blue", pch =20)

fit1 <- lm(globalChangeArrNumDf$time ~ globalChangeArrNumDf$array_size)
if(gbutils::isNA(fit1$coefficients[2])) {
  fit1$coefficients[2] <- 0
}
abline(fit1, col = "red", lwd=2)

fit2 <- lm(sharedChangeArrNumDf$time ~ sharedChangeArrNumDf$array_size)
if(gbutils::isNA(fit2$coefficients[2] == "NA")) {
  fit2$coefficients[2] = 0
}
abline(fit2, col = "blue", lwd=2)

legend("topleft", legend=c("Global Mem", "Shared Mem"),col=c("red", "blue"), pch = c(20, 20, 20))

#speed up
speedup <- paste("Global Time / Shared Time =", toString(mean(globalChangeArrNumDf$time / sharedChangeArrNumDf$time)))
mtext(speedup, side = 3)
