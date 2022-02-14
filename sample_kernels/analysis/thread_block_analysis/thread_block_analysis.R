library(ggplot2)
library(forcats)

setwd("./")

## Transpose
transposeFileList <- list.files("./transpose")

transposeAllDf <- data.frame()
for (i in transposeFileList) {
  temp <- paste0("./transpose/", i)
  df <- read.csv(temp)
  transposeAllDf <- rbind(transposeAllDf, df)
}

transposeSummary <- aggregate(transposeAllDf$power_draw_mW, by = list(transposeAllDf$type), FUN = max)

colnames(transposeSummary) <- c("Type", "Max_Power")

png("transpose_max.png")
plot.new()
ggplot(transposeSummary, aes(x=Type, y=Max_Power)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = FALSE, color="black", fill = "white") +
  labs(x = "Configuration",
       y = "Max Power (mW)") +
  ggtitle("Transpose") +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

## Coalescing
coalescingFileList <- list.files("./coalescing")

coalescingAllDf <- data.frame()
for (i in coalescingFileList) {
  temp <- paste0("./coalescing/", i)
  df <- read.csv(temp)
  coalescingAllDf <- rbind(coalescingAllDf, df)
}

coalescingSummary <- aggregate(coalescingAllDf$power_draw_mW, by = list(coalescingAllDf$type), FUN = max)

colnames(coalescingSummary) <- c("Type", "Max_Power")

png("coalescing_max.png")
plot.new()
ggplot(coalescingSummary, aes(x=Type, y=Max_Power)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = FALSE, color="black", fill = "white") +
  labs(x = "Configuration",
       y = "Max Power (mW)") +
  ggtitle("Coalescing") +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

## Word Count
wcFileList <- list.files("./word_count")

wcAllDf <- data.frame()
for (i in wcFileList) {
  temp <- paste0("./word_count/", i)
  df <- read.csv(temp)
  wcAllDf <- rbind(wcAllDf, df)
}

wcSummary <- aggregate(wcAllDf$power_draw_mW, by = list(wcAllDf$type), FUN = max)

colnames(wcSummary) <- c("Type", "Max_Power")

png("wc_max.png")
plot.new()
ggplot(wcSummary, aes(x=Type, y=Max_Power)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = FALSE, color="black", fill = "white") +
  labs(x = "Configuration",
       y = "Max Power (mW)") +
  ggtitle("Word Count") +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()

## FFT
fftFileList <- list.files("./simpleCUFFT")

fftAllDf <- data.frame()
for (i in fftFileList) {
  temp <- paste0("./simpleCUFFT/", i)
  df <- read.csv(temp)
  fftAllDf <- rbind(fftAllDf, df)
}

fftSummary <- aggregate(fftAllDf$power_draw_mW, by = list(fftAllDf$type), FUN = max)

colnames(fftSummary) <- c("Type", "Max_Power")

png("fft_max.png")
plot.new()
ggplot(fftSummary, aes(x=Type, y=Max_Power)) + 
  geom_bar(position = "dodge", stat="identity", show.legend = FALSE, color="black", fill = "white") +
  labs(x = "Configuration",
       y = "Max Power (mW)") +
  ggtitle("FFT") +
  theme(plot.title = element_text(size = 14),
        # Hide panel borders and remove grid lines
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # Change axis line
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.background = element_rect(fill="white"),
        text = element_text(size=14))
dev.off()
