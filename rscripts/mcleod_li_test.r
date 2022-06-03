library(TSA)
args = commandArgs(trailingOnly=TRUE)
if (length(args) != 2) {
  stop("Two arguments must be supplied (input file and output file).n", call.=FALSE)
}
ts <- read.csv(file=args[1])
#data(CREF);
#r.cref <- diff(log(CREF))*100;
res <- McLeod.Li.test(y=ts, plot=FALSE, gof.lag=10)
write.csv(res, file="_temp_output.csv", row.names=FALSE)
