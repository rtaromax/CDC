library(mice)
library(parallel)

fun <- function(X){return(mice(df_nonmiss,m=X,maxit=1,defaultMethod =c("pmm","logreg","polyreg"),diagnostics=FALSE))}

#fpath <- commandArgs(trailingOnly = TRUE)
fpath <- '/Users/rtaromax/Documents/cdc/datagroup_task/variable_correlation/3.1/dialysis_treated_3_1.csv'
df_dialysis_treatment <- read.csv(fpath)

#unique samples
df_dialysis_treatment <- unique(df_dialysis_treatment)[0:100,]

#convert to numerics
df_dialysis_treatment$FluidRemoved <- as.numeric(as.matrix(df_dialysis_treatment$FluidRemoved))
df_dialysis_treatment$LitersProcessed <- as.numeric(as.matrix(df_dialysis_treatment$LitersProcessed))

df_nonmiss <- df_dialysis_treatment[df_dialysis_treatment$TreatmentTypeCategory!='Missed',]
df_missed <- df_dialysis_treatment[df_dialysis_treatment$TreatmentTypeCategory=='Missed',]
df_date <- df_nonmiss$TreatmentStart
df_ID <- df_nonmiss$PatientIDNumber

df_nonmiss$TreatmentStart<-NULL
df_nonmiss$LocationID<-NULL
df_nonmiss$txSubtype<-NULL
df_nonmiss$PatientIDNumber<-NULL
df_nonmiss$InfectionPresent<-as.factor(df_nonmiss$InfectionPresent)

#outlier detection
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

for(i in names(df_nonmiss[c(3:21)])){
  df_nonmiss[[i]] <- remove_outliers(df_nonmiss[[i]])
}

#parallel
temp = mclapply(list(1,1,1,1),fun,mc.cores = 2)

eval_df_nonmiss <- sum(apply(df_nonmiss[,3:21],2,sd,na.rm=TRUE)/apply(df_nonmiss[,3:21],2,mean,na.rm=TRUE))
eval = 0
for (i in 1:4){
  impnam = paste('imp',i,sep = '')
  assign(impnam, complete(temp[[i]]))
  eval_temp <- sum(apply(get(impnam)[,3:21],2,sd)/apply(get(impnam)[,3:21],2,mean))
  if (1/(abs(eval_temp - eval_df_nonmiss)) > eval){
    eval <- eval_temp
    nam <- impnam
    }
}

impc = get(nam)
impe = eval

write.csv(impc, '/Users/rtaromax/Documents/cdc/Hospitalization/imp1.csv')
