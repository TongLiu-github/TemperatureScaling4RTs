library(cvms)
library(mgcv)
library(groupdata2)
library(dplyr)
library(gamclass)
library(lme4)
library(stargazer)
library(stringr)

print("0. finished the loading")

args <- commandArgs(trailingOnly = TRUE)

arg1 <- args[1]
model_name <- args[2]
K <- args[3]
dir0 <- args[4]

BASE_DIR1 = paste(arg1, "/", model_name, "/", model_name, "_", K, "_", sep = "")
print(BASE_DIR1)

set.seed(42)
control <- lmerControl(optCtrl = list(maxfun = 100000))
Unzip <- function(...) rbind(data.frame(), ...)
prefix <- function(x) {
    return(str_sub(x, 1, 2))
}

BASE_DIR2 = paste(dir0, "data/all.txt.annotation.filtered.csv",  sep = "")
eye_data <- read.csv(BASE_DIR2, header = T, sep = "\t", quote = '"')

print("1. opened the file")
print(nrow(eye_data))

file = "scores.csv"
data <- read.csv(paste(BASE_DIR1, file, sep = ""), header = T, sep = "\t")
print(nrow(data))

data <- cbind(data, eye_data)



# copy columns
data$length0 <- data$word_length
data$length_prev_10 <- data$length_prev_1
data$log_gmean_freq0 <- data$log_gmean_freq




data$length <- scale(data$word_length)
data$gmean_freq <- scale(data$log_gmean_freq)
data$length_prev_1 <- scale(data$length_prev_1)
data$gmean_freq_prev_1 <- scale(data$log_gmean_freq_prev_1)
data$log_gmean_freq <- scale(data$log_gmean_freq)
data$log_gmean_freq_prev_1 <- scale(data$log_gmean_freq_prev_1)

data$surprisals_sum_raw <- data$surprisals_sum
data$surprisals <- data$surprisals_sum
data$surprisals_sum <- scale(data$surprisals_sum)
data$surprisals_sum_prev_1 <- scale(data$surprisals_sum_prev_1)
data$surprisals_sum_prev_2 <- scale(data$surprisals_sum_prev_2)
data$surprisals_sum_prev_3 <- scale(data$surprisals_sum_prev_3)

subdata <- subset(data, RT > 0)
subdata <- subset(subdata, has_num == "False")
# subdata <- subset(subdata, has_num_prev_1 == "False")
subdata <- subset(subdata, has_punct == "False")
# subdata <- subset(subdata, has_punct_prev_1 == "False")
# subdata <- subset(subdata, is_first_last == "False")
# subdata <- subset(subdata, correct > 4)

# print(file)
print("2. got the data")
print("3. start to fit")
base_mod_linear <- lmer(RT ~ log_gmean_freq * length + log_gmean_freq_prev_1 * length_prev_1 + (1 | item) + (1 | WorkerId), data = subdata, REML = FALSE)
base_linear_fit_logLik <- logLik(base_mod_linear) / nrow(subdata)
lm_mod <- lmer(RT ~ surprisals_sum + surprisals_sum_prev_1 + surprisals_sum_prev_2 + log_gmean_freq * length + log_gmean_freq_prev_1 * length_prev_1  + (1 | item) + (1 | WorkerId), data = subdata, REML = FALSE)
sup_linear_fit_logLik <- logLik(lm_mod) / nrow(subdata)
chi_p_linear <- anova(base_mod_linear, lm_mod)$Pr[2]





# predcit_base <- predict(base_linear_fit_logLik, subdata2)
# predcit_lm <- predict(sup_linear_fit_logLik, subdata2)





print("4. finished the fitting")





out = paste(BASE_DIR1, "PPP.txt")
out2 = paste(BASE_DIR1, "residuals.txt")

if (file.exists(out)) {
    file.remove(out)
}
if (file.exists(out2)) {
    file.remove(out2)
}
file.create(out, showWarnings = TRUE)
write(paste("linear_fit_logLik: ", sup_linear_fit_logLik), file = out, append = T)
write(paste("delta_linear_fit_logLik: ", sup_linear_fit_logLik - base_linear_fit_logLik), file = out, append = T)
write(paste("delta_linear_fit_chi_p: ", chi_p_linear), file = out, append = T)

residuals.frame <- do.call(Unzip, as.list(residuals(lm_mod)))
colnames(residuals.frame) <- c("residual")
residual_subdata <- cbind(residuals.frame, subdata$pos, subdata$word, subdata$length0, subdata$surprisals, subdata$log_gmean_freq0, subdata$ner)
write.csv(residual_subdata, out2)

print("5. finished the saving")

# print(summary(base_mod_linear))
