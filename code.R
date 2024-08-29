# Load necessary libraries  
library(MASS)    # For Linear Discriminant Analysis (LDA)  
library(klaR)    # for additional LDA functions  
library(psych)   # For factor analysis  

# Read the dataset  
data <- read.table("wind-quality-white1.txt", header = TRUE)  

# Calculate the correlation matrix  
cormatrix <- cor(data)  

# Perform PCA using the correlation matrix  
eig.pca <- eigen(cormatrix)  
print(eig.pca$values)  

# Plot the eigenvalues  
plot(1:length(eig.pca$values), eig.pca$values, type = "b",   
     xlab = "Index", ylab = "Eigenvalues", main = "Eigenvalues from PCA")  
abline(h = 1, col = "red")  

# Fit PCA models  
fit <- princomp(covmat = cormatrix)  
fit1 <- princomp(data)  
summary(fit)  
fit$loadings  
fit1$scores  

# Perform PCA using the prcomp function  
fit2 <- prcomp(data, center = TRUE, scale. = TRUE)  
summary(fit2)  

# Scree plot for explained variance  
screeplot(fit2, type = "l", main = "Scree Plot")  
p <- cumsum(fit2$sdev^2 / sum(fit2$sdev^2))  
plot(p, xlab = "Principal Components", ylab = "Cumulative Variance Explained",  
     main = "Cumulative Variance Explained")  
abline(h = 0.98, col = "blue")  
abline(v = 5, col = "blue")  

# Factor analysis using varimax rotation  
m_varimax <- factanal(covmat = cormatrix, factors = 5, rotation = "varimax")  
l_varimax <- m_varimax$loadings  

# Factor analysis using promax rotation  
m_promax <- factanal(covmat = cormatrix, factors = 5, rotation = "promax")  
l_promax <- m_promax$loadings  

# Calculate correlation and uniqueness  
psi <- diag(m_varimax$uniquenesses)  
rhat <- l_varimax %*% t(l_varimax) + psi  
r <- m_varimax$correlation  

# Factor analysis using psych package  
fa_results_varimax <- fa(r = data, nfactors = 5, rotate = "varimax", fm = "pa")  
fa_results_none <- fa(r = data, nfactors = 5, rotate = "none", fm = "pa")  

# Prepare data for LDA  
head(data)  
summary(data)  
table(data$quality)  

# Create a training and testing sample  
set.seed(1234)  
training_sample <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.6, 0.4))  
train <- data[training_sample, ]  
test <- data[!training_sample, ]  

# Fit the LDA model  
lda_model <- lda(quality ~ ., data = train)  

# Plot LDA results  
x.lda <- lda(quality ~ ., data = train)  
plot(x.lda, col = as.integer(train$quality) + 1)  

# Predictions on training data  
lda_train_pred <- predict(x.lda)  
train$lda <- lda_train_pred$class  
train_confusion <- table(train$lda, train$quality)  

# Predictions on testing data  
lda_test_pred <- predict(x.lda, test)  
test$lda <- lda_test_pred$class  
test_confusion <- table(test$lda, test$quality)  

# Confusion matrix and accuracy  
ct <- table(test$lda, test$quality)  
accuracy <- diag(prop.table(ct, 1))  
app_AER <- 1 - sum(diag(table(test$lda, test$quality))) / nrow(test)  

# Print accuracy  
print(paste("Approximate Accuracy Rate (AER):", round(app_AER, 4)))