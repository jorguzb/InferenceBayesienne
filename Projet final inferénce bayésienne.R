## Inférence bayésienne avec "rstanarm"

library(ggplot2)
library(bayesplot)

## Q1. Régression bayésienne gaussienne 

db = read.csv("datasets/mutations2.csv")
hist(db$Barre, breaks = 30)

# Pour la suite on va conserver uniquement la variable "Matiere" comme variable
# catégorielle car on gardant toutes les autres variables catégorielles on aura 
# des petites échantillons par modalités ce qui empêchera la modélisation 
# (exp. LYCEE RAFA - ANGLAIS, 2 observations)
# Il est nécessaire de conserver "barre" en première position  

db0 = db[,-c(1,2,3,4)]
db1 = db0[,c(2,1,3:ncol(db0))]

# Data train et test

set.seed(3)
ind = sample(1:nrow(db1), size = floor(nrow(db1)*0.9), replace = FALSE)
train = db1[ind,]
test = db1[-ind,]

# la library(BMS) n'est pas utilisée car elle ne reçoit pas des les variables
# catégorielles donc la library "rstanarm" a été utilisée

library(rstanarm)

# La librairie « rstanarm »  reçoit des variables catégorielles sans besoin 
# d'une transformation numérique mais les chaînes de markof ne convergent pas 
# même en augmentant le nombre d'interactions, les modèles réalisées font partie 
# de la documentation de ce projet (avec toutes les variables, uniquement avec 
# les variables « matière » et « établissement » 

# à fin de constater:
# load("barre_bglm_1.RData")# All variables
# load("barre_bglm_2.RData")# Uniquement les variables "matiere" et "etablissement"
# Il est possible que la non convergence du modèle soit liée au set des données
# que compte avec un nombre faible des observations en considérant la combinaison
# « établissement » et « matière » donc la variable « établissement » a été
# exclue de l'analyse, cette exclusion a permis la convergence du modèle mais
# l'estimation de la variable  « barre » a été mauvaise.

mod_b3 = stan_glm(Barre ~ ., data = train, cores = 2, seed = 31416, iter = 10000)
save(mod_b3, file = "mod_b3.RData")
load("mod_b3.RData")
summary(residuals(mod_b3))
pred_b_dist = posterior_predict(mod_b3, test)
y_pred_b = apply(pred_b_dist, 2, mean)
y_b = test$Barre
MSEB = mean((y_b-y_pred_b)^2)
hist(y_b-y_pred_b, breaks = 20)
# > MSEB
# [1] 82962.44
# > sqrt(MSEB)
# [1] 288.032

# Q2. Modèle fréquentiste classique
# ce modèle est rapide et sans transformation des variables 

mod = lm(Barre ~ ., data = train)
summary(mod)

# Dans le "sumamry" de ce modèle linéaire on peut constater un nombre 
# importante des variables significatives mais un "stepwise" par AIC a été fait 
# afin de retenir lesmeilleurs variables explicatives

mod_l = step(mod)
summary(mod_l) # la variable "établissement" ne sort pas 
y_pred = predict(mod_l, test)
y = test$Barre
MSE = mean((y-y_pred)^2)
hist(y-y_pred, breaks = 20)

# > MSE
# [1] 79927.19
# > sqrt(MSE)
# [1] 282.714

# Les erreurs quadratique  bayésiennes et fréquentiste sont éléves donc le data set 
# a été divisé selon le histogramme de distribution de la variable "Barre"
# car on peut observer deux clusters

hist(db1$Barre, breaks = 30)

# Sur la base de l'analyse prrécedent le data set passe de 516 observations
# à 471

db2 = db1[db1$Barre<1000,]
hist(db2$Barre)
set.seed(12345)
ind = sample(1:nrow(db2), size = floor(nrow(db2)*0.9), replace = FALSE)
train2 = db2[ind,]
test2 = db2[-ind,]
mod_b4 = stan_glm(Barre ~ ., data = train2, cores = 2, seed = 31416, iter = 10000)
save(mod_b4, file = "mod_b4.RData")
load("mod_b4.RData")
pred_b_dist = posterior_predict(mod_b4, test2)
y_pred_b = apply(pred_b_dist, 2, mean)
y_b = test2$Barre
MSEB = mean((y_b-y_pred_b)^2)
hist(y_b-y_pred_b, breaks = 20)

# L'erreur quadratique a été divisé par deux
# > MSEB
# [1] 10440.97
# > sqrt(MSEB)
# [1] 102.1811

# Modèle linéaire sur la nouvelle data

mod_lin = lm(Barre ~ ., data = train2)
mod_step = step(mod_lin)
y_pred = predict(mod_lin, test2)
y = test2$Barre
MSE = mean((y-y_pred)^2)
hist(y-y_pred, breaks = 20)

# L'erreur quadratique a été divisé par deux et proche de celui de l'estimation bayésienne
# > MSE
# [1] 10471.54
# > sqrt(MSE)
# [1] 102.3305

# Sur la base de l'erreur quadratique les modèles sont pareils

# Analyse des coefficients 

summary(mod_b4)

# Estimates:
#                                      mean   sd     10%    50%    90% 
# (Intercept)                        -216.1  176.5 -443.0 -217.7   10.4
# MatiereANGLAIS                     -128.7   36.1 -174.7 -129.0  -83.1
# MatiereARTS PLAST                   -74.8   90.2 -188.9  -75.0   39.8
# MatiereBIOCH.BIOL                  -166.5   60.3 -243.5 -167.0  -88.4
# MatiereBIOTECHNOL                  -209.6   66.7 -294.6 -210.0 -124.9
# MatiereDOC LYCEES                  -177.8   47.3 -238.2 -177.9 -117.1
# MatiereE. P. S                       -6.3   42.3  -60.1   -6.1   47.5
# MatiereECO.GE.COM                   -97.3   50.0 -161.5  -97.1  -33.2
# MatiereECO.GE.CPT                  -111.3   87.9 -223.1 -111.5    2.5
# MatiereECO.GE.FIN                   -63.2   50.9 -127.7  -63.7    1.8
# MatiereECO.GE.MK                    -63.3   44.6 -120.3  -63.4   -5.4
# MatiereECO.GE.VEN                  -121.9   57.5 -195.3 -121.7  -48.0
# MatiereEDUCATION                   -142.2   43.8 -198.0 -142.3  -86.3
# MatiereESPAGNOL                    -123.5   38.5 -172.5 -123.5  -74.5
# MatiereESTH.COSME                  -198.4  118.7 -351.0 -198.9  -47.0
# MatiereG.ELECTRON                   -79.6  121.2 -233.7  -79.6   75.7
# MatiereG.ELECTROT                  -122.7   89.3 -237.1 -122.8   -8.3
# MatiereG.IND.BOIS                  -320.2  118.9 -471.4 -320.1 -169.4
# MatiereHIST. GEO.                   -85.8   36.9 -132.8  -85.8  -38.6
# MatiereITALIEN                      141.5  121.8  -12.3  140.9  297.0
# MatiereLET ANGLAI                  -264.9   66.6 -349.8 -264.8 -179.9
# MatiereLET ESPAGN                  -142.8   74.7 -238.7 -142.5  -47.3
# MatiereLET MODERN                   -23.5   39.3  -73.6  -23.6   27.3
# MatiereLET.HIS.GE                  -175.8   48.7 -238.5 -175.7 -113.6
# MatiereLETT CLASS                   -32.3   44.9  -89.5  -32.3   25.4
# MatiereMATH.SC.PH                  -179.1   54.3 -248.0 -179.2 -109.5
# MatiereMATHS                       -112.8   35.4 -157.8 -113.2  -67.5
# MatiereNRC                         -173.6   88.0 -285.2 -174.7  -60.9
# MatierePHILO                        -39.6   44.1  -95.7  -39.7   16.5
# MatierePHY.CHIMIE                  -129.1   42.4 -183.3 -128.7  -75.4
# MatiereS. V. T.                     -77.8   40.5 -129.7  -77.6  -26.1
# MatiereSC.ECO.SOC                  -141.8   46.4 -200.8 -141.2  -82.6
# MatiereSII.EE                       133.6   66.0   48.5  132.8  218.0
# MatiereSII.ING.ME                   -71.8   66.2 -156.2  -71.9   12.5
# MatiereSII.SIN                     -153.6   74.5 -248.4 -153.7  -57.3
# effectif_presents_serie_l             0.0    0.5   -0.6    0.0    0.7
# effectif_presents_serie_es           -0.5    0.4   -1.0   -0.5    0.0
# effectif_presents_serie_s             0.4    0.3    0.0    0.4    0.8
# taux_brut_de_reussite_serie_l        -1.1    0.8   -2.1   -1.1   -0.1
# taux_brut_de_reussite_serie_es       -0.4    1.3   -2.0   -0.4    1.3
# taux_brut_de_reussite_serie_s        -1.2    2.0   -3.7   -1.2    1.4
# taux_reussite_attendu_serie_l        -0.8    2.0   -3.4   -0.8    1.8
# taux_reussite_attendu_serie_es       -3.1    2.4   -6.2   -3.1    0.0
# taux_reussite_attendu_serie_s         0.5    2.9   -3.3    0.5    4.3
# effectif_de_seconde                  -0.1    0.2   -0.4   -0.1    0.1
# effectif_de_premiere                  0.1    0.2   -0.2    0.1    0.4
# taux_acces_brut_seconde_bac           0.2    1.8   -2.0    0.2    2.5
# taux_acces_attendu_seconde_bac       -4.4    2.8   -8.0   -4.4   -0.7
# taux_acces_brut_premiere_bac         -0.1    3.2   -4.2   -0.2    4.0
# taux_acces_attendu_premiere_bac      11.6    6.0    3.9   11.7   19.4
# taux_brut_de_reussite_total_series    2.4    4.0   -2.7    2.3    7.4
# taux_reussite_attendu_total_series    2.1    6.7   -6.4    2.1   10.8
# sigma                               113.9    4.2  108.6  113.8  119.3

summary(mod_lin)

# Coefficients:
#                                      Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                        -218.12367  177.47343  -1.229 0.219832    
# MatiereANGLAIS                     -129.05155   36.33561  -3.552 0.000432 ***
# MatiereARTS PLAST                   -74.37260   90.01997  -0.826 0.409234    
# MatiereBIOCH.BIOL                  -166.66600   60.41503  -2.759 0.006091 ** 
# MatiereBIOTECHNOL                  -209.72334   66.29374  -3.164 0.001687 ** 
# MatiereDOC LYCEES                  -178.34815   47.38864  -3.764 0.000195 ***
# MatiereE. P. S                       -6.74828   42.31645  -0.159 0.873384    
# MatiereECO.GE.COM                   -97.33050   50.23861  -1.937 0.053460 .  
# MatiereECO.GE.CPT                  -111.86961   88.07576  -1.270 0.204826    
# MatiereECO.GE.FIN                   -63.66233   50.17459  -1.269 0.205302    
# MatiereECO.GE.MK                    -63.38475   44.54710  -1.423 0.155614    
# MatiereECO.GE.VEN                  -122.59178   57.76885  -2.122 0.034491 *  
# MatiereEDUCATION                   -142.43964   44.08451  -3.231 0.001344 ** 
# MatiereESPAGNOL                    -123.55147   38.54036  -3.206 0.001464 ** 
# MatiereESTH.COSME                  -200.83485  119.20900  -1.685 0.092882 .  
# MatiereG.ELECTRON                   -80.32113  121.10062  -0.663 0.507576    
# MatiereG.ELECTROT                  -123.11746   88.82992  -1.386 0.166582    
# MatiereG.IND.BOIS                  -320.64146  119.45661  -2.684 0.007597 ** 
# MatiereHIST. GEO.                   -86.18408   36.84018  -2.339 0.019844 *  
# MatiereITALIEN                      143.01874  122.19647   1.170 0.242591    
# MatiereLET ANGLAI                  -265.96430   66.91796  -3.974 8.48e-05 ***
# MatiereLET ESPAGN                  -143.43246   74.76229  -1.919 0.055813 .  
# MatiereLET MODERN                   -23.58659   39.55532  -0.596 0.551343    
# MatiereLET.HIS.GE                  -176.20469   48.99969  -3.596 0.000367 ***
# MatiereLETT CLASS                   -32.58909   45.09827  -0.723 0.470366    
# MatiereMATH.SC.PH                  -179.32415   54.58506  -3.285 0.001116 ** 
# MatiereMATHS                       -113.16043   35.60536  -3.178 0.001606 ** 
# MatiereNRC                         -173.86325   88.25654  -1.970 0.049585 *  
# MatierePHILO                        -39.95855   44.20618  -0.904 0.366628    
# MatierePHY.CHIMIE                  -129.60139   42.47335  -3.051 0.002442 ** 
# MatiereS. V. T.                     -78.17194   40.52933  -1.929 0.054521 .  
# MatiereSC.ECO.SOC                  -142.29302   46.47790  -3.062 0.002363 ** 
# MatiereSII.EE                       133.41524   66.82961   1.996 0.046626 *  
# MatiereSII.ING.ME                   -72.02632   66.30201  -1.086 0.278035    
# MatiereSII.SIN                     -153.65548   73.76822  -2.083 0.037940 *  
# effectif_presents_serie_l             0.04471    0.50345   0.089 0.929289    
# effectif_presents_serie_es           -0.54721    0.38248  -1.431 0.153358    
# effectif_presents_serie_s             0.35921    0.31032   1.158 0.247798    
# taux_brut_de_reussite_serie_l        -1.09851    0.77951  -1.409 0.159604    
# taux_brut_de_reussite_serie_es       -0.36600    1.29415  -0.283 0.777481    
# taux_brut_de_reussite_serie_s        -1.20719    1.99136  -0.606 0.544742    
# taux_reussite_attendu_serie_l        -0.84305    2.07550  -0.406 0.684838    
# taux_reussite_attendu_serie_es       -3.11184    2.44063  -1.275 0.203101    
# taux_reussite_attendu_serie_s         0.50371    2.94972   0.171 0.864502    
# effectif_de_seconde                  -0.11330    0.19297  -0.587 0.557479    
# effectif_de_premiere                  0.07224    0.22322   0.324 0.746399    
# taux_acces_brut_seconde_bac           0.27752    1.75779   0.158 0.874639    
# taux_acces_attendu_seconde_bac       -4.46453    2.84211  -1.571 0.117070    
# taux_acces_brut_premiere_bac         -0.18986    3.24449  -0.059 0.953368    
# taux_acces_attendu_premiere_bac      11.82115    6.06613   1.949 0.052083 .  
# taux_brut_de_reussite_total_series    2.35449    4.02207   0.585 0.558640    
# taux_reussite_attendu_total_series    2.03817    6.75786   0.302 0.763126 


# Le retour de l'inférence bayésienne est la distribution des valeurs qui 
# pourra prendre les coefficients. En revanche, la régression linéaire retour 
# qu'un seule valeur pour chaque coefficient ainsi qu'une valeur de probabilité 
# associée qui permet d'établir un niveau de significativité. Il est observé 
# que si l'intervalle du coefficient de la statistique bayésienne touche le zéro
# ce coefficient n'est pas significatif dans la régression linéaire.

# Q3. Modelisation de "MATHS" et "ANGLAIS":

# Sélection des covariables

db0 = db[,-c(1,2,3,4)]
db1 = db0[,c(2,1,3:ncol(db0))]
db2 = db1[db1$Barre<400,]# îdem filtre inf à 1000 mais pour MATHS
db3 = db2[db2$Matiere == "MATHS", ]
hist(db3$Barre, breaks = 20)
hist(log(db3$Barre), breaks = 20)
db3$Matiere = NULL

# Train et test
set.seed(1234)
ind = sample(1:nrow(db3), size = floor(nrow(db3)*0.9), replace = FALSE)
train3 = db3[ind,]
test3 = db3[-ind,]

# Modèle linéraie MATHS

mod_lin_mat = lm(Barre ~ ., data = train3)
mod_step_mat = step(mod_lin_mat)
summary(mod_step_mat)
y_pred_mat = predict(mod_step_mat, test3)
y = test3$Barre
MSE = mean((y-y_pred_mat)^2)

# > MSE
# [1] 4109.474
# > sqrt(MSE)
# [1] 64.10518#

# Régression bayésienne MATHS

mod_b5 = stan_glm(Barre ~ ., data = train3, cores = 2, seed = 31416, iter = 10000)
save(mod_b5, file = "mod_b5.RData")
load("mod_b5.RData")
pred_b_dist = posterior_predict(mod_b5, test3)
y_pred_b = apply(pred_b_dist, 2, mean)
y_b = test3$Barre
MSEB = mean((y_b-y_pred_b)^2)

# > MSEB
# [1] 1946.262
# > sqrt(MSEB)
# [1] 44.11646

# ANGLAIS

db0 = db[,-c(1,2,3,4)]
db1 = db0[,c(2,1,3:ncol(db0))]
db2 = db1[db1$Barre<1000,]# îdem filtre inf à 1000 mais pour ANGLAIS
db3 = db2[db2$Matiere == "ANGLAIS", ]
hist(db3$Barre)
db3$Matiere = NULL
set.seed(1234)
ind = sample(1:nrow(db3), size = floor(nrow(db3)*0.9), replace = FALSE)
train3 = db3[ind,]
test3 = db3[-ind,]

#Modèle linéraie ANGLAIS

mod_lin_ang = lm(Barre ~ ., data = train3)
mod_step_ang = step(mod_lin_ang)
y_pred_ang = predict(mod_step_ang, test3)
y = test3$Barre
MSE = mean((y-y_pred_ang)^2)

# > MSE
# [1] 10082.4
# > sqrt(MSE)
# [1] 100.4112

# Régression bayésienne ANGLAIS

mod_b6 = stan_glm(Barre ~ ., data = train3, cores = 2, seed = 31416, iter = 10000)
save(mod_b6, file = "mod_b6.RData")
load("mod_b6.RData")
pred_b_dist = posterior_predict(mod_b6, test3)
y_pred_b = apply(pred_b_dist, 2, mean)
y_b = test3$Barre
MSEB = mean((y_b-y_pred_b)^2)

# > MSEB
# [1] 9806.122
# > sqrt(MSEB)
# [1] 99.02586


summary(mod_step_mat)
summary(mod_b5)
summary(mod_step_ang)
summary(mod_b6)


## 2. Loi de Pareto

# Q4. Loi de Pareto pour different valeurs d'alpha

library(Pareto)
valores = c(0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.75, 0.9, 1)
for(i in 1:length(valores)) {
    alpha = valores[i] 
    if(i == 1) plot(dPareto(21:5000, 21, alpha), xlab = "x", ylab = "PDF(x)", col = i, type = "l")
    else points(dPareto(21:5000, 21, alpha), col = i, type = "l")
}
legend("topright", paste("alpha = ",valores,sep=""), col=1:8, lty=1,cex=0.70,bg="#f1f1f1")

#
# Q5. Choix de la loi a priori pour alpha
# le meilleur estimateur d'alpha via le max de vraisemblance

summary(db$Barre)
# m correspond au plus petit valeur de la variable "Barre"

m=21 #Cf. le summary

pareto.MLE <- function(X) {
   n <- length(X)
   umbral <- min(X)
   alfa <- n/sum(log(X)-log(m))
   return( c(umbral,alfa) ) 
}
param = pareto.MLE(X=db1$Barre)
# [1] 21.0000000  0.4502063

hist(db1$Barre, breaks = 20, freq = FALSE)
curve(dPareto(x, param[1], param[2]), col = "red", xlim = c(21,5000), add = TRUE)
#
# On regardant le histogramme de Barre, on peut partir sur l'hypothèse
# que cette variable suit une distribution de Pareto, donc on pourrais supposer
# que la fonction de max de vraisemblance prend cette forme.
# sur la base de la littérature l'a priori aura une distribution Gamma.
#
# https://en.wikipedia.org/wiki/Conjugate_prior
#
# Cette distribution Gamma compte avec deux paramètres alpha et beta
# qui seront estimés via le max de vraisemblance
# https://en.wikipedia.org/wiki/Gamma_distribution
#

x = db1$Barre
N = length(x)

# estimation d'alpha  = k

k = (N * sum(x)) / (N*sum(x*log(x))-(sum(log(x))*sum(x)))
alfa = k

# [1] 0.9490404
# Estimation theta = 1/beta

theta = (1/N^2)*(N*sum(x*log(x))-(sum(log(x))*sum(x)))
beta = 1/theta
beta
# [1] 0.002948104
#
# Q6. Donner la loi à posteriori d'alpha 
# Les paramètres seran les suivants conforme à la litérature
#
c(alfa+N,beta+sum(log(x))-N*log(21))
# [1]  516.949 1146.144
#
# Q7. Intervalle de crédibilité à 95%
#

set.seed(1234)
post_sample = rgamma(5000,alfa+N,beta+sum(log(x))-N*log(21))
post_sample_ord = post_sample[order(post_sample, decreasing = FALSE)]
hist(post_sample)

lim_inf = post_sample_ord[126]
lim_sup = post_sample_ord[5000-126]

# > lim_inf
# [1] 0.4145262
# > lim_sup
# [1] 0.4905078
#
# medianne
#
# > post_sample_ord[2500]
# [1] 0.4511104
#
# Q8. Répetition de l'analyse pour Mathématiques et anglais
#

db_mat = db1[db1$Matiere == "MATHS",]
db_ang = db1[db1$Matiere == "ANGLAIS",]

barre_mat = db_mat$Barre
barre_ang = db_ang$Barre
par(mfrow=c(1,2))
hist(barre_mat, breaks = 20)
hist(barre_ang, breaks = 20)
#
# Calculant alpha pour Mathématiques et anglais
#

pareto.MLE <- function(X) {
   n <- length(X)
   umbral <- min(X)
   alfa <- n/sum(log(X)-log(umbral))
   return( c(umbral,alfa) ) 
}
param_mat = pareto.MLE(X=barre_mat)

# > param_mat
# [1] 21.0000000  0.5045626

param_ang = pareto.MLE(X=barre_ang)

# > param_ang
# [1] 21.0000000  0.4851748
#
# Ajuster la distribution de Pareto pour mathématiques et anglais 


par(mfrow=c(1,2))
hist(barre_mat, breaks = 20, freq = FALSE)
curve(dPareto(x, param_mat[1], param_mat[2]), col = "red", xlim = c(21,5000), add = TRUE)
hist(barre_ang, breaks = 20, freq = FALSE)
curve(dPareto(x, param_ang, param_ang), col = "blue", xlim = c(21,5000), add = TRUE)
#
# Alpha et Beta pour Mathématiques et Anglais
#

x = barre_mat
N_mat = length(x)
N = length(x)

k = (N * sum(x)) / (N*sum(x*log(x))-(sum(log(x))*sum(x)))
alfa_mat = k

# [1] 2.487811
# theta = 1/beta
theta_mat = (1/N^2)*(N*sum(x*log(x))-(sum(log(x))*sum(x)))
beta_mat = 1/theta_mat

# [1] 0.01323996

x = barre_ang
N_ang = length(x)
N = length(x)

# alfa  = k

k = (N * sum(x)) / (N*sum(x*log(x))-(sum(log(x))*sum(x)))
alfa_ang = k

# [1] 1.356126
# theta = 1/beta

theta_ang = (1/N^2)*(N*sum(x*log(x))-(sum(log(x))*sum(x)))
beta_ang = 1/theta_ang

# [1] 0.00598782


c(alfa_mat+N_mat,beta_mat+sum(log(barre_mat))-N_mat*log(21))

# [1]  61.48781 116.94620

c(alfa_ang+N_ang,beta_ang+sum(log(barre_ang))-N_ang*log(21))

# [1]  53.35613 107.18385
#
# Intervalle de crédibilité
#


set.seed(1234)
post_sample_mat = rgamma(5000,alfa_mat+N_mat,beta_mat+sum(log(barre_mat))-N_mat*log(21))
post_sample_mat_ord = post_sample_mat[order(post_sample_mat, decreasing = FALSE)]
hist(post_sample_mat)

lim_inf_mat = post_sample_mat_ord[126]
lim_sup_mat = post_sample_mat_ord[5000-126]

# > lim_inf_mat
# [1] 0.406576
# > lim_sup_mat
# [1] 0.6633221


#
# Medianne
post_sample_mat_ord[2500]
# [1] 0.5245885
#
# Pour Anglais

set.seed(1234)
post_sample_ang = rgamma(5000,alfa_ang+N_ang,beta_ang+sum(log(barre_ang))-N_ang*log(21))
post_sample_ang_ord = post_sample_ang[order(post_sample_ang, decreasing = FALSE)]
hist(post_sample_ang)
 
lim_inf_ang = post_sample_ang_ord[126]
lim_sup_ang = post_sample_ang_ord[5000-126]

# > lim_inf_ang
# [1] 0.3769371
# > lim_sup_ang
# [1] 0.6392664
#
# Medianne 
post_sample_ang_ord[2500]

#[1] 0.4964046
