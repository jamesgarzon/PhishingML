%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PHISHING - FINAL PROJECT        %
% Developed by:                   %
% James Garzón                    %
% Yoiner Gomez                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

addpath(genpath('KNN'))
addpath(genpath('RANDOM_FORES5T'))
addpath(genpath('SVM'))
addpath(genpath('RNA'))
addpath(genpath('GDA'))
addpath(genpath('FISHER'))
addpath(genpath('SFS'))
addpath(genpath('PCA'))
load('dataset.mat');
X=dataset(:,1:30);
Y=dataset(:,end);
folds = 10;

option=input('MENU PRINCIPAL\n1. k vecinos más cercanos \n2. Random Forest \n3. Máquina de Soporte Vectorial \n4. Funciones Discriminantes Gaussianas \n5. Redes neuronales \n6. Discriminante de Fisher \n7. Selección SFS \n8. PCA  \nSeleccione una opción:  ');

if option == 1
   neighbors=[1,3,5,7,9,11,13];
   [sensitivity_knn, specificity_knn, accuracy_knn, efficiency_knn] = KnnModel(X,Y,folds,neighbors);
elseif option == 2
    trees=[50 100 250 500 750 1000];
    [sensitivity, specificity, accuracy, efficiency] = RandomForestModel(X,Y,folds,trees);
elseif option == 3
    gamma=[0.01 0.1 1 10 100];
    box=[0.01 0.1 1 10 100];
    [sensitivity, specificity, accuracy, efficiency] = SvmModel(X,Y,folds,gamma,box);
elseif option == 4
    %With quadractic do not converge
    %functionType ={'linear','diaglinear','quadratic'}; 
    functionType ={'linear','diaglinear'};
    [sensitivity_gda, specificity_gda, accuracy_gda, efficiency_gda] = GDAModel(X,Y,folds,functionType);
elseif option == 5
     epochs=[100 400 800 1000];
     neurons=[50 55 60 65 70];
     [sensitivity, specificity, accuracy, efficiency] = RNAModel(X,Y,folds,neurons, epochs);
     retults = [sensitivity, specificity, accuracy, efficiency];
elseif option == 6
    [coeff, corr] = FisherSelection(X,Y);
    stem(coeff);
    text = ['Indice de Fisher: ', num2str(coeff)];
    disp(text);
elseif option == 7
    features = SFSSelection(X,Y);
elseif option == 8
    VarianceLevel = 95;
    [coefCompPrincipals, numCompAdmiteds] = extractionPCA(X,VarianceLevel );
end