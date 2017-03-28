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
addpath(genpath('RANDOM_FOREST'))
addpath(genpath('SVM'))

load('dataset.mat');
X=dataset(:,1:30);
Y=dataset(:,end);
folds = 10;

option=input('MENU PRINCIPAL\n1. k vecinos más cercanos \n2. Random Forest \n3. Máquina de Soporte Vectorial \n4. Funciones Discriminantes Gaussianas \n5. Redes neuronales \n6. Discriminante de Fisher \n7. Correlación de Pearson \n8. SFS \nSeleccione una opción:  ');

if option == 1
   neighbors=[1,3,5,7,9,11,13];
   [sensitivity, specificity, accuracy, efficiency] = KnnModel(X,Y,folds,neighbors);
elseif option == 2
    trees=[50 100 250 500 750 1000];
    [sensitivity, specificity, accuracy, efficiency] = RandomForestModel(X,Y,folds,trees);
elseif option == 3
    gamma=[0.01 0.1 1 10 100];
    box=[0.01 0.1 1 10 100];
    [sensitivity, specificity, accuracy, efficiency] = SvmModel(X,Y,folds,gamma,box);
end