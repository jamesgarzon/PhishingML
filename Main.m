%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PHISHING - FINAL PROJECT        %
% Developed by:                   %
% James Garzón                    %
% Yoiner Gomez                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

addpath(genpath('KNN'))

load('dataset.mat');
X=dataset(:,1:30);
Y=dataset(:,end);
folds = 2;

option=input('MENU PRINCIPAL\n1. k vecinos más cercanos \n2. Random Forest \n3. Máquina de Soporte Vectorial \n4. Funciones Discriminantes Gaussianas \n5. Redes neuronales \n6. Discriminante de Fisher \n7. Correlación de Pearson \n8. SFS \nSeleccione una opción:  ');

if option == 1
   
   neighbors=[1,3,5,7,9,11,13];
   [sensitivity, specificity, precision, efficiency] = KnnModel(X,Y,folds,neighbors);
   
end