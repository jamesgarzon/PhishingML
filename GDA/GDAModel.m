function [finalSensitivity, finalSpecificity, finalAccuracy, finalEfficiency] = GDAModel(x,y,folds,functionTypes)
 
    sampleSize= size(x,1);
    functionTypesSize = size(functionTypes,2);
    
    sensitivity=zeros(functionTypesSize,folds);
    specificity=zeros(functionTypesSize,folds);
    accuracy=zeros(functionTypesSize,folds);
    efficiency=zeros(functionTypesSize,folds);
    
    
    
    for functionType=1:functionTypesSize
        for fold=1:folds
            rng('default');
            partition=cvpartition(sampleSize,'Kfold',folds);
            Xtrain=x(partition.training(fold),:);
            Xtest=x(partition.test(fold),:);
            Ytrain=y(partition.training(fold));
            Ytest=y(partition.test(fold));

            %Yesti=KnnTrain(Xtest,Xtrain,Ytrain,neighbors(neighbor));
            %modeloRF=TrainFOREST(trees(tree),Xtrain,Ytrain');
            %Yesti=testFOREST(modeloRF,Xtest);
            %Yesti=GDATrain(Xtrain,Ytrain,functionTypes(functionType));
            Yesti = GDATrain(Xtest, Xtrain, Ytrain, functionTypes(functionType));
            
            FN=(sum(Yesti~=Ytest))-(sum(Yesti==-1 & Yesti~=Ytest));
            FP=(sum(Yesti~=Ytest))-(sum(Yesti==1 & Yesti~=Ytest));
            TP=sum(Yesti==Ytest & Yesti==-1);
            TN=sum(Yesti==Ytest)-TP;
            sensitivity(functionType,fold)=(TP)/(TP+FN);
            specificity(functionType,fold)=(TN)/(TN+FP);
            accuracy(functionType,fold)=(TP)/(TP+FP);
            efficiency(functionType,fold)=(TP+TN)/(TP+TN+FP+FN);
            message=['Tipo de funcion: ', functionTypes(functionType),' fold: ',num2str(fold)];
            disp(message);

        end
    end
    
    finalSensitivity=zeros(functionTypesSize,2);
    finalSpecificity=zeros(functionTypesSize,2);
    finalAccuracy=zeros(functionTypesSize,2);
    finalEfficiency=zeros(functionTypesSize,2);
       
    for i=1:functionTypesSize
        finalEfficiency(i,1)=mean(efficiency(i,:));
        finalEfficiency(i,2)=std(efficiency(i,:));
        finalSpecificity(i,1)=mean(specificity(i,:));
        finalSpecificity(i,2)=std(specificity(i,:));
        finalSensitivity(i,1)=mean(sensitivity(i,:));
        finalSensitivity(i,2)=std(sensitivity(i,:));
        finalAccuracy(i,1)=mean(accuracy(i,:));
        finalAccuracy(i,2)=std(accuracy(i,:));
    end
end