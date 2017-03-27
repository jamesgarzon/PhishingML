function [finalSensitivity, finalSpecificity, finalPrecision, finalEfficiency] = KnnModel(x,y,folds,neighbors)
    
    sampleSize= size(x,1);
    neighborsSize = size(neighbors,2);
    
    sensitivity=zeros(neighborsSize,folds);
    specificity=zeros(neighborsSize,folds);
    precision=zeros(neighborsSize,folds);
    efficiency=zeros(neighborsSize,folds);
    
    for neighbor=1:neighborsSize
        for fold=1:folds
            rng('default');
            partition=cvpartition(sampleSize,'Kfold',folds);
            Xtrain=x(partition.training(fold),:);
            Xtest=x(partition.test(fold),:);
            Ytrain=y(partition.training(fold));
            Ytest=y(partition.test(fold));

            Yesti=vecinosCercanos(Xtest,Xtrain,Ytrain,neighbors(neighbor));
            FN=(sum(Yesti~=Ytest))-(sum(Yesti==-1 & Yesti~=Ytest));
            FP=(sum(Yesti~=Ytest))-(sum(Yesti==1 & Yesti~=Ytest));
            TP=sum(Yesti==Ytest & Yesti==-1);
            TN=sum(Yesti==Ytest)-TP;
            sensitivity(neighbor,fold)=(TP)/(TP+FN);
            specificity(neighbor,fold)=(TN)/(TN+FP);
            precision(neighbor,fold)=(TP)/(TP+FP);
            efficiency(neighbor,fold)=(TP+TN)/(TP+TN+FP+FN);
            message=['vecinos: ', num2str(neighbors(neighbor)),' fold: ',num2str(fold)];
            disp(message);

        end
    end
    
    finalSensitivity=zeros(neighborsSize,2);
    finalSpecificity=zeros(neighborsSize,2);
    finalPrecision=zeros(neighborsSize,2);
    finalEfficiency=zeros(neighborsSize,2);
       
    for i=1:neighborsSize
        finalEfficiency(i,1)=mean(efficiency(i,:));
        finalEfficiency(i,2)=std(efficiency(i,:));
        finalSpecificity(i,1)=mean(specificity(i,:));
        finalSpecificity(i,2)=std(specificity(i,:));
        finalSensitivity(i,1)=mean(sensitivity(i,:));
        finalSensitivity(i,2)=std(sensitivity(i,:));
        finalPrecision(i,1)=mean(precision(i,:));
        finalPrecision(i,2)=std(precision(i,:));
    end
end