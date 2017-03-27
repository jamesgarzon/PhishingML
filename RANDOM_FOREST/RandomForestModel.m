function [finalSensitivity, finalSpecificity, finalaccuracy, finalEfficiency] = RandomForestModel(x,y,folds,trees)
    
    sampleSize= size(x,1);
    treeSize = size(trees,2);
    
    sensitivity=zeros(treeSize,folds);
    specificity=zeros(treeSize,folds);
    accuracy=zeros(treeSize,folds);
    efficiency=zeros(treeSize,folds);
    
    for tree=1:treeSize
        for fold=1:folds
            rng('default');
            partition=cvpartition(sampleSize,'Kfold',folds);
            Xtrain=x(partition.training(fold),:);
            Xtest=x(partition.test(fold),:);
            Ytrain=y(partition.training(fold));
            Ytest=y(partition.test(fold));

            modeloRF=TrainFOREST(trees(tree),Xtrain,Ytrain');
            Yesti=TestFOREST(modeloRF,Xtest);
            FN=(sum(Yesti~=Ytest))-(sum(Yesti==-1 & Yesti~=Ytest));
            FP=(sum(Yesti~=Ytest))-(sum(Yesti==1 & Yesti~=Ytest));
            TP=sum(Yesti==Ytest & Yesti==-1);
            TN=sum(Yesti==Ytest)-TP;
            sensitivity(tree,fold)=(TP)/(TP+FN);
            specificity(tree,fold)=(TN)/(TN+FP);
            accuracy(tree,fold)=(TP)/(TP+FP);
            efficiency(tree,fold)=(TP+TN)/(TP+TN+FP+FN);
            texto=['Arboles = ', num2str(trees(tree)),' fold: ',num2str(fold)];
            disp(texto);

        end
    end
    finalSensitivity=zeros(treeSize,2);
    finalSpecificity=zeros(treeSize,2);
    finalaccuracy=zeros(treeSize,2);
    finalEfficiency=zeros(treeSize,2);
       
    for i=1:treeSize
        finalEfficiency(i,1)=mean(efficiency(i,:));
        finalEfficiency(i,2)=std(efficiency(i,:));
        finalSpecificity(i,1)=mean(specificity(i,:));
        finalSpecificity(i,2)=std(specificity(i,:));
        finalSensitivity(i,1)=mean(sensitivity(i,:));
        finalSensitivity(i,2)=std(sensitivity(i,:));
        finalaccuracy(i,1)=mean(accuracy(i,:));
        finalaccuracy(i,2)=std(accuracy(i,:));
    end
    
end

