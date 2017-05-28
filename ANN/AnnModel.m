function [finalSensitivity, finalSpecificity, finalAccuracy, finalEfficiency] = AnnModel(x,y,folds,neurons, epochs)

    sampleSize= size(x,1);
    neuronsSize = size(neurons,2);
    networksY = zeros(sampleSize,2);
    epochsSize = size(epochs,2);
    
    sensitivity=zeros(neuronsSize,folds);
    specificity=zeros(neuronsSize,folds);
    accuracy=zeros(neuronsSize,folds);
    efficiency=zeros(neuronsSize,folds);

    index = 1;
    
    for epoch = 1:epochsSize
        for neuron=1:neuronsSize
            for i=1:sampleSize
                if(y(i)==1)
                    networksY(i,1)=1;
                else
                    networksY(i,2)=1;
                end
            end
            for fold=1:folds

                %%% partition %%%

                rng('default');
                particion=cvpartition(sampleSize,'Kfold',folds);
                Xtrain=x(particion.training(fold),:);
                Xtest=x(particion.test(fold),:);
                Ytrain = networksY(particion.training(fold),:);
                [~,Ytest]=max(networksY(particion.test(fold),:),[],2);

                %%% Train %%%

                Modelo=TrainRNA(Xtrain,Ytrain,neurons(neuron), epochs(epoch));

                %%% Validation %%%

                Yesti=testRNA(Modelo,Xtest);
                [~,Yesti]=max(Yesti,[],2);

                FN=sum(Yesti==2 & Yesti~=Ytest);
                FP=sum(Yesti==1 & Yesti~=Ytest);
                TP=sum(Yesti==Ytest & Yesti==1);
                TN=sum(Yesti==Ytest)-TP;
                sensitivity(index,fold)=(TP)/(TP+FN);
                specificity(index,fold)=(TN)/(TN+FP);
                accuracy(index,fold)=(TP)/(TP+FP);
                efficiency(index,fold)=(TP+TN)/(TP+TN+FP+FN); 
                texto=['Epocas: ',num2str(epochs(epoch)),'  Neuronas: ', num2str(neurons(neuron)),'  fold: ',num2str(fold)];
                disp(texto);
                index = index + 1;
            end
            
        end
    end
    finalEfficiency=zeros((neuronsSize*epochsSize),2);
    finalSpecificity=zeros((neuronsSize*epochsSize),2);
    finalSensitivity=zeros((neuronsSize*epochsSize),2);
    finalAccuracy=zeros((neuronsSize*epochsSize),2);
    for i=1:neuronsSize*epochsSize
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