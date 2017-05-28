function [finalSensitivity, finalSpecificity, finalaccuracy, finalEfficiency] = SvmModel(x,y,folds,gamma,box)

    sampleSize= size(x,1);

    sensitivity=zeros(5,folds);
    specificity=zeros(5,folds);
    accuracy=zeros(5,folds);
    efficiency=zeros(5,folds);

  for boxind=1:5
    for gammaind=1:5
        for fold=1:folds
            rng('default');
            particion=cvpartition(sampleSize,'Kfold',folds);
            Xtrain=x(particion.training(fold),:);
            Xtest=x(particion.test(fold),:);
            Ytrain=y(particion.training(fold));
            Ytest=y(particion.test(fold));

            Ytrain1=Ytrain;
            Ytrain1(Ytrain~=1)=-1;
            modelo1=entrenarSVM(Xtrain,Ytrain1,'classification',box(boxind),gamma(gammaind));
            Ytrain2=Ytrain;
            Ytrain2(Ytrain~=-1)=-1;
            Ytrain2(Ytrain==-1)=1;
            modelo2=entrenarSVM(Xtrain,Ytrain2,'classification',box(boxind),gamma(gammaind));
            [~,Yest1]=testSVM(modelo1,Xtest);
            [~,Yest2]=testSVM(modelo2,Xtest);
            [~,Yesti] =max([Yest1,Yest2],[],2);
            MatrizConfusion=zeros(2,2);
            for i=1:size(Xtest,1)
                posTest= 1;
                if Ytest(i)== -1
                    posTest = 2;
                end
                MatrizConfusion(Yesti(i),posTest) = MatrizConfusion(Yesti(i),posTest) + 1;
            end
            TP=MatrizConfusion(2,2);
            TN=MatrizConfusion(1,1);
            FN=MatrizConfusion(1,2);
            FP=MatrizConfusion(2,1);
            sensitivity(gammaind,fold)=(TP)/(TP+FN);
            specificity(gammaind,fold)=(TN)/(TN+FP);
            accuracy(gammaind,fold)=(TP)/(TP+FP);
            efficiency(gammaind,fold)=(TP+TN)/(TP+TN+FP+FN);

            texto=['Gamma = ', num2str(gamma(gammaind)),' fold: ',num2str(fold), ' Box: ',num2str(box(boxind))];
            disp(texto);

        end
    end
    finalEfficiency=zeros(5,2);
    finalSpecificity=zeros(5,2);
    finalSensitivity=zeros(5,2);
    finalaccuracy=zeros(5,2);
    for i=1:5
        finalEfficiency(i,1)=mean(efficiency(i,:));
        finalEfficiency(i,2)=std(efficiency(i,:));
        finalSpecificity(i,1)=mean(specificity(i,:));
        finalSpecificity(i,2)=std(specificity(i,:));
        finalSensitivity(i,1)=mean(sensitivity(i,:));
        finalSensitivity(i,2)=std(sensitivity(i,:));
        finalaccuracy(i,1)=mean(accuracy(i,:));
        finalaccuracy(i,2)=std(accuracy(i,:));
    end

    texto1=['SVM/RESULTS/eficienciaFinalsvm',num2str(boxind),'.mat'];
    texto2=['SVM/RESULTS/especificidadFinalsvm',num2str(boxind),'.mat'];
    texto3=['SVM/RESULTS/sensibilidadFinalsvm',num2str(boxind),'.mat'];
    texto4=['SVM/RESULTS/precisionFinalsvm',num2str(boxind),'.mat'];
    save(texto1,'finalEfficiency');
    save(texto2,'finalSpecificity');
    save(texto3,'finalSensitivity');
    save(texto4,'finalaccuracy');

  end
end
