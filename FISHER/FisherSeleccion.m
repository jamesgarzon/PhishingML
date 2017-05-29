function coef = FisherSeleccion( X,Y )

X = zscore(X);
    
    alpha = 0.05;
    [correlacion,p]= corrcoef([X,Y],'alpha',alpha);
    
    indicesClase0 = find(Y == 1);
    indicesClase1 = find(Y == -1);
    
    mediaClase0 = mean(X(indicesClase0,:) ,1);
    mediaClase1 = mean(X(indicesClase1,:) ,1);
    
    media = [mediaClase0; mediaClase1];
    
    varClase0 = var(X(indicesClase0,:) ,1);
    varClase1 = var(X(indicesClase1,:) ,1);
    
    varianza = [varClase0;varClase1];
    
    coef = zeros(1,30);
    
    for i=1:2
        for j=1:2
            if (j ~= i)
                numerador = (media(i,:) - media(j,:)).^2;
                denominador = varianza(i,:) + varianza(j,:);
                coef = coef + (numerador./denominador);
            end
        end
    end
    
coef = coef / max(coef);
end


