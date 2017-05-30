function [coefCompPrincipales, numCompAdmitidos, puntosUmbral,ejeComponentes] = extractionPCA(X,umbralPorcentajeDeVarianza )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [coefCompPrincipales,scores,covarianzaEigenValores,~,porcentajeVarianzaExplicada,~] = pca(X);
        
        numVariables = length(covarianzaEigenValores);
        numCompAdmitidos = 0;
        
        porcentajeVarianzaAcumulada = zeros(numVariables,1);
        puntosUmbral = ones(numVariables,1)*umbralPorcentajeDeVarianza;
        ejeComponentes = 1:numVariables;
        
        for k=1:numVariables
            
            porcentajeVarianzaAcumulada(k) = sum(porcentajeVarianzaExplicada(1:k));
            
            if (sum(porcentajeVarianzaExplicada(1:k)) >= umbralPorcentajeDeVarianza) && (numCompAdmitidos == 0)
                numCompAdmitidos = k;
            end
        end
        %{
        aux = Xtrain*coefCompPrincipales;
        Xtrain = aux(:,1:numCompAdmitidos);
        
        aux = Xtest*coefCompPrincipales;
        Xtest = aux(:,1:numCompAdmitidos);
        %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

