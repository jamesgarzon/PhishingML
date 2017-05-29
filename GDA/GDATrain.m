function y = GDATrain(Xentrenamiento, Xvalidacion, Yentrenamiento, type)
   % type = char(tipo)

   %isAstring = isstring(type)
    %if 1 == strcmp(tipo,'linear')
        [y, err] = classify(Xentrenamiento, Xvalidacion, Yentrenamiento,char(type));
    %elseif 1 == strcmp(tipo,'diaglinear')
    %    [y, err] = classify(Xentrenamiento, Xvalidacion, Yentrenamiento,'diaglinear');
    %else
    %    [y, err] = classify(Xentrenamiento, Xvalidacion, Yentrenamiento,'diagquadratic');
    %end
    
end