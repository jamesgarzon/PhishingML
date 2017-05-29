function y = GDATrain(Xentrenamiento, Xvalidacion, Yentrenamiento, type)

    [y, ~] = classify(Xentrenamiento, Xvalidacion, Yentrenamiento,char(type));
    
end