function features = SFSSelection(X,Y)
    partition = cvpartition(Y, 'k', 10);
    features = sequentialfs(@functionForest, X, Y, 'cv', partition);
end

