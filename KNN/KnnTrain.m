function Yesti = KnnTrain(Xval,Xent,Yent,k)
    N=size(Xent,1);
    M=size(Xval,1);
    Yesti=zeros(M,1);
    dis=zeros(N,1);

    for j=1:M
        dis=distance(Xent, Xval(j,:));
        [dis, sortIndexes]= sort(dis);
        temp = Yent(sortIndexes(1:k));
        Yesti(j)= mode (temp);
    end
    
end