function Modelo = TrainRNA(X,Y,NeuronSize, epochs)

hiddenLayerSize = NeuronSize;
net = patternnet(hiddenLayerSize);
net.trainParam.epochs = epochs;
Modelo = train(net,X',Y');

end
