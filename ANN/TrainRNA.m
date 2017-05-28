function Modelo = TrainRNA(X,Y,NeuronSize, epochs)

%hiddenLayerSize = NeuronSize;
%net = patternnet(hiddenLayerSize);
%Modelo = train(net,X',Y');

net = feedforwardnet(NeuronSize);
net.trainParam.epochs = epochs; % Define el número de épocas
Modelo = train(net,X',Y');

end
