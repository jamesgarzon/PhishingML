function Salida = TestFOREST(Modelo,X)

	Salida = predict(Modelo,X);
    Salida = str2double(Salida);

end
