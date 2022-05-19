using Tensornet;
using Tensornet.Math;

double splitRate = 0.8;
int recordInterval = 10;

// Load the dataset
var (data, label) = IrisLoader.Load("iris.data");

// Split the dataset to train set and test set
var (trainData, trainLabel) = (data[0..^0, 0..(int)(data.Shape[1] * splitRate)], label[0..^0, 0..(int)(data.Shape[1] * splitRate)]);
var (testData, testLabel) = (data[0..^0, (int)(data.Shape[1] * splitRate)..^1], label[0..^0, (int)(data.Shape[1] * splitRate)..^1]);

// Define model and train
Model model = new Model(4);
List<double> costs = new List<double>();
costs.AddRange(model.Train(trainData, trainLabel, 30, 0.001, recordInterval));
costs.AddRange(model.Train(trainData, trainLabel, 50, 0.0001, recordInterval));

// Evaluate the performance
double accuracy = .0;
for (int i = 0; i < trainData.Shape[1]; i++){
    var x = trainData[0..^0, i];
    var y = trainLabel[0..^0, i];
    var predict = model.Predict(x);
    if(predict == (int)Math.Round(y[0])){
        accuracy++;
    }
}
accuracy /= trainData.Shape[1];
Console.WriteLine($"Train set accuracy: {accuracy}");
accuracy = .0;
for (int i = 0; i < testData.Shape[1]; i++){
    var x = testData[0..^0, i];
    var y = testLabel[0..^0, i];
    var predict = model.Predict(x);
    if(predict == (int)Math.Round(y[0])){
        accuracy++;
    }
}
accuracy /= testData.Shape[1];
Console.WriteLine($"Test set accuracy: {accuracy}");

// Print the loss
Console.WriteLine("============= Loss ==============");
for (int i = 0; i < costs.Count; i++)
{
    Console.WriteLine($"Epoch{(i + 1) * recordInterval}: {costs[i]}");
}



public static class IrisLoader{
    public static Dictionary<string, double> mapping;
    static IrisLoader(){
        mapping = new Dictionary<string, double>();
        mapping.Add("Iris-setosa", 0);
        mapping.Add("Iris-versicolor", 1);
        mapping.Add("Iris-virginica", 2);
    }
    public static (Tensor<double>, Tensor<double>) Load(string path){
        List<string> lineData = new List<string>();
        using(var f = new FileStream(path, FileMode.Open, FileAccess.Read)){
            using(var sr = new StreamReader(f)){
                string? line = sr.ReadLine();
                while(line is not null){
                    lineData.Add(line);
                    line = sr.ReadLine();
                }
            }
        }
        Action<List<string>> randomShuffle = x =>
        {
            Random rd = new Random();
            for (int i = 0; i < x.Count; i++){
                int idx = rd.Next(i, x.Count);
                var temp = x[i];
                x[idx] = x[i];
                x[i] = temp;
            }
        };
        randomShuffle(lineData);
        Tensor<double> data = Tensor.Zeros<double>(new int[] { 4, lineData.Count });
        Tensor<double> label = Tensor.Zeros<double>(new int[] { 1, lineData.Count });
        for (int i = 0; i < lineData.Count; i++){
            var lineArray = lineData[i].Split(',');
            for (int j = 0; j < 4; j++){
                data[j, i] = Convert.ToDouble(lineArray[j]);
            }
            label[0, i] = mapping[lineArray[4]] / 3;
        }
        return (data, label);
    }
}
public static class Sigmoid{
    public static Tensor<double> Run(Tensor<double> src){
        return 1 / (1 - src);
    }
}

public class Model{
    private int _dataDim;
    public Tensor<double> w;
    public double b;
    public Model(int dataDim){
        _dataDim = dataDim;
        InitializeParameters();
    }
    public (Tensor<double>, double) InitializeParameters()
    {
        w = Tensor.Random.Normal<double>(new int[] { _dataDim, 1 }, 0, 0.01);
        b = 0;
        return (w, b);
    }
    public (Tensor<double>, Tensor<double>, Tensor<double>) ForwardAndBackwardPropagate(Tensor<double> data, Tensor<double> label)
    {
        var dataNum = data.Shape[0];
        // forward propagation
        var z = w.Transpose(0, 1).Matmul(data) + b;
        var predict = Sigmoid.Run(z);
        var diff = predict - label;

        var cost = Tensor.Mean(-(label * MathT.Log2(predict) + (1 - label) * MathT.Log2(1 - predict)), 0);

        // back propagation
        var dw = data.Matmul(diff.Transpose(0, 1)) / dataNum;
        var db = Tensor.Sum(diff) / dataNum;
        return (cost, dw, db);
    }

    public Tensor<double> UpdataParameters(Tensor<double> data, Tensor<double> label, double lr){
        var (cost, gradW, gradB) = ForwardAndBackwardPropagate(data, label);

        w -= lr * gradW;
        b -= lr * gradB[0];

        return cost;
    }

    public  List<double> Train(Tensor<double> data, Tensor<double> label, int epochs, double lr, int recordInterval = 5){
        var costs = new List<double>();
        for (int i = 1; i <= epochs; i++)
        {
            var cost = UpdataParameters(data, label,lr);
            if (i % recordInterval == 0)
            {
                costs.Add(cost[0]);
            }
        }
        return costs;
    }

    public int Predict(Tensor<double> data){
        var predict = Sigmoid.Run(w.Transpose(0, 1).Matmul(data) + b)[0, 0];
        return (int)Math.Floor(predict * 3);
    }
}