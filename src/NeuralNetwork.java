
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    List<FeedForwardLayer> layers;

    public NeuralNetwork(){
        layers = new ArrayList<>();
    }

    public void addLayer(FeedForwardLayer feedForwardLayer){
        layers.add(feedForwardLayer);
    }

    public void train(int[][] trainingSetFeatures, int[] trainingSetLabels){

        int epochs = 0;
        final double ACCEPTABLE_ERROR = 0.001;
        boolean ACCEPTABLE;
        do{
            for (int i = 0; i < trainingSetFeatures.length; i++){
                for (int j = 0; j < trainingSetFeatures[i].length; j++){
                    layers.get(0).getNeurons()[j].setOutput(trainingSetFeatures[i][j]);
                }
                layers.get(layers.size() - 1).getNeurons()[0].setOutput(trainingSetLabels[i]);
            }
            feedForward();
            ACCEPTABLE = backPropagate(, ACCEPTABLE_ERROR);
            epochs++;


        } while(epochs <= 500 && !ACCEPTABLE);

        for (int i = 0; i < trainingSetFeatures.length; i++){
            for (int j = 0; j < trainingSetFeatures.length; j++){
                layers.get(0).getNeurons()[i].getWeights();
            }
        }

    }

    public void feedForward(){
        for (FeedForwardLayer layer : layers){

            if (layer.isInput()) continue;

            else if (layer.isHidden() || layer.isOutput()){
                for (int j = 0; j < layer.getNEURONS_SIZE(); j++){
                    double outputSum = 0;
                    for (int k = 0; k < layer.getPrevious().getNEURONS_SIZE(); k++){
                        outputSum+= calculateOutput(layer.getPrevious().getNeurons()[k].getOutput(), layer.getPrevious().getNeurons()[k].getWeights()[j]);
                    }
                    layer.getNeurons()[j].setOutput(layer.getActivationFunction().activateActivationFunction(outputSum));
                }
            }
        }
    }

    public boolean backPropagate(int actualOutput, double acceptableError){
        for (int i = layers.size() - 1; i >= 0; i--){

            if (layers.get(i).isOutput()){
                double MSE = calculateMeanSquareError(layers.get(i).getNeurons(), actualOutput);
                if (MSE <= acceptableError) return true;
                for (int j = 0; j < layers.get(i).getNEURONS_SIZE(); j++){
                    layers.get(i).getNeurons()[j].setError(calculateOutputLayerError(layers.get(i).getNeurons()[j].getOutput(), actualOutput));
                }
            } else if (layers.get(i).isHidden()){
                calculateHiddenLayerError(layers.get(i));
            }

        }
        return false;
    }

    public double calculateOutput(double input, double weight){
        return input * weight;
    }

    public double calculateMeanSquareError(Neuron[] neurons, int actualOutput){
        double MSE = 0;
        for (Neuron neuron : neurons) {
            MSE += Math.pow((neuron.getOutput() - actualOutput), 2);
        }
        return MSE/2;
    }

    public double calculateOutputLayerError(double predictedOutput, int actualOutput){
        return (predictedOutput - actualOutput) * predictedOutput * (1 - predictedOutput);
    }

    public void calculateHiddenLayerError(FeedForwardLayer layer){
        for (int i = 0; i < layer.getNEURONS_SIZE(); i++){
            double neuronError = 0;
            for (int j = 0; j < layer.getNext().getNEURONS_SIZE(); j++){
                neuronError+= layer.getNext().getNeurons()[j].getError() * layer.getNeurons()[i].getWeights()[j];
            }
            neuronError = neuronError * layer.getNeurons()[i].getOutput() * (1 - layer.getNeurons()[i].getOutput());
            layer.getNeurons()[i].setError(neuronError);
        }
    }



}
