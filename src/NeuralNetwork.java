import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNetwork {

    private List<FeedForwardLayer> layers;
    private double MSE = Double.POSITIVE_INFINITY;
    private double[][][] savedWeights;

    private double[] trainingSetFeaturesMean;
    private double[] trainingSetFeaturesStandardDev;

    private double[] testingSetFeaturesMean;
    private double[] testingSetFeaturesStandardDev;

    private double[] predictionSampleMeanAndStandardDev; // {MEAN, STANDARD DEV}
    private double learningRate = 0.001;
    private double ACCEPTABLE_ERROR = 0.001;
    private int NUMBER_OF_EPOCHS = 50000;

    public NeuralNetwork(){
        layers = new ArrayList<>();
    }

    public void addLayer(FeedForwardLayer feedForwardLayer){
        int layersSize = layers.size();
        if (layersSize == 0){
            feedForwardLayer.setPrevious(null);
        } else{
            feedForwardLayer.setPrevious(layers.get(layersSize - 1));
            layers.get(layersSize - 1).setNext(feedForwardLayer);
        }
        feedForwardLayer.setNext(null);
        layers.add(feedForwardLayer);
    }

    public void train(int[][] trainingSetFeatures, int[] trainingSetLabels) throws IOException {
        int epochs = 0;
//        final double ACCEPTABLE_ERROR = 0.001;
        do{
            for (int i = 0; i < trainingSetFeatures.length; i++){
                for (int j = 0; j < trainingSetFeatures[i].length; j++){
                    layers.get(0).getNeurons()[j].setOutput(
                            (trainingSetFeatures[i][j] - trainingSetFeaturesMean[i]) / trainingSetFeaturesStandardDev[i]); // Standardization
                }
                feedForward();
                backPropagate(trainingSetLabels[i], ACCEPTABLE_ERROR);
                epochs++;
            }

        } while(epochs <= NUMBER_OF_EPOCHS && MSE <= ACCEPTABLE_ERROR);
    }

    public void feedForward(){
        for (FeedForwardLayer layer : layers){
            if (layer.isInput()) continue;

            if (layer.isHidden() || layer.isOutput()){
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

    private double calculateOutput(double input, double weight){
        return input * weight;
    }

    public void backPropagate(int actualOutput, double acceptableError) throws IOException {
        for (int i = layers.size() - 1; i >= 0; i--){
            if (layers.get(i).isOutput()){
                double MSE = calculateMeanSquareError(layers.get(i).getNeurons(), actualOutput);
                this.MSE = this.MSE == Double.POSITIVE_INFINITY? MSE : this.MSE;
                if (this.MSE >= MSE){
                    this.MSE = MSE;
                    saveWeights();
                }

                if (MSE <= acceptableError) return;
                for (int j = 0; j < layers.get(i).getNEURONS_SIZE(); j++){
                    double error = calculateOutputLayerError(layers.get(i).getNeurons()[j].getOutput(), actualOutput);
                    layers.get(i).getNeurons()[j].setError(error);
                }
            } else if (layers.get(i).isHidden()){
                calculateHiddenLayerError(layers.get(i));
            }
        }

        for (int i = 0; i < layers.size() - 1; i++){
            adjustLayerWeights(layers.get(i));
        }
    }

    private void saveWeights(){
        savedWeights = new double[layers.size() - 1][][];
        for (int i = 0; i < layers.size() - 1; i++){
            savedWeights[i] = new double[layers.get(i).getNEURONS_SIZE()][];
            for (int j = 0; j < layers.get(i).getNEURONS_SIZE(); j++) {
                savedWeights[i][j] = new double[layers.get(i).getNext().getNEURONS_SIZE()];
                for (int k = 0; k < layers.get(i).getNext().getNEURONS_SIZE(); k++){
                    savedWeights[i][j][k] = layers.get(i).getNeurons()[j].getWeights()[k];
            }
            }
        }
    }

    public void save(String fileName) throws IOException {
        File file = new File(fileName);
        FileWriter fileWriter = new FileWriter(file, false);
        for (int i = 0; i < savedWeights.length; i++){
            fileWriter.write("Layer " + (i+1) + ":\n");
            for (int j = 0; j < savedWeights[i].length; j++) {
                for (int k = 0; k < savedWeights[i][j].length; k++){
                    fileWriter.write(savedWeights[i][j][k] + " ");
            }
                fileWriter.write("\n");
            }
        }
        fileWriter.close();
    }

    public void load(String fileName){
        File file = new File(fileName);
        try {
            Scanner scanner = new Scanner(file);
            int layerNumber = 0;
            while (scanner.hasNext()){
                String layerLine = scanner.nextLine();

                if (layerLine.contains("Layer")){
                    layerNumber = Integer.parseInt(layerLine.trim().replace("Layer", "").replace(":", "").replace("\n","")
                            .replace(" ", "")) - 1;
                }
                for (int j = 0; j < layers.get(layerNumber).getNEURONS_SIZE(); j++){
                    for (int k = 0; k < layers.get(layerNumber).getNext().getNEURONS_SIZE(); k++){
                        layers.get(layerNumber).getNeurons()[j].getWeights()[k] = scanner.nextDouble();
                    }
                    scanner.nextLine();
                }

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    private void adjustLayerWeights(FeedForwardLayer layer){
        for (int i = 0; i < layer.getNEURONS_SIZE(); i++){
            for (int j = 0; j < layer.getNext().getNEURONS_SIZE(); j++){
                double newWeight = layer.getNeurons()[i].getWeights()[j] // Old Weight + (ErrorJ * OutputI)
                        + (learningRate * layer.getNext().getNeurons()[j].getError() * layer.getNeurons()[i].getOutput());
                layer.getNeurons()[i].getWeights()[j] = newWeight;
            }
        }
    }

    private double calculateMeanSquareError(Neuron[] neurons, int actualOutput){
        double MSE = 0;
        for (Neuron neuron : neurons) {
            MSE += Math.pow((neuron.getOutput() - actualOutput), 2);
        }
        return MSE/2;
    }

    private double calculateOutputLayerError(double predictedOutput, int actualOutput){
        return (predictedOutput - actualOutput) * predictedOutput * (1 - predictedOutput);
    }

    private void calculateHiddenLayerError(FeedForwardLayer layer){
        for (int i = 0; i < layer.getNEURONS_SIZE(); i++){
            double neuronError = 0;
            for (int j = 0; j < layer.getNext().getNEURONS_SIZE(); j++){
                neuronError+= layer.getNext().getNeurons()[j].getError() * layer.getNeurons()[i].getWeights()[j];
            }
            neuronError = neuronError * layer.getNeurons()[i].getOutput() * (1 - layer.getNeurons()[i].getOutput());
            layer.getNeurons()[i].setError(neuronError);
        }
    }

    public void reset(){
        for (FeedForwardLayer layer: layers){
            if (layer.isHidden() || layer.isInput()){
                for (int i = 0; i < layer.getNEURONS_SIZE(); i++) {
                    double[] weights = randomizeWeights(layer.getNext().getNEURONS_SIZE());
                    layer.getNeurons()[i] = new Neuron(weights);
                }
            }else if (layer.isOutput()){
                for (int j = 0; j < layer.getNEURONS_SIZE(); j++){
                    layer.getNeurons()[j] = new Neuron();
                }
            }
        }
    }

    private double[] randomizeWeights(int size){
        double[] weights = new double[size];
        for (int i = 0; i < weights.length; i++) {
            Random random = new Random();
            weights[i] = random.doubles(-0.5, 0.5).findFirst().getAsDouble();
//            weights[i] = random.doubles(-0.03, 0.03).findFirst().getAsDouble(); //
        }
        return weights;
    }

    public int[] predict(int[][] testingSetFeatures){
        int[] predictedLabels = new int[10];
        for (int i = 0; i < testingSetFeatures.length; i++){
            for (int j = 0; j < testingSetFeatures[i].length; j++){
                layers.get(0).getNeurons()[j].setOutput((testingSetFeatures[i][j] - testingSetFeaturesMean[i]) / testingSetFeaturesStandardDev[i]);
            }
            feedForward();
            predictedLabels[i] = layers.get(layers.size() - 1).getNeurons()[0].getOutput() <= 0.5? 0 : 1;
        }
        return predictedLabels;
    }

    public int predict(int[] predictionSample){
        int predictedLabel;
        for (int i = 0; i < predictionSample.length; i++){
            layers.get(0).getNeurons()[i].setOutput((predictionSample[i] - predictionSampleMeanAndStandardDev[0]) / predictionSampleMeanAndStandardDev[1]);
        }
        feedForward();
        predictedLabel = layers.get(layers.size() - 1).getNeurons()[0].getOutput() >= 0.0049? 0 : 1;
//        predictedLabel = layers.get(layers.size() - 1).getNeurons()[0].getOutput() >= 0.005? 0 : 1;

        return predictedLabel;
    }

    public double calculateAccuracy(int[] predictedLabels, int[] testingSetLabels){
        int hit = 0;
        for (int i = 0; i < testingSetLabels.length; i++){
            hit = predictedLabels[i] == testingSetLabels[i]? hit+1: hit;
        }
        return (double) hit/testingSetLabels.length;
    }


    public void setTrainingSetFeaturesMeanAndStandardDev(double[] trainingSetFeaturesMean, double[] trainingSetFeaturesStandardDev) {
        this.trainingSetFeaturesMean = trainingSetFeaturesMean;
        this.trainingSetFeaturesStandardDev = trainingSetFeaturesStandardDev;
    }

    public void setTestingSetFeaturesMeanAndStandardDev(double[] testingSetFeaturesMean, double[] testingSetFeaturesStandardDev) {
        this.testingSetFeaturesMean = testingSetFeaturesMean;
        this.testingSetFeaturesStandardDev = testingSetFeaturesStandardDev;
    }

    public void setPredictionSampleMeanAndStandardDev(double[] sampleMeanAndStandardDev){
        this.predictionSampleMeanAndStandardDev = new double[sampleMeanAndStandardDev.length];
        this.predictionSampleMeanAndStandardDev = sampleMeanAndStandardDev;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getACCEPTABLE_ERROR() {
        return ACCEPTABLE_ERROR;
    }

    public void setACCEPTABLE_ERROR(double ACCEPTABLE_ERROR) {
        this.ACCEPTABLE_ERROR = ACCEPTABLE_ERROR;
    }

    public int getNUMBER_OF_EPOCHS() {
        return NUMBER_OF_EPOCHS;
    }

    public void setNUMBER_OF_EPOCHS(int NUMBER_OF_EPOCHS) {
        this.NUMBER_OF_EPOCHS = NUMBER_OF_EPOCHS;
    }
}
