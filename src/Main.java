import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class ImageData {
    int[] pixels;
    int label;
    double mean;
    double standardDev;

    ImageData(){}
    ImageData(int imgHeight, int imgWidth) { pixels = new int[imgHeight*imgWidth]; }
    public void setPixels(int[] pixels) { this.pixels = pixels; }
    public void setLabel(int lbl) { label = lbl; }
}

public class Main {
    public static void main(String[] args) throws IOException {
        //Load Data
        File[] images= new File("Cats & Dogs Sample Dataset").listFiles();
        ImageData[] data = new ImageData[images.length];
        for (int i = 0; i < images.length; i++) {
            data[i]=new ImageData();
            data[i].setPixels(ImageHandler.ImageToIntArray(images[i]));
            data[i].setLabel(images[i].getName().contains("cat")? 0 : 1);
            double[] meanAndStandard = ImageHandler.calculateImageMeanAndStandardDev(images[i]);
            data[i].mean = meanAndStandard[0];
            data[i].standardDev = meanAndStandard[1];
        }

        //Shuffle
        List<ImageData> tempData = Arrays.asList(data);
        Collections.shuffle(tempData);
        tempData.toArray(data);

        //Split the data into training (75%) and testing (25%) sets
        final int TRAINING_SET_SIZE = (int) (data.length * 0.75);
        int[][] trainingSetFeatures = new int[TRAINING_SET_SIZE][], testingSetFeatures = new int[data.length - TRAINING_SET_SIZE][];
        int[] trainingSetLabels = new int[TRAINING_SET_SIZE], testingSetLabels = new int[data.length - TRAINING_SET_SIZE];

        double[] trainingSetFeaturesMean = new double[TRAINING_SET_SIZE];
        double[] trainingSetFeaturesStandardDev = new double[TRAINING_SET_SIZE];

//        double[] testingSetFeaturesMean = new double[TRAINING_SET_SIZE];
//        double[] testingSetFeaturesStandardDev = new double[TRAINING_SET_SIZE];
        double[] testingSetFeaturesMean = new double[data.length - TRAINING_SET_SIZE];
        double[] testingSetFeaturesStandardDev = new double[data.length - TRAINING_SET_SIZE];

        for (int i = 0; i < TRAINING_SET_SIZE; i++){
            trainingSetFeaturesMean[i] = data[i].mean;
            trainingSetFeaturesStandardDev[i] = data[i].standardDev;
            trainingSetFeatures[i] = data[i].pixels;
            trainingSetLabels[i] = data[i].label;
        }

        for (int i = 0; i < (data.length - TRAINING_SET_SIZE); i++){
            testingSetFeaturesMean[i] = data[i + TRAINING_SET_SIZE].mean;
            testingSetFeaturesStandardDev[i] = data[i + TRAINING_SET_SIZE].standardDev;
            testingSetFeatures[i] = data[i + TRAINING_SET_SIZE].pixels;
            testingSetLabels[i] = data[i + TRAINING_SET_SIZE].label;
        }

        //Create the NN
        NeuralNetwork nn = new NeuralNetwork();
        //Set the NN architecture
        final int LABEL_SIZE = 2;
        final int NEURONS_SIZE_HIDDEN_LAYER = (1600 * (2/3)) + LABEL_SIZE;
        nn.addLayer(new FeedForwardLayer(1600)); // Input layer
//        nn.addLayer(new FeedForwardLayer(1068)); // Hidden layer
        nn.addLayer(new FeedForwardLayer(100)); // Hidden layer
        nn.addLayer(new FeedForwardLayer(1)); // Hidden layer
        nn.setLearningRate(0.3);
        nn.setACCEPTABLE_ERROR(0.001);
        nn.setNUMBER_OF_EPOCHS(500);
        nn.reset();

//        Train the NN
        nn.setTrainingSetFeaturesMeanAndStandardDev(trainingSetFeaturesMean, trainingSetFeaturesStandardDev);
        nn.train(trainingSetFeatures, trainingSetLabels);

        //Test the model
        nn.setTestingSetFeaturesMeanAndStandardDev(testingSetFeaturesMean, testingSetFeaturesStandardDev);
        int[] predictedLabels = nn.predict(testingSetFeatures);
        double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);

        //Save the model (final weights)
        nn.save("model.txt");

        //Load the model and use it on an image
        NeuralNetwork nn2 = new NeuralNetwork();
        nn2.addLayer(new FeedForwardLayer(1600));
//        nn2.addLayer(new FeedForwardLayer(1068)); 100 produced was more accurate compared to 1068
        nn2.addLayer(new FeedForwardLayer(100));
        nn2.addLayer(new FeedForwardLayer(1));
        nn2.reset();
        nn2.load("model.txt");


        nn2.setPredictionSampleMeanAndStandardDev(ImageHandler.calculateImageMeanAndStandardDev(new File("dog.jpg")));
        int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File("dog.jpg"));
        int samplePrediction = nn2.predict(sampleImgFeatures);
        ImageHandler.showImage("dog.jpg");
        //Print "Cat" or "Dog"
        System.out.println(samplePrediction == 0? "Cat" : "Dog");

        nn2.setPredictionSampleMeanAndStandardDev(ImageHandler.calculateImageMeanAndStandardDev(new File("sample.jpg")));
        sampleImgFeatures = ImageHandler.ImageToIntArray(new File("sample.jpg"));
        samplePrediction = nn2.predict(sampleImgFeatures);
        ImageHandler.showImage("sample.jpg");
        //Print "Cat" or "Dog"
        System.out.println(samplePrediction == 0? "Cat" : "Dog");

    }

}
