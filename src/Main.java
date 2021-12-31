import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class ImageData {
    int[] pixels;
    int label;

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
//            System.out.println("image[" + i + "]-Name: " + images[i].getName());
//            System.out.println("data[" + i + "]-Pixels: " + data[i].pixels.length);
            data[i].setLabel(images[i].getName().contains("cat")? 0 : 1);
//            System.out.println("data[" + i + "]-Label: " + data[i].label);

        }

        //Shuffle
        List<ImageData> tempData = Arrays.asList(data);
        Collections.shuffle(tempData);
        tempData.toArray(data);

        //Split the data into training (75%) and testing (25%) sets
        final int TRAINING_SET_SIZE = (int) (data.length * 0.75);
        int[][] trainingSetFeatures = new int[TRAINING_SET_SIZE][], testingSetFeatures = new int[data.length - TRAINING_SET_SIZE][];
        int[] trainingSetLabels = new int[TRAINING_SET_SIZE], testingSetLabels = new int[data.length - TRAINING_SET_SIZE];

        for (int i = 0; i < TRAINING_SET_SIZE; i++){
                trainingSetFeatures[i] = data[i].pixels;
                trainingSetLabels[i] = data[i].label;
        }
        System.out.println("---------------------------------------------------");
        for (int i = 0; i < data[0].pixels.length; i++){
            System.out.println("Pixel[" + i + "]: " + data[0].pixels[i]);
        }

        System.out.println("----------------------------------------------------");
        for (int i = 0; i < data[1].pixels.length; i++){
            System.out.println("Pixel[" + i + "]: " + data[1].pixels[i]);
        }
        System.out.println("----------------------------------------------------");


        /*

            ...
         */

        //Create the NN
        NeuralNetwork nn = new NeuralNetwork();
        //Set the NN architecture
        /*

            ...
         */
//        final int LABEL_SIZE = 2;
//        final int NEURONS_SIZE_HIDDEN_LAYER = (data.length * (2/3)) + LABEL_SIZE;
//        nn.addLayer(new FeedForwardLayer(data.length)); // Input layer
//        nn.addLayer(new FeedForwardLayer(NEURONS_SIZE_HIDDEN_LAYER - 1)); // Hidden layer
////        nn.addLayer(new FeedForwardLayer(LABEL_SIZE)); // Output layer
//        nn.addLayer(new FeedForwardLayer(1)); // Hidden layer
//
//        //Train the NN
//        nn.train(trainingSetFeatures, trainingSetLabels);
//
//        //Test the model
//        int[] predictedLabels = nn.predict(testingSetFeatures);
//        double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);
//
//        //Save the model (final weights)
//        nn.save("model.txt");
//
//        //Load the model and use it on an image
//        NeuralNetwork nn2 = NeuralNetwork.load("model.txt");
//        int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File("sample.jpg"));
//        int samplePrediction = nn2.predict(sampleImgFeatures);
//        ImageHandler.showImage("sample.jpg");
//        //Print "Cat" or "Dog"
//        /*
//            ...
//         */
    }
}
