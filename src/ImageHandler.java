import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

public class ImageHandler {
    public static int[] ImageToIntArray(File file) throws IOException {
        BufferedImage img = ImageIO.read(file);
        int width = img.getWidth(), height = img.getHeight();
        int[] imgArr = new int[height*width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                imgArr[i*width+j] = img.getData().getSample(j, i, 0);
            }
        }
        return imgArr;
    }

    public static double calculateImageMean(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        double sum = 0.0;
        Raster raster = image.getRaster();
        for (int y = 0; y < image.getHeight(); ++y){
            for (int x = 0; x < image.getWidth(); ++x){
                sum += raster.getSample(x, y, 0);
            }
        }
        return sum / (image.getWidth() * image.getHeight());
    }

    public static double calculateImageStandardDev(File file, double mean) throws IOException {
        BufferedImage image = ImageIO.read(file);
        double standardDev = 0.0;
        Raster raster = image.getRaster();
        for (int y = 0; y < image.getHeight(); ++y){
            for (int x = 0; x < image.getWidth(); ++x){
                standardDev+= Math.pow(raster.getSample(x, y, 0) - mean, 2);
            }
        }
        standardDev = Math.sqrt(standardDev/ (image.getWidth() * image.getHeight()));
        return standardDev;
    }

    public static void showImage(String filename) throws IOException {
        BufferedImage img = ImageIO.read(new File(filename));
        JFrame frame=new JFrame();
        ImageIcon icon=new ImageIcon(img);
        frame.setSize(img.getWidth()*5, img.getHeight()*5);
        JLabel lbl=new JLabel(icon);
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static double[] calculateImageMeanAndStandardDev(File file) throws IOException {
        double mean = ImageHandler.calculateImageMean(file);
        double standardDev = ImageHandler.calculateImageStandardDev(file, mean);
        return new double[]{mean, standardDev};
    }
}
