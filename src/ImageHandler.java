import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
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
}
