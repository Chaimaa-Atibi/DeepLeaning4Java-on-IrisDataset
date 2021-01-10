import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class IrisPrediction {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));
        System.out.println("Prédiction");
        String[] labels= {"Iris-setosa","Iris-versicolor","Iris-virginica"};
        INDArray inputData= Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2},
                {4.9,3.0,1.4,0.2},
                {6.7,3.1,4.4,1.4},
                {5.6,3.0,4.5,1.5},
                {6.0,3.0,4.8,1.8},
                {6.9,3.1,5.4,2.1}
        });
        INDArray output=model.output(inputData);
        int[] classes=output.argMax(1).toIntVector();

        System.out.println(output);
        for (int i = 0; i < classes.length; i++) {
            System.out.println("Classe: "+labels[classes[i]]);
        }
    }
}
