import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class IrisApp {
    public static void main(String[] args) throws Exception {
        double learninRate=0.001;
        int numInputs=4;
        int numHidden=10;
        int numOutputs=3;
        System.out.println("Création du modèle");
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learninRate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build()
                    )
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .build()
                )
                .build();
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();
        //System.out.println(configuration.toJson());


        //Démarrage du serveur du monitoring du processus d'apprentissage
        UIServer uiServer=UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);

        model.setListeners(new StatsListener(inMemoryStatsStorage));

        System.out.println("Entraînement du modèle");

        File fileTrain=new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain=new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        int batchSize=1;
        int classIndex=4;
        DataSetIterator dataSetIteratorTrain=
                new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,numOutputs);

        while(dataSetIteratorTrain.hasNext()){
            System.out.println("------------------------------");
            DataSet dataSet=dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }
        int numEpochs=500;
        for(int i = 0; i<numEpochs; i++){
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("Evaluation du modèle");

        File fileTest=new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest=new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,numOutputs);
        Evaluation evaluation=new Evaluation();
        while ((dataSetIteratorTest.hasNext())){
            DataSet dataSetTest=dataSetIteratorTest.next();
            INDArray features=dataSetTest.getFeatures();
            INDArray targetlabels=dataSetTest.getLabels();
            INDArray predictedlabels=model.output(features);
            evaluation.eval(predictedlabels,targetlabels);
        }
        System.out.println(evaluation.stats());

        ModelSerializer.writeModel(model,"irisModel.zip",true);

    }
}
