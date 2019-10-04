package javaspark;

import java.io.File;
import java.io.IOException;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
//import org.nd4j.linalg.lossfunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.conf.layers.OutputLayer;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class deeplearning2 {

	public static final String datafiledir1 = "C:/Users/super/Desktop/serenastraining/csvharrypotter";

	public static final String datafiledir = "C:/Users/super/Desktop/serenastraining/csvharrypotter/amrepwinglev";

	public static final String labelfiledir = "C:/Users/super/Desktop/serenastraining/csvharrypotter/amrepwinglevlabels";
	
	public static final String irisdatadir = "C:/Users/super/Desktop/serenastraining/iris";
	
	
	public static void main(String[] args) throws IOException, InterruptedException {
		
		// number of examples / time series in each batch
		int miniBatchSize = 10;
		
		int numPossibleLabels = 3;
		// use regression = false for classification
		boolean regression = false;
		
		SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ",");
		SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");
		
		File f = new File(datafiledir+"/harrypotterfeb0.csv");
		
		File f1 = new File(datafiledir1+"/totalharrypotter.csv");
		
		File firis = new File(irisdatadir+"/Iris.csv");
		
		featureReader.initialize(new NumberedFileInputSplit(datafiledir+"/harrypotterfeb%d.csv", 0, 22));
		
		
		labelReader.initialize(new NumberedFileInputSplit(labelfiledir+"/harrypotterfeblabel%d.csv", 0, 22));
		
		DataSetIterator iter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression);

		
		//rr.initialize(new FileSplit(new File("/path/to/directory")));

		//DataSetIterator iterR = (DataSetIterator) new RecordReaderDataSetIterator.Builder(featureReader, 32);
		
		// DataSetIterator can then be passed to MultiLayerNetwork.fit() to train the network.

		int seed = 100;
		
		// learning rate: hyperparameter that controls how much adjust weights of network w.r.t loss gradient
		// large value 0.1, and then lower 0.01 and 0.001, etc
		double learningRate=0.1;
		
		// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/LSTMCharModellingExample.java
		// LSTM parameters
		int lstmLayerSize = 200;	
		int nOut= 3;
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters

		
		/*
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				//.seed(seed)
				//.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				//.learningRate(learningRate)
	            //.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
	            //.activation(Activation.IDENTITY).nIn(10).nOut(1).build())
	            .build();
	      */      
        System.out.println("feature labels"+featureReader.getLabels());
        System.out.println("iter input columns"+ iter.getLabels());
        
        //BasicConfigurator.configure();
		
        /*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(10)
				.l2(0.0001)
	            .weightInit(WeightInit.XAVIER)
	            .updater(new Adam(0.005))
				.list()
				.layer(new LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
						.activation(Activation.TANH).build())
				//.layer(new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
						//.activation(Activation.TANH).build())
				//.layer(new RnnOutputLayer.Builder(LossFunctions.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
				//.nIn(lstmLayerSize).nOut(nOut).build())
	            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
				.build();
		
        
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                //.activation("relu")
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.05))
                // ... other hyperparameters
                .list()
                .backprop(true)
                .layer(new DenseLayer.Builder().nIn(7).nOut(7))
                .build();
        
        
        int numInputs2 = 22;
        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs2).nOut(3)
                    .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                    .build())
                //.layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                  //  .activation(Activation.SOFTMAX)
                   // .nIn(3).nOut(outputNum).build())
                .build();
        */
        
		//MultiLayerNetwork net = new MultiLayerNetwork(conf3);
		
		//net.init();
		
		// int inputs = 10;
        
		/*
		INDArray features = Nd4j.zeros(inputs);
		for (int i=0; i<inputs; i++) 
		    features.putScalar(new int[] {i}, Math.random() < 0.5 ? 0 : 1);
		*/
		
		int numLinesToSkip = 1; // first row has names 
		
        char delimiter = ',';
        

		RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
		
        recordReader.initialize(new FileSplit(f1));
        
        //recordReader.initialize(new FileSplit(firis));
        
        System.out.println("labels"+ recordReader.getLabels());
        
        System.out.println("next rec reader"+recordReader.next());
        
        System.out.println("next rec reader"+recordReader.next());
        
        // 71 columns
        int labelIndex = 70;  // label is the last column 
        int numClasses=3; // 3 classes / types of wand movements
        int batchSize = 25; // number of rows 
        
        // iris
        //labelIndex= 5;
        //batchSize = 150;
        
        //DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSetIterator iterator2 = new RecordReaderDataSetIterator.Builder(recordReader, 70).build();
        
        System.out.println("iterator2 labels:"+iterator2.getLabels());
        
        DataSet allData = iterator2.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.70); // use 70% of data for training
        
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        
        // normalizing data
        DataNormalization normalizer = new NormalizerStandardize();

        // 
        final int numInputs=70;
        int outputNum = 3;
        
        // build model for csv file
        MultiLayerConfiguration confcsv = new NeuralNetConfiguration.Builder()
        		.seed(seed)
        		.activation(Activation.TANH)
        		.weightInit(WeightInit.XAVIER)
        		.updater(new Sgd(0.1))
        		.l2(1e-4)
        		.list()
        		.layer(new DenseLayer.Builder().nIn(numInputs).nOut(outputNum).build())
        		.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(numInputs).nOut(outputNum).build())
        		.build();
        
        // run model
        MultiLayerNetwork modelcsv = new MultiLayerNetwork(confcsv);
        modelcsv.init();
        modelcsv.setListeners(new ScoreIterationListener(100));
        
        for (int i=0; i<300; i++) {
        	modelcsv.fit(trainingData);
        }
        
        Evaluation evalcsv = new Evaluation(3);
        INDArray output = modelcsv.output(testData.getFeatures());
        evalcsv.eval(testData.getLabels(), output);
        
		//DataSetIterator iterfeatures = new SequenceRecordReaderDataSetIterator(features, labelReader, miniBatchSize, numPossibleLabels, regression);

		/*
		net.fit(iter);
		

		// using an evaluation class
		Evaluation eval = new Evaluation(3); //create an evaluation object with 3 possible classes
		 */
		
	}

}
