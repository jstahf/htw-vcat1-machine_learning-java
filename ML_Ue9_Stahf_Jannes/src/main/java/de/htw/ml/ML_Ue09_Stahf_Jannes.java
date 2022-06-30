package de.htw.ml;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
import org.jblas.FloatMatrix;
import org.jblas.util.Random;

import java.io.IOException;

/**
 * @author Jannes Stahf
 */
public class ML_Ue09_Stahf_Jannes {
	
	private static final int TrainingIterations = 1000;
	private static final float LearnRate = 3f;
	
	public static void main(String[] args) throws IOException {

		/* //TODO: EXERCISE 1 (test data)
		FloatMatrix x = new FloatMatrix(2,2);
		x.putRow(0, new FloatMatrix(new float[]{0.35f, 0.9f}));
		x.putRow(1, new FloatMatrix(new float[]{0.1f, -0.7f}));

		FloatMatrix y = new FloatMatrix(2,1);
		y.putRow(0, new FloatMatrix(new float[]{0.5f}));
		y.putRow(1, new FloatMatrix(new float[]{0.35f}));

		FloatMatrix[] thetas = new FloatMatrix[2];

		thetas[0] = new FloatMatrix(3,2);
		thetas[0].putRow(0, new FloatMatrix(new float[]{0f, 0f}));
		thetas[0].putRow(1, new FloatMatrix(new float[]{0.1f, 0.4f}));
		thetas[0].putRow(2, new FloatMatrix(new float[]{0.8f, 0.6f}));

		thetas[1] = new FloatMatrix(3,1);
		thetas[1].putRow(0, new FloatMatrix(new float[]{0f}));
		thetas[1].putRow(1, new FloatMatrix(new float[]{0.3f}));
		thetas[1].putRow(2, new FloatMatrix(new float[]{0.9f}));

		NeuralNet nn1 = new NeuralNet(TrainingIterations, LearnRate);
		nn1.train(x, y, thetas);

		float[] trainingErorrs = nn1.getTrainingErrors();
		System.out.println("Before weight change: RMSE = " + trainingErorrs[0]);
		System.out.println("After weight change: RMSE = " + trainingErorrs[trainingErorrs.length-1]);
		*/

		// ---- EXERCISE 2 (XOR PROBLEM) ----

		FloatMatrix x = new FloatMatrix(4,2);
		x.putRow(0, new FloatMatrix(new float[]{0f, 0f}));
		x.putRow(1, new FloatMatrix(new float[]{1f, 0f}));
		x.putRow(2, new FloatMatrix(new float[]{0f, 1f}));
		x.putRow(3, new FloatMatrix(new float[]{1f, 1f}));

		FloatMatrix y = new FloatMatrix(4,1);
		y.putRow(0, new FloatMatrix(new float[]{0f}));
		y.putRow(1, new FloatMatrix(new float[]{1f}));
		y.putRow(2, new FloatMatrix(new float[]{1f}));
		y.putRow(3, new FloatMatrix(new float[]{0f}));

		Random.seed(17);
		FloatMatrix[] thetas = new FloatMatrix[] {FloatMatrix.rand(3,2), FloatMatrix.rand(3,1), FloatMatrix.rand(2,1)}; // Note: Be careful changing nodes and layers. Not all configs are compatible

		NeuralNet nn1 = new NeuralNet(TrainingIterations, LearnRate);
		nn1.train(x, y, thetas);

		float[] trainingErorrs = nn1.getTrainingErrors();
		System.out.println("Before weight change: RMSE = " + trainingErorrs[0]);
		System.out.println("After weight change: RMSE = " + trainingErorrs[trainingErorrs.length-1]);



		FXApplication.plot(trainingErorrs);
		Application.launch(FXApplication.class);
	}
	
	
	
	// ---------------------------------------------------------------------------------
	// ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
	// ---------------------------------------------------------------------------------
	
	/**
	 * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
	 * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
	 * 
	 * @author Nico Hezel
	 *
	 */
	public static class FXApplication extends Application {

		private static float[] trainingsError;
		
		/**
		 * Start the application and plot the data
		 * @param trainingsError
		 */
		public static void plot(float[] trainingsError) {
			FXApplication.trainingsError = trainingsError;
		}

		/**
		 * Draw the plot
		 */	
		@Override public void start(Stage stage) {

			stage.setTitle("Training Error");

			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel("Iteration");
			final NumberAxis yAxis = new NumberAxis();
			yAxis.setLabel("Erorr (RMSE)");

			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);

			XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
			series1.setName("Alpha = 3");
			for (int i = 0; i < trainingsError.length; i++) {
				series1.getData().add(new XYChart.Data<Number, Number>(i, trainingsError[i]));
			}

			sc.setAnimated(false);
			sc.setCreateSymbols(true);

			sc.getData().addAll(series1);

			Scene scene = new Scene(sc, 500, 400);
			stage.setScene(scene);
			stage.show();
	    }

	}
}
