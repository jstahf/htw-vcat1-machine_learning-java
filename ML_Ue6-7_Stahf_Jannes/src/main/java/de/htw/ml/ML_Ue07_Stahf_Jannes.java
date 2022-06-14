package de.htw.ml;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import org.jblas.FloatMatrix;
import org.jblas.util.Random;
import org.w3c.dom.ranges.Range;

import java.io.IOException;
import java.util.Arrays;

public class ML_Ue07_Stahf_Jannes {

	public static final String title = "Logistic Regression (with bias) 200 iterations";
	public static final String xAxisLabel = "Iteration";
	public static final String yAxisLabel = "Prediction rate";

	private static FloatMatrix bestTheta;
	private static FloatMatrix test;
	private static FloatMatrix training;
	
	public static void main(String[] args) throws IOException {


		FloatMatrix credit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");

		// extract xValues
		FloatMatrix xVals = credit.getRange(0,credit.rows, 1, credit.columns);

		// extract yValues
		FloatMatrix creditability = credit.getColumn(0);

		// normalize xValues
		FloatMatrix xNorm = normalizeMinMax(xVals);

		// Theta values
		Random.seed(17);
		//FloatMatrix theta = FloatMatrix.rand(1,20);

		/*
		// TODO: ---- LinReg with no bias (and binarized results for RMSE calculation) ----
		int iterations = 100;

		float[] rmseValues = linReg(iterations, theta, 2f, xNorm, creditability);

		System.out.println("Best RMSE: " + rmseValues[rmseValues.length-1]);

		// plot the RMSE values
		FXApplication.plot(rmseValues);
		Application.launch(FXApplication.class);
		//TODO Exercise 1 end*/

		//TODO ---- LogReg (200) iterations with bias ----

		// dividing data:
		training = new FloatMatrix(1, 21);
		test = new FloatMatrix(1, 21);

		divideData(normalizeMinMax(credit));

		FloatMatrix trainingY = training.getColumn(0);
		FloatMatrix testY = test.getColumn(0);

		// adding bias
		FloatMatrix trainingX = FloatMatrix.concatHorizontally(FloatMatrix.ones(900), training.getRange(0, training.rows, 1, training.columns));
		FloatMatrix testX = FloatMatrix.concatHorizontally(FloatMatrix.ones(100), test.getRange(0, test.rows, 1, test.columns));

		// ---- Logistic Regression ----
		int iterations = 200;
		// creating random theta
		FloatMatrix theta = FloatMatrix.rand(1,21);

		float[] errorSum = logReg(iterations, theta, 2f, trainingX, trainingY, testX, testY);

		// plot the error-sum of testSet with best training theta
		FXApplication.plot(errorSum);
		Application.launch(FXApplication.class);

	}

	// ---- Logistic Regression ----
	public static float[] logReg(int iterations, FloatMatrix theta, float alpha, FloatMatrix xTraining, FloatMatrix yTraining, FloatMatrix xTest, FloatMatrix yTest) {

		float[] trainingValues = new float[iterations];
		float[] testValues = new float[iterations];
		float bestResult = 0;

		for(int i = 0; i<iterations; i++) {
			FloatMatrix results = binarize(denormalize(getH_Log(xTraining, theta), yTraining));

			trainingValues[i] = sumDiff(results, yTraining);

			/* not sure if theta should also change for worse for test-set
			if(bestTheta == null || trainingValues[i]>bestResult) {
				bestResult = trainingValues[i];
				bestTheta = theta;
				System.out.println(bestResult);
			}*/

			testValues[i] = sumDiff(binarize(denormalize(getH_Log(xTest, theta), yTest)), yTest);

			theta = changeTheta(xTraining, normalizeMinMax(yTraining), theta, alpha, true);
		}

		return testValues;
	}

	// ---- dividing date into test and training set ----
	public static void divideData(FloatMatrix all) {
		// NOTE: this method/solution is not the best. But i couldn't figure out how to
		// change specifically access a zero-sum row without another loop.
		// also the static fields are not optimal. But since java doesn't allow
		// multiple return types without creating another class, i went the short path.
		int zeros = 0;
		int ones = 0;
		int i = 0;

		while(i<all.rows) {

			if(zeros<50 && all.getRow(i).get(0) == 0) {
				if(test.getRow(0).sum() == 0) {
					test.putRow(0, all.getRow(i));

				} else {
					test = FloatMatrix.concatVertically(test, all.getRow(i));
				}
				zeros++;

			} else if (test.rows<100 && all.getRow(i).get(0) == 1 && ones < 50) {

				if(test.getRow(0).sum() == 0) {
					test.putRow(0, all.getRow(i));
				} else {
					test = FloatMatrix.concatVertically(test, all.getRow(i));
				}
				ones++;

			} else {

				if(training.getRow(0).sum() == 0) {
					training.putRow(0, all.getRow(i));
				} else {
					training = FloatMatrix.concatVertically(training, all.getRow(i));
				}
			}
			i++;
		}
	}

	// ---- Linear Regression ----
	public static float[] linReg(int iterations, FloatMatrix theta, float alpha, FloatMatrix xValues, FloatMatrix yValues) {
		float[] values = new float[iterations];

		for(int i = 0; i<iterations; i++) {
			FloatMatrix results = denormalize(getH(xValues, theta), yValues);
			results = binarize(results);
			double error = getRMSE(results, yValues);
			values[i] = (float) error;
			theta = changeTheta(xValues, normalizeMinMax(yValues), theta, alpha, false);
		}

		return values;
	}

	// ---- hyptothesis ----
	public static FloatMatrix getH(FloatMatrix xValues, FloatMatrix theta) {
		FloatMatrix y;

		if(xValues.rows != theta.columns) {
			y = theta.mmul(xValues.transpose());
		} else {
			y = theta.mmul(xValues);
		}

		return y;
	}

	// ---- sigmoid function ----
	public static FloatMatrix getH_Log(FloatMatrix xValues, FloatMatrix theta) {
		FloatMatrix y = getH(xValues, theta);

		for(int i = 0; i<y.columns; i++) { // sorry, i didn't know how to do e^-z without loop :(
			y.put(i, (float) (1/(1+Math.pow(Math.E, -y.get(i)))));
		}

		return y;
	}

	// ---- calculate new theta ----
	public static FloatMatrix changeTheta(FloatMatrix xValues, FloatMatrix yValues, FloatMatrix oldTheta, float alpha, boolean logReg) {
		FloatMatrix newTheta;

		if(logReg) {
			newTheta = oldTheta.sub(getH_Log(xValues.transpose(), oldTheta).sub(yValues).mmul(xValues).mul(alpha / yValues.length));
		} else {
			newTheta = oldTheta.sub(getH(xValues.transpose(), oldTheta).sub(yValues).mmul(xValues).mul(alpha / xValues.length));
		}

		return newTheta;
	}

	// ---- normalize ----
	public static FloatMatrix normalizeMinMax(FloatMatrix values) {

		FloatMatrix norm = new FloatMatrix(values.rows, values.columns);

		for(int column = 0; column<values.columns; column++) {
			float min = values.getColumn(column).min();
			float max = values.getColumn(column).max();

			FloatMatrix normCol = values.getColumn(column).sub(min).div(max - min);

			norm.putColumn(column, normCol);
		}

		return norm;
	}

	// ---- denormalize ----
	public static FloatMatrix denormalize(FloatMatrix values, FloatMatrix origin) {

		float min = origin.min();
		float max = origin.max();

		FloatMatrix denorm = values.mul(max-min).add(min);

		return denorm;
	}

	// ---- Prediction Error (Logistic Reg) ----
	private static float sumDiff(FloatMatrix results, FloatMatrix yValues) {
		float rate = results.sub(yValues).norm1();
		rate = (yValues.length-rate)/yValues.length * 100;
		return  rate;
	}

	// ---- RMSE calculation ----
	public static double getRMSE(FloatMatrix y, FloatMatrix y1) {
		double rmse = Math.sqrt(y1.sub(y).mul(y1.sub(y)).mean());

		return rmse;
	}


	// binarizes a result matrix
	private static FloatMatrix binarize(FloatMatrix results) {
		return results.ge(0.5f);
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
	
		/**
		 * equivalent to linspace in Octave
		 * 
		 * @param lower
		 * @param upper
		 * @param num
		 * @return
		 */
		private static FloatMatrix linspace(float lower, float upper, int num) {
	        float[] data = new float[num];
	        float step = Math.abs(lower-upper) / (num-1);
	        for (int i = 0; i < num; i++)
	            data[i] = lower + (step * i);
	        data[0] = lower;
	        data[data.length-1] = upper;
	        return new FloatMatrix(data);
	    }
		
		// y-axis values of the plot 
		private static float[] dataY;
		
		/**
		 * Draw the values and start the UI
		 */
		public static void plot(float[] yValues) {
			dataY = yValues;
		}
		
		/**
		 * Draw the UI
		 */
		@SuppressWarnings("unchecked")
		@Override 
		public void start(Stage stage) {
	
			stage.setTitle(title);
			
			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel(xAxisLabel);
	        final NumberAxis yAxis = new NumberAxis();
	        yAxis.setLabel(yAxisLabel);
	        
			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
	
			XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
			series1.setName("Alpha = 2");
			for (int i = 0; i < dataY.length; i++) {
				series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
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
