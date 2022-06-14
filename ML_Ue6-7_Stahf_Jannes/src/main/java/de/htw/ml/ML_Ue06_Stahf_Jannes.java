package de.htw.ml;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import org.jblas.FloatMatrix;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import org.jblas.util.Random;

public class ML_Ue06_Stahf_Jannes {

	public static final String title = "Linear Regression (without bias) 100 iterations";
	public static final String xAxisLabel = "Iteration";
	public static final String yAxisLabel = "RMSE";
	
	public static void main(String[] args) throws IOException {
		/*
		 //TODO: First part of the exercise. cars.csv with 6 random theta values (without bias) and 1f as alpha value. Remove comment for result.
		FloatMatrix cars = FloatMatrix.loadCSVFile("cars_jblas.csv");

		// extract xValues
		FloatMatrix xVals = cars.getRange(0,cars.rows, 0, cars.columns-1);
		// extract yValues
		FloatMatrix mpg = cars.getColumn(6);

		// normalize xValues
		xVals = normalizeMinMax(xVals);

		// ---- LinReg with no bias ----
		int iterations = 100;

		Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(6,1);

		theta = theta.transpose();

		float[] rmseValues = linReg(iterations, theta, 1f, xVals, mpg);

		// plot the RMSE values
		FXApplication.plot(rmseValues);
		Application.launch(FXApplication.class);
		*/
		//TODO: Exercise 1 end

		//TODO Exercise 2:

		FloatMatrix credit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");

		// extract xValues
		FloatMatrix xVals1 = credit.getRange(0,credit.rows, 0, 5);
		FloatMatrix xVals2 = credit.getRange(0, credit.rows, 6, credit.columns);
		FloatMatrix xVals = FloatMatrix.concatHorizontally(xVals1, xVals2);

		// extract yValues
		FloatMatrix creditAmount = credit.getColumn(5);

		// normalize xValues
		xVals = normalizeMinMax(xVals);

		// ---- LinReg with no bias ----
		int iterations = 100;

		Random.seed(11);
		FloatMatrix theta = FloatMatrix.rand(20,1);

		theta = theta.transpose();

		float[] rmseValues = linReg(iterations, theta, 7f, xVals, creditAmount);

		System.out.println("Best RMSE: " + rmseValues[rmseValues.length-1]);

		// plot the RMSE values
		FXApplication.plot(rmseValues);
		Application.launch(FXApplication.class);
		//TODO Exercise 2 end
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

	// ---- RMSE calculation ----
	public static double getRMSE(FloatMatrix y, FloatMatrix y1) {
		double rmse = Math.sqrt(y1.sub(y).mul(y1.sub(y)).mean());

		return rmse;
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

	// ---- cost function ----
	public static double getCost(FloatMatrix y, FloatMatrix y1) {
		double cost = y1.sub(y).mul(y1.sub(y)).mean()/2;

		return cost;
	}

	// ---- calculate new theta ----
	public static FloatMatrix changeTheta(FloatMatrix xValues, FloatMatrix yValues, FloatMatrix oldTheta, float alpha) {
		FloatMatrix newTheta = oldTheta.sub(getH(xValues.transpose(), oldTheta).sub(yValues).mmul(xValues).mul(alpha / xValues.length));

		return newTheta;
	}

	// ---- Linear Regression ----
	public static float[] linReg(int iterations, FloatMatrix theta, float alpha, FloatMatrix xValues, FloatMatrix yValues) {
		float[] values = new float[iterations];

		for(int i = 0; i<iterations; i++) {
			double error = getRMSE(denormalize(getH(xValues, theta), yValues), yValues);
			values[i] = (float) error;
			theta = changeTheta(xValues, normalizeMinMax(yValues), theta, alpha);
		}

		return values;
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
			series1.setName("Alpha = 7");
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
