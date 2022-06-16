package de.htw.ml;

import org.jblas.FloatMatrix;

/**
 *
 * @author Nico Hezel
 */
public class LogisticRegression {
	
	protected int trainingIterations;
	protected float learnRate;
	protected float[] predictionRates;
	protected float[] trainErrors;	
	
	public LogisticRegression(int trainingIterations, float learnRate) {
		this.trainingIterations = trainingIterations;
		this.learnRate = learnRate;
	}

	public FloatMatrix train(FloatMatrix xTest, FloatMatrix yTest, FloatMatrix xTrain, FloatMatrix yTrain) {
		this.predictionRates = new float[trainingIterations];
		this.trainErrors = new float[trainingIterations];
		
		// initialize the weights
		org.jblas.util.Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(xTrain.getColumns(), 1);
		
		// current training error
		trainErrors[0] = cost(predict(xTrain, theta), yTrain);

		// best combination of weights
		FloatMatrix bestTheta = theta.dup();
		float bestPredictionRate = predictionRates[0] = predictionRate(predict(xTest, theta), yTest);
		
		// training
		for (int iteration = 0; iteration < trainingIterations; iteration++) {
			predictionRates[iteration] = predictionRate(predict(xTest, theta), yTest);
			trainErrors[iteration] = cost(predict(xTrain, theta), yTrain);

			if(predictionRates[iteration] > bestPredictionRate) {
				bestPredictionRate = predictionRates[iteration];
				bestTheta = theta;
			}
			theta = changeTheta(xTrain, yTrain, theta, learnRate);
		}
		
		return bestTheta;
	}

	private FloatMatrix changeTheta(FloatMatrix xTrain, FloatMatrix yTrain, FloatMatrix theta, float learnRate) {

		FloatMatrix h = predict(xTrain, theta).sub(yTrain).transpose();

		FloatMatrix newTheta = theta.sub(h.mmul(xTrain).mul(learnRate / yTrain.length));

		return newTheta;
	}

	/**
	 * Calculates a prediction of the input data X and the current weights theta
	 * 
	 * @param x
	 * @param theta
	 * @return
	 */
	public static FloatMatrix predict(FloatMatrix x, FloatMatrix theta) {
		FloatMatrix y;
		if(x.columns != theta.rows) {
			y = x.transpose().mmul(theta);
		} else {
			y = x.mmul(theta);
		}

		y = sigmoidi(y);

		return y;
	}
		
	/**
	 * Calculates the training error according to the logistical cost function or RMSE.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float cost(FloatMatrix prediction, FloatMatrix y) {
		return prediction.sub(y).norm1() / y.length;
	}

	/**
	 * Calculates a prediction rate between the prediction and the desired result Y.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float predictionRate(FloatMatrix prediction, FloatMatrix y) {
		prediction = prediction.ge(0.5f);
		float rate = prediction.sub(y).norm1();
		rate = (y.length-rate)/y.length * 100;
		return  rate;
	}

	/**
	 * Prediction rates of the last training
	 * 
	 * @return
	 */
	public float[] getPredictionRates() {
		return predictionRates;
	}
	
	/**
	 * error rates of the last training
	 * 
	 * @return
	 */
	public float[] getTrainError() {
		return trainErrors;
	}
	
	/**
	 * Replaces the values in the Input Matrix with their sigmoid variant.
	 * 
	 * @param input
	 * @return
	 */
	public static FloatMatrix sigmoidi(FloatMatrix input) {
		for (int i = 0; i < input.data.length; i++)
			input.data[i] = (float) (1. / ( 1. + Math.exp(-input.data[i]) ));
		return input;
	}
}