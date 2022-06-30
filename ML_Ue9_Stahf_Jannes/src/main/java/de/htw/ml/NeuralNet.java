package de.htw.ml;

import org.jblas.FloatMatrix;

/**
 * @author Jannes Stahf
 */
public class NeuralNet {

    protected int trainingIterations;
    protected float learnRate;
    protected float[] trainingErrors;
    private FloatMatrix[] thetas;

    public NeuralNet(int trainingIterations, float learnRate) {
        this.trainingIterations = trainingIterations;
        this.learnRate = learnRate;
        this.trainingErrors = new float[trainingIterations];
    }

    public FloatMatrix[] train(FloatMatrix xValues, FloatMatrix y, FloatMatrix[] thetas) {

        FloatMatrix[] x = new FloatMatrix[thetas.length];
        FloatMatrix[] a = new FloatMatrix[thetas.length];
        FloatMatrix[] deltaT = new FloatMatrix[thetas.length];

        // training
        for(int t = 0; t<trainingIterations; t++) {

            // --- forward pass ---
            for (int i = 0; i < thetas.length; i++) {
                // add bias
                x[i] = i == 0 ? addBias(xValues) : addBias(a[i - 1]);

                // prediction
                FloatMatrix z = predict(x[i], thetas[i]);

                // sigmoid
                a[i] = sigmoidi(z);

            }

            trainingErrors[t] = cost(a[thetas.length-1], y);

            FloatMatrix d = a[thetas.length - 1].sub(y).mul(a[thetas.length - 1].mul(a[thetas.length - 1].sub(1).neg()));

            // --- backpropagation ---
            for (int i = thetas.length - 1; i >= 0; i--) {

                if (i == thetas.length - 1) {
                    deltaT[i] = x[i].transpose().mmul(d).mmul(learnRate / xValues.rows);
                } else {
                    deltaT[i] = d.mmul(thetas[i + 1].transpose());
                    deltaT[i] = x[i].transpose().mmul(learnRate / x[i + 1].rows).mmul(deltaT[i].getRange(0, deltaT[i].rows, 1, deltaT[i].columns).mul(a[i].mul(a[i].sub(1).neg())));
                }
            }

            // assigning new thetas (has to happen after, not during, backpropagation if we use the same array again)
            for (int i = 0; i < thetas.length; i++) {
                thetas[i] = thetas[i].sub(deltaT[i]);
            }

        }

        // ---- console output ----
        System.out.println("Prediction =");
        for(int i = 0; i<a[thetas.length-1].rows; i++) {
            System.out.println(a[thetas.length-1].get(i));
        }

        this.thetas = thetas;
        return thetas;
    }

    private float cost(FloatMatrix prediction, FloatMatrix y) {
        return (float) Math.sqrt(prediction.sub(y).mul(prediction.sub(y)).mean());
    }

    private FloatMatrix addBias(FloatMatrix x) {
        return FloatMatrix.concatHorizontally(FloatMatrix.ones(x.rows,1), x);
    }

    private FloatMatrix predict(FloatMatrix x, FloatMatrix theta) {
        return x.mmul(theta);
    }

    public FloatMatrix sigmoidi(FloatMatrix input) {
        for (int i = 0; i < input.data.length; i++)
            input.data[i] = (float) (1. / ( 1. + Math.exp(-input.data[i]) ));
        return input;
    }

    public float[] getTrainingErrors() {return trainingErrors;}
}
