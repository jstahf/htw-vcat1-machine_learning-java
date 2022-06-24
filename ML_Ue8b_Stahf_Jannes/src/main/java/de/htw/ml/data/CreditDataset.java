package de.htw.ml.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.jblas.FloatMatrix;

/**
 * There are a lot TODOs here.
 * The class divides the german credit dataset into train and test data.
 *
 * @author Nico Hezel
 */
public class CreditDataset implements Dataset {

	protected Random rnd = new Random(7);

	protected FloatMatrix xTrain;
	protected FloatMatrix yTrain;

	protected FloatMatrix xTest;
	protected FloatMatrix yTest;

	protected int[] categories;

	public CreditDataset() throws IOException {

		int predictColumn = 15; // typ of apartment
		FloatMatrix data = FloatMatrix.loadCSVFile("german_credit_jblas.csv");

		// List with all categories in the predictColumn
		final FloatMatrix outputData = data.getColumn(predictColumn);
		categories = IntStream.range(0, outputData.rows).map(idx -> (int)outputData.data[idx]).distinct().sorted().toArray();
		int[] categorySizes = IntStream.of(categories).map(v -> (int)outputData.eq(v).sum()).toArray();
		System.out.println("The unique values of y are "+ Arrays.toString(categories)+" and there number of occurrences are "+Arrays.toString(categorySizes));

		// Array with all rows that are not predictColumn
		int[] xColumns = IntStream.range(0, data.columns).filter(value -> value != predictColumn).toArray();

		// Input and output data
		FloatMatrix x = data.getColumns(xColumns);
		FloatMatrix y = data.getColumn(predictColumn);

		// min and maximum for all columns
		FloatMatrix xMin = x.columnMins();
		FloatMatrix xMax = x.columnMaxs();

		// normalize the data sets and add the bias column
		FloatMatrix xNorm = x.subRowVector(xMin).diviRowVector(xMax.sub(xMin));
		xNorm = FloatMatrix.concatHorizontally(FloatMatrix.ones(xNorm.rows, 1), xNorm);

		int testDataPerCategory = data.getRows() / 10 / categories.length; // 10% test set
		int testDataCount = testDataPerCategory * categories.length;
		System.out.println("Use "+testDataCount+" as test data with "+testDataPerCategory+" elements per category.\n");


		divideData(xNorm, y, testDataPerCategory);
	}

	/**
	 * Takes all data and divides it into a training and test set. Ultimately writes the values into
	 * the classes fields holding the 4 corresponding matrices.
	 *
	 * @param xNorm normalized x values
	 * @param y y values
	 * @param testDataPerCategory how many data for each (of 3) catergories
	 * @author Jannes Stahf
	 */
	private void divideData(FloatMatrix xNorm, FloatMatrix y, int testDataPerCategory) {
		FloatMatrix data = xNorm.copy(xNorm);

		int y1 = 0;
		int y2 = 0;
		int y3 = 0;

		// the following reduces in-loop-code by a few lines
		yTrain = y.getRow(0);
		xTrain = xNorm.getRow(0);

		while (xTest == null || xTrain.length+xTest.length<xNorm.length){
			int i = rnd.nextInt(data.rows); // random values for better represantation/results

			switch((int) y.get(i)) {
				case 1:
					y1++;
					break;
				case 2:
					y2++;
					break;
				case 3:
					y3++;
					break;
			}

			if((y.get(i) == 1 && y1>testDataPerCategory) || (y.get(i) == 2 && y2>testDataPerCategory) || (y.get(i) == 3 && y3>testDataPerCategory)){
				yTrain = FloatMatrix.concatVertically(yTrain, y.getRow(i));
				xTrain = FloatMatrix.concatVertically(xTrain, xNorm.getRow(i));
			} else {
				if (yTest == null) {
					yTest = y.getRow(i);
					xTest = xNorm.getRow(i);
				} else {
					yTest = FloatMatrix.concatVertically(yTest, y.getRow(i));
					xTest = FloatMatrix.concatVertically(xTest, xNorm.getRow(i));
				}
			}
			data = FloatMatrix.concatVertically(data.getRange(0, i,0 ,data.columns), data.getRange(i, data.rows, 0 , data.columns));
		}
	}

	public int[] getCategories() {
		return categories;
	}

	/**
	 * The train data should contain as many train entries as possible but the ration
	 * between data points of the desired category and data points of a different category
	 * should be 50:50. All Y data are binarized:
	 *  - desired category = 1
	 *  - other category = 0
	 *
	 * @param category
	 * @return {x Matrix,y Matrix}
	 * @author Nico Hezel & Jannes Stahf
	 */
	public Dataset getSubset(int category) {

		// Search as many other lines with a different category. Remove indices if
		// necessary, to ensure the size of both set are the same

		List<Integer> desiredIndices = new ArrayList<>();
		List<Integer> otherIndices = new ArrayList<>();

		// check for desired category and save index
		for(int i = 0; i<yTrain.length; i++) {
			if(yTrain.get(i) == category) {
				desiredIndices.add(i);
			} else {
				otherIndices.add(i);
			}
		}

		// if one set is bigger -> make them 50:50
		if(desiredIndices.size()>otherIndices.size()) {
			desiredIndices = desiredIndices.subList(0, otherIndices.size());
		} else {
			otherIndices = otherIndices.subList(0, desiredIndices.size());
		}

		otherIndices.addAll(desiredIndices); // join them together

		int[] rowIndizies = otherIndices.stream().mapToInt(i -> i).toArray(); // convert to int[]

		// Get the desired data points and binarize the Y-values
		return new Dataset() {

			@Override
			public FloatMatrix getXTrain() {
				return xTrain.getRows(rowIndizies);
			}

			@Override
			public FloatMatrix getYTrain() {
				return yTrain.getRows(rowIndizies).eq(category);
			}

			@Override
			public FloatMatrix getXTest() {
				return xTest;
			}

			@Override
			public FloatMatrix getYTest() {
				return yTest.eq(category);
			}
		};
	}

	@Override
	public FloatMatrix getXTrain() {
		return xTrain;
	}

	@Override
	public FloatMatrix getYTrain() {
		return yTrain;
	}

	@Override
	public FloatMatrix getXTest() {
		return xTest;
	}

	@Override
	public FloatMatrix getYTest() {
		return yTest;
	}
}
