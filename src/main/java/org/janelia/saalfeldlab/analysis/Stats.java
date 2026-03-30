package org.janelia.saalfeldlab.analysis;

import java.util.ArrayList;
import java.util.Collections;

public class Stats {
	
	public final int n;
	public final double average;
	public final double median;
	public final double min;
	public final double q1;
	public final double q3;
	public final double max;
	
	private Stats(
			int n,
			double average,
			double min,
			double q1,
			double median,
			double q3,
			double max
			) {
		
		this.n = n;
		this.average = average;
		this.min = min;
		this.q1 = q1;
		this.median = median;
		this.q3 = q3;
		this.max = max;
	}

	public static Stats compute( ArrayList<Double> data ) {

        if (data == null || data.isEmpty()) {
            System.out.println("No data to analyze");
            return null;
        }
      
        // Create a sorted copy for percentile calculations
        ArrayList<Double> sorted = new ArrayList<>(data);
        Collections.sort(sorted);
        
        int n = sorted.size();
        
        // Calculate average
        double sum = 0;
        for (double val : data) {
            sum += val;
        }
        double average = sum / n;
        
        // Calculate median (Q2)
        double median = getPercentile(sorted, 50);
        
        // Calculate quartiles
        double q1 = getPercentile(sorted, 25);
        double q3 = getPercentile(sorted, 75);
        
        // Get min and max
        double min = sorted.get(0);
        double max = sorted.get(n - 1);

        return new Stats(n, average,
        		min, q1, median, q3, max );
	}

	
	public String toString() {

		StringBuffer out = new StringBuffer();
		out.append("Count:   " + n);
		out.append("\n");
        out.append("Average: " + String.format("%.4f", average));
		out.append("\n");
        out.append("Min:     " + String.format("%.4f", min));
		out.append("\n");
        out.append("Q1:      " + String.format("%.4f", q1));
		out.append("\n");
        out.append("Median:  " + String.format("%.4f", median));
		out.append("\n");
        out.append("Q3:      " + String.format("%.4f", q3));
		out.append("\n");
        out.append("Max:     " + String.format("%.4f", max));
		out.append("\n");
        out.append("Range:   " + String.format("%.4f", (max - min)));
		out.append("\n");
        out.append("IQR:     " + String.format("%.4f", (q3 - q1)));
		out.append("\n");
		return out.toString();
	}
	
	public String printCsvRow() {
		return String.format(
				"%d,%f,%f,%f,%f,%f,%f",
				n, average, min, q1, median, q3, max);
	}

	/**
	 * Calculates a percentile from sorted data
	 * 
	 * @param sorted
	 *            the sorted ArrayList
	 * @param percentile
	 *            the percentile to calculate (0-100)
	 * @return the percentile value
	 */
	private static double getPercentile(ArrayList<Double> sorted, double percentile) {

		if (sorted.isEmpty()) {
			return 0.0;
		}

		int n = sorted.size();
		double index = (percentile / 100.0) * (n - 1);
		int lower = (int)Math.floor(index);
		int upper = (int)Math.ceil(index);

		if (lower == upper) {
			return sorted.get(lower);
		}

		// Linear interpolation
		double weight = index - lower;
		return sorted.get(lower) * (1 - weight) + sorted.get(upper) * weight;
	}

}
