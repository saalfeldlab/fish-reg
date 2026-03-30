package org.janelia.saalfeldlab.points;

import java.util.ArrayList;

import org.janelia.saalfeldlab.vis.PointPlotter.Projection;

import net.preibisch.legacy.mpicbg.PointMatchGeneric;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoint;

public class CategorizedPoints {

	private double[][] points;
	private int[] categories;

	public CategorizedPoints(double[][] pts, int[] cats) {

		this.points = pts;
		this.categories = cats;
	}

	public static CategorizedPoints from(final Projection projection,
			final ArrayList<PointMatchGeneric<InterestPoint>> allMatches, ArrayList<Integer> inlierSetSizes) {

		int N = allMatches.size();
		double[][] pts = new double[N][2];
		int[] categories = new int[N];

		int id = 0;
		int cumulativeSize = inlierSetSizes.get(0);

		for (int i = 0; i < N; i++) {

			if (i >= cumulativeSize) {
				id++;
				cumulativeSize += inlierSetSizes.get(id);
			}

			if (projection == Projection.XY) {
				pts[i][0] = allMatches.get(i).getPoint1().getL()[0];
				pts[i][1] = allMatches.get(i).getPoint1().getL()[1];
			} else if (projection == Projection.XZ) {
				pts[i][0] = allMatches.get(i).getPoint1().getL()[0];
				pts[i][1] = allMatches.get(i).getPoint1().getL()[2];
			} else if (projection == Projection.YZ) {
				pts[i][0] = allMatches.get(i).getPoint1().getL()[1];
				pts[i][1] = allMatches.get(i).getPoint1().getL()[2];
			}

			categories[i] = id;
		}

		return new CategorizedPoints(pts, categories);
	}

	public double[][] getPoints() {

		return points;
	}

	public int[] getCategories() {

		return categories;
	}

}

