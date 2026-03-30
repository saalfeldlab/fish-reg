package org.janelia.saalfeldlab.io;

import java.net.URI;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import mpicbg.spim.data.SpimDataException;
import mpicbg.spim.data.sequence.ViewId;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoint;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoints;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.ViewInterestPointLists;

public class InterestPointsToCsv {

	URI xmlURI;
	SpimData2 data;

	String detectionNameMvg;
	String detectionNameFix;
	ViewId viewId;

	public InterestPointsToCsv( URI uri, String detectionNameMvg,  String detectionNameFix, ViewId viewId) {

		try {
			data = new XmlIoSpimData2().load(uri);
		} catch (SpimDataException e) {
			e.printStackTrace();
		}


		this.detectionNameMvg = detectionNameMvg;
		this.detectionNameFix = detectionNameFix;
		this.viewId = viewId;
	}

	public void run() {

		final Map<ViewId, ViewInterestPointLists> iMap = data.getViewInterestPoints().getViewInterestPoints();
		final InterestPoints ipM = iMap.get(viewId).getInterestPointList(detectionNameMvg);
		final InterestPoints ipF = iMap.get(viewId).getInterestPointList(detectionNameFix);

		final List<InterestPoint> ptsM = ipM.getInterestPointsCopy();
		final List<InterestPoint> ptsF = ipF.getInterestPointsCopy();

		if (ptsM.size() != ptsF.size()) {
			System.out.println("points have different lengths " + ptsM.size() + " vs " + ptsF.size());
		}

		for (int i = 0; i < ptsM.size(); i++) {
			System.out.println(String.format("%s,%s,%s,%s", 
					quotes("Pt-"+i),
					quotes("true"),
					print(ptsM.get(i).getL()),
					print(ptsF.get(i).getL())
			));
		}
	}

	public static String print(final double[] x) {
		return Arrays.stream(x)
			.mapToObj(Double::toString)
			.map(InterestPointsToCsv::quotes)
			.collect(Collectors.joining(","));
	}

	public static String quotes(String x) {
		return "\""+x+"\"";
	}

}
