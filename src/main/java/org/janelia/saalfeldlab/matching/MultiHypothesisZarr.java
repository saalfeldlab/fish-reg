package org.janelia.saalfeldlab.matching;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.janelia.saalfeldlab.analysis.Stats;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import org.janelia.saalfeldlab.points.TransformPointsZarr;
import org.janelia.saalfeldlab.vis.KDTreeRendererZarr;
import org.janelia.saalfeldlab.vis.PointPlotter;

import mpicbg.models.AbstractAffineModel3D;
import mpicbg.models.AffineModel3D;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.Model;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.RigidModel3D;
import mpicbg.models.TranslationModel3D;
import mpicbg.spim.data.SpimData;
import mpicbg.spim.data.registration.ViewRegistration;
import mpicbg.spim.data.sequence.ViewId;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RealPoint;
import net.imglib2.img.Img;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.preibisch.legacy.mpicbg.PointMatchGeneric;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoint;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPoints;
import net.preibisch.mvrecon.fiji.spimdata.interestpoints.ViewInterestPointLists;
import net.preibisch.mvrecon.process.interestpointregistration.pairwise.methods.rgldm.RGLDMMatcher;
import bigwarp.landmarks.LandmarkTableModel;
import bigwarp.transforms.BigWarpTransform;
import picocli.CommandLine;
import picocli.CommandLine.Option;

@CommandLine.Command(
	name = "MultiHypothesisZarr",
	mixinStandardHelpOptions = true,
	description = "Registers two sets of 3D points stored in Zarr arrays by finding point correspondences using geometric descriptor matching (RGLDC) and iterative RANSAC."
			+ " Extracts multiple independent consensus sets (multi-hypothesis), fitting a separate transform model to each, and writes the resulting models,"
			+ " inlier correspondences, error statistics, and BigWarp landmark files to disk."
)
public class MultiHypothesisZarr implements Runnable {
	
	static String TRANSLATION = "translation";
	static String AFFINE = "affine";
	static String RIGID = "rigid";

	@Option( names = { "-m", "--moving-zarr" }, required = true )
	String mvgZarrPath;
	
	@Option( names = { "-f", "--fixed-zarr" }, required = true )
	String fixedZarrPath;

	@Option( names = { "-d", "--destination" }, required = true )
	String baseDestination;

	@Option( names = { "-t", "--model-type" }, required = true )
	String modelType;

	@Option( names = { "-tt", "--total-model-type" }, required = false )
	String totalModelType = AFFINE;

	@Option( names = { "-s", "--subsampling-factor" }, required = false )
	int subsamplingFactor = 1;
	
	// matching parameters
	@Option( names = { "--num-neighbors" }, required = false )
	int numNeighbors = 3; // number of neighbors the descriptor is built from

	@Option( names = { "--redundancy" }, required = false )
	int redundancy = 0; // redundancy of the descriptor (adds more neighbors and tests all combinations)

	@Option( names = { "--ratio-of-distance" }, required = false )
	float ratioOfDistance = 4.0f; // how much better the best than the second best descriptor need to be

	@Option( names = { "--search-radius" }, required = false )
	float searchRadius = 30.0f; // the search radius

	boolean limitSearchRadius = true; // limit search to a radius

	@Option( names = { "--min-correspondences" }, required = false )
	int minNumCorrespondences = 5;

	@Option( names = { "--num-iterations" }, required = false )
	int numIterations = 1000;

	@Option( names = { "--max-epsilon" }, required = false )
	double maxEpsilon = 30;

	@Option( names = { "--min-inlier-ratio" }, required = false )
	double minInlierRatio = 0.1;

	@Option( names = { "--apply" }, required = false )
	boolean applyToPoints;

	@Option( names = { "--render" }, required = false )
	boolean render;

	@Option( names = { "--vis-points" }, required = false )
	boolean visPoints;

	@Option( names = { "--image-orientation" }, required = false )
	String imageOrientation = "XY";

	File allLandmarksFile;
	String transformedPointsPath;

	String renderedImagePath;

	String pointImgPath;

	public static void main(String[] args) {
		int exitCode = new CommandLine(new MultiHypothesisZarr()).execute(args);
		System.exit(exitCode);
	}
	
	private static Model<?> getModel( String modelType ) {

		final String modelTypeNorm = modelType.toLowerCase();
		if( modelTypeNorm.equals(TRANSLATION))
			return new TranslationModel3D();
		else if( modelTypeNorm.equals(RIGID))
			return new RigidModel3D();
		else if( modelTypeNorm.equals(AFFINE))
			return new AffineModel3D();
		else
		{
			System.out.println("Unknown model type: " + modelType);
			return null;
		}
	}

	final static int DIM = 0;
	final static int IDX = 1;

	public static <T extends RealType<T>> List<InterestPoint> loadZarr( String path, int subsamplingFactor ) {

		final N5Reader zarr = new N5Factory().openReader(StorageFormat.ZARR, path);

		@SuppressWarnings("unchecked")
		final Img<T> pointImg = (Img<T>)N5Utils.open(zarr, "");
		final RandomAccess<T> ra = pointImg.randomAccess();

		final int nd = (int)pointImg.dimension(DIM);
		final int N = (int)pointImg.dimension(IDX);
		System.out.println(Intervals.toString(pointImg));
		
		ArrayList<InterestPoint> points = new ArrayList<>();
		for( int i = 0; i < N; i+= subsamplingFactor ) {
			points.add(new InterestPoint(i, getPoint(nd, i, ra)));
		}

		return points;
	}

	private static <T extends RealType<T>>  double[] getPoint(int nd, int i, RandomAccess<T> ra) {

		final double[] pt = new double[nd];
		ra.setPosition(0, DIM);
		ra.setPosition(i, IDX);

		pt[0] = ra.get().getRealDouble();

		ra.fwd(DIM);
		pt[1] = ra.get().getRealDouble();

		ra.fwd(DIM);
		pt[2] = ra.get().getRealDouble();

		return pt;
	}
	
	public void makeBaseDir() {
		new File(baseDestination).getParentFile().mkdirs();
	}
	
	public void applyToPoints() {

		System.out.println("apply transform to points");
		TransformPointsZarr tf = new TransformPointsZarr();
		tf.landmarksArg  = allLandmarksFile;
		tf.transformType = BigWarpTransform.TPS;
		tf.zarrPath = mvgZarrPath;
		tf.outputZarrPath = transformedPointsPath;
		tf.run();
	}

	public void renderToImage() {

		System.out.println("render to image");
		KDTreeRendererZarr renderer = new KDTreeRendererZarr();
		renderer.input = transformedPointsPath + "/points";
		renderer.output = renderedImagePath;
		renderer.radius = 4;
		renderer.run(); 
	}

	@Override
	public void run() {
		
		makeBaseDir();

		// get interest points
		final List<InterestPoint> ipMoving = loadZarr(mvgZarrPath, subsamplingFactor);
		final List<InterestPoint> ipFixed = loadZarr(fixedZarrPath, subsamplingFactor);
		
		System.out.println("num moving pts: " + ipMoving.size());
		System.out.println("num fixed pts: " + ipFixed.size());
		System.out.println("");

		limitSearchRadius = searchRadius > 0f;

		System.out.println("parameters:");
		System.out.println("using model type: " + modelType);
		System.out.println("numNeighbors " + numNeighbors);
		System.out.println("redundancy " + redundancy);
		System.out.println("ratioOfDistance " + ratioOfDistance);
		System.out.println("limitSearchRadius " + limitSearchRadius);
		System.out.println("searchRadius " + searchRadius);
		System.out.println("minNumCorrespondences " + minNumCorrespondences);
		System.out.println("numIterations " + numIterations);
		System.out.println("maxEpsilon " + maxEpsilon);
		System.out.println("minInlierRatio " + minInlierRatio);



		transformedPointsPath = String.format("%s_transformed.zarr", baseDestination);
		renderedImagePath = String.format("%s_transformed_image.tif", baseDestination);

		Model model = getModel(modelType);

		PrintWriter modelWriter;
		PrintWriter statsWriter;
		try {

			modelWriter = new PrintWriter(new FileWriter(
					new File(String.format("%s_%s-models.csv", baseDestination, modelType))));

			statsWriter = new PrintWriter(new FileWriter(
					new File(String.format("%s_%s-stats.csv", baseDestination, modelType))));

		} catch (IOException e) {
			e.printStackTrace();
			return;
		}

		final RGLDMMatcher<InterestPoint> matcher = new RGLDMMatcher<>();
		List<PointMatchGeneric<InterestPoint>> candidates = matcher.extractCorrespondenceCandidates(
				ipMoving,
				ipFixed,
				numNeighbors,
				redundancy,
				ratioOfDistance,
				Float.MAX_VALUE,
				limitSearchRadius,
				searchRadius);

		System.out.println("Found " + candidates.size() + " correspondence candidates.");

		// perform RANSAC (warning: not safe for multi-threaded over pairs of
		// images, this needs point duplication)
		ArrayList<PointMatchGeneric<InterestPoint>> inliers = new ArrayList<>();

		final ArrayList<PointMatchGeneric<InterestPoint>> allMatches = new ArrayList<>();
		final ArrayList<Integer> inlierSetSizes = new ArrayList<>();
		final ArrayList<Stats> statsPerModel = new ArrayList<>();

		final HashSet<Integer> movingIds = new HashSet<>();
		final HashSet<Integer> fixedIds = new HashSet<>();

		int consensusSetId = 0;
		boolean multiConsenus = true;
		boolean modelFound = false;
		do {
			inliers.clear();
			try {
				modelFound = model.filterRansac(
						candidates,
						inliers,
						numIterations,
						maxEpsilon, minInlierRatio);

			} catch (NotEnoughDataPointsException e) {
				System.out.println("Not enough points for matching. stopping.");
				break;
			}

			if (modelFound && inliers.size() >= minNumCorrespondences) {

				// highly suggested in general
				// inliers = RANSAC.removeInconsistentMatches( inliers );

				System.out.println("Found " + inliers.size() + "/" + candidates.size() + " inliers with model: " + model);

				allMatches.addAll(inliers);
				inlierSetSizes.add(inliers.size());
				for (int i = 0; i < inliers.size(); i++) {
					movingIds.add(inliers.get(i).getPoint1().getId());
					fixedIds.add(inliers.get(i).getPoint2().getId());
				}

				writeModel(modelWriter, consensusSetId, (AbstractAffineModel3D<?>)model);

				final ArrayList<Double> errors = errors(model, inliers);
				final Stats stats = Stats.compute(errors);
				statsPerModel.add(stats);
				statsWriter.println(String.format("%d,%s", consensusSetId, stats.printCsvRow()));
				
				System.out.println("  writing BigWarp landmarks");
				writeBigWarpLandmarks(new File(baseDestination + "_landmarks_" + consensusSetId + ".csv"), inliers);

				consensusSetId++;

				if (multiConsenus)
					candidates = removeInliers(candidates, inliers, movingIds, fixedIds);

			} else if (modelFound) {
				System.out.println("Model found, but not enough points (" + inliers.size() + "/" + minNumCorrespondences + ").");
			} else {
				System.out.println("NO model found.");
			}
		} while (multiConsenus && modelFound && inliers.size() >= minNumCorrespondences);

		try {

			if (allMatches.size() > 4) {
				Model<?> totalModel = getModel(totalModelType);
				System.out.println("Fitting model " + totalModel.getClass() + " with all points, total: " + allMatches.size());
				totalModel.fit(allMatches);
				writeModel(modelWriter, -1, (AbstractAffineModel3D<?>)totalModel);

				final ArrayList<Double> errors = errors(model, allMatches);
				final Stats stats = Stats.compute(errors);
				statsPerModel.add(stats);
				statsWriter.println(String.format("%d,%s", -1, stats.printCsvRow()));

				allLandmarksFile = new File(baseDestination + "_landmarks_all.csv");
				writeBigWarpLandmarks(allLandmarksFile, allMatches);

				if (applyToPoints) {
					applyToPoints();
				}

				if (render) {
					renderToImage();
				}

			}
		} catch (NotEnoughDataPointsException e) {
			e.printStackTrace();
		} catch (IllDefinedDataPointsException e) {
			e.printStackTrace();
		}

		if (visPoints) {
			pointImgPath = String.format("%s_%s-pts-vis-%s.png", baseDestination, modelType, imageOrientation);
			System.out.println("writing point image: " + pointImgPath);
			PointPlotter.makeImage(pointImgPath, allMatches, inlierSetSizes, imageOrientation);
		}

		modelWriter.close();
		statsWriter.close();
	}

	static ArrayList<Double> errors(
			Model model, List< PointMatchGeneric< InterestPoint > > inliers ) {

		int N = inliers.size();
		ArrayList<Double> errors = new ArrayList<>();
		for (int i = 0; i < N; i++)
			errors.add(error(model, inliers.get(i)));
		
		return errors;
	}

	static double error(
			Model model, PointMatchGeneric< InterestPoint > match ) {

		return distance(match.getPoint1().getW(), match.getPoint2().getW());
	}
	
	static double distance( final double[] position1, final double[] position2 ) {
		double dist = 0;

		for ( int d = 0; d < position1.length; ++d ) {
			final double pos = position2[ d ] - position1[ d ];
			dist += pos * pos;
		}

		return Math.sqrt( dist );
	}

	static void writeModel(final PrintWriter writer, int id, AbstractAffineModel3D<?> model ) {
		final double[] params = new double[12];
		model.getMatrix(params);
		writer.println( String.format("%d,%s",
				id, print(params)));
	}
	
	static String print(final double[] x) {
		return Arrays.stream(x)
			.mapToObj(Double::toString)
			.collect(Collectors.joining(","));
	}

	public static void writeBigWarpLandmarks( File f, ArrayList<PointMatchGeneric<InterestPoint>> inliers ) {

		// Bigwarp creates transforms from target space to moving space 
		// because it applies the transformation to images
		// 
		// in this context we want the opposite,
		// so swap the roles of moving and fixed when exporting

		final LandmarkTableModel ltm = new LandmarkTableModel( 3 );
		int i = 0;
		for( final PointMatchGeneric<InterestPoint> match : inliers ) {
	
			ltm.add( match.getPoint2().getL(), match.getPoint1().getL() );
			int id1 = match.getPoint2().getId();
			int id2 = match.getPoint1().getId();
			ltm.setColumnName(i, "Pt-" + i + "(" + id1 + "," + id2 + ")");
			i++;
		}
		try {
			ltm.save( f );
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static AffineTransform3D getTransform( final SpimData data, final ViewId viewId )
	{
		final Map<ViewId, ViewRegistration> rMap = data.getViewRegistrations().getViewRegistrations();
		final ViewRegistration reg = rMap.get( viewId );
		reg.updateModel();
		return reg.getModel();
	}

	public static List<InterestPoint> getInterestPoints( final SpimData2 data, final ViewId viewId, final String label )
	{
		final Map<ViewId, ViewInterestPointLists> iMap = data.getViewInterestPoints().getViewInterestPoints();
		final ViewInterestPointLists iplists = iMap.get( viewId );

		// this is net.preibisch.mvrecon.fiji.spimdata.interestpoints.InterestPointsN5
		final InterestPoints ips = iplists.getInterestPointList( label );

		// load interest points
		return ips.getInterestPointsCopy();
	}

	public static List<InterestPoint> overlappingPoints( final List<InterestPoint> ip1, final Interval intervalImg2, final AffineTransform3D t2 )
	{
		// use the inverse affine transform of the other view to map the points into the local interval of img2
		final AffineTransform3D t2inv = t2.inverse();

		final RealPoint p = new RealPoint( intervalImg2.numDimensions() );

		return ip1.stream().filter( ip -> {
			ip.localize( p );
			t2inv.apply( p, p );
			return Intervals.contains( intervalImg2 , p );
		} ).collect( Collectors.toList() );
	}

	public static List<PointMatchGeneric<InterestPoint>> removeInliers(
			final List<PointMatchGeneric<InterestPoint>> candidates, 
			final List<PointMatchGeneric<InterestPoint>> matches,
			final Set<Integer> movingIds, final Set<Integer> fixedIds) {

		System.out.println("removing inliers. num init candidates " + candidates.size());
		final HashSet<PointMatchGeneric<InterestPoint>> matchesSet = new HashSet<>(matches);
		
		// not only do we have to remove candidates, but we also need to remove
		// other candidates that use the same points.
		// so track the ids of points that have been used
		for (final PointMatchGeneric<InterestPoint> match : matches) {
			movingIds.add(match.getPoint1().getId());
			fixedIds.add(match.getPoint2().getId());
		}	

		final ArrayList<PointMatchGeneric<InterestPoint>> res = new ArrayList<>();
		for (int i = 0; i < candidates.size(); i++) {

			final PointMatchGeneric<InterestPoint> match = candidates.get(i);
			final boolean hasConflictingId = 
					movingIds.contains(match.getPoint1().getId()) ||
					fixedIds.contains(match.getPoint2().getId());

			if( !matchesSet.contains(match) && !hasConflictingId) {
				res.add(match);
			}
		}

		System.out.println("removing inliers. num final candidates " + res.size());
		return res;
	}

}
