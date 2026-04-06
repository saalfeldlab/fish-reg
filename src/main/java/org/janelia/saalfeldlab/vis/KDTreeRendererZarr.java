package org.janelia.saalfeldlab.vis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;

import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import ij.IJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.KDTree;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealLocalizable;
import net.imglib2.RealPoint;
import net.imglib2.RealRandomAccessible;
import net.imglib2.exception.ImgLibException;
import net.imglib2.img.Img;
import net.imglib2.img.imageplus.ImagePlusImgs;
import net.imglib2.img.imageplus.ShortImagePlus;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.parallel.TaskExecutors;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.realtransform.Scale;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;

public class KDTreeRendererZarr<T extends RealType<T>,P extends RealLocalizable> implements Runnable
{

	@Option(names = {"-i", "--input"}, required = true, description = "Input point ")
	public String input;

	@Option(names = {"-o", "--output"}, required = false, description = "Output image path")
	public String output;

	@Option(names = {"-p", "--psf-radius"}, required = true, description = "Radius for synapse point spread function")
	public double radius;

	@Option(names = {"-r", "--output-resolution"}, required = false, description = "Resolution of output image")
	public String resolution;

	@Option(names = {"-ir", "--input-resolution"}, required = false, description = "Resolution of input points")
	public String inputResolution;
	
	@Option(names = {"-fz", "--flip-z"}, required = false, description = "Whether to flip the z dimension for each input path")
	public String flipZArg;

	@Option(names = {"-s", "--size"}, required = false, description = "Output image size")
	public String sizeString;

	@Option(names = {"-a", "--affine"}, required = false, description = "Affine applied to first set of points")
	public String affineArg;

	@Option(names = {"-q", "--nThreads"}, required = false, description = "Number of threads")
	public int nThreads = 10;

	private double searchDist;

	private double searchDistSqr;

	private double invSquareSearchDistance; 

	private KDTree< T > tree;
	
	public static void main(String[] args) throws ImgLibException, IOException {

		KDTreeRendererZarr inst = new KDTreeRendererZarr();
		int exitCode = new CommandLine(inst).execute(args);
		if (inst.output != null)
			System.exit(exitCode);
	}

	public KDTreeRendererZarr() { }

	public KDTreeRendererZarr( List<T> vals, List<P> pts, double searchDist )
	{
		tree = new KDTree< T >( vals, pts );
		setSearchDist( searchDist );
	}
	
	@Override
	public void run() {

		final BdvOptions opts = BdvOptions.options().numRenderingThreads(nThreads);
		final boolean flipZ = flipZArg == null ? false : flipZArg.toLowerCase().equals("true");
		if (output == null) {

			System.out.println("display " + input);
			System.out.println("flipZ " + flipZ);
			System.out.println(" ");

			BdvStackSource bdv = displayPoints(flipZ, opts);
			opts.addTo(bdv);

		} else {
			final int i = 0;
			final ImagePlus imp = copyToImagePlus(render(flipZ), nThreads);
			IJ.save(imp, output);
		}

		System.out.println("done");
	}

	public RealRandomAccessible<T> getRealRandomAccessible(
			final double searchDist,
			final DoubleUnaryOperator rbf )
	{
		RBFInterpolator.RBFInterpolatorFactory< T > interp = 
				new RBFInterpolator.RBFInterpolatorFactory< T >( 
						rbf, searchDist, false,
						tree.firstElement().copy() );

		return Views.interpolate( tree, interp );
	}
	
	public void setSearchDist( final double searchDist )
	{
		this.searchDist = searchDist;
		searchDistSqr = searchDist * searchDist;
		invSquareSearchDistance = 1.0 / (searchDist * searchDist); 
	}

	public static double rbf( final double rsqr, final double dSqr, 
			final double invDSqr )
	{
		if (rsqr > dSqr)
			return 0;
		else
			return 50 * (1 - (rsqr * invDSqr));
	}

	public BdvStackSource displayPoints(boolean flipZ, BdvOptions opts )
	{
		final AffineTransform3D affine = affineArg != null ? parseAffine(affineArg) : new AffineTransform3D();
		final List<RealPoint> pts = loadZarr(input, 1, flipZ, affine);
		final Interval itvl = boundingMaxInterval(pts);
		System.out.println(Intervals.toString(itvl));
		
		double[] res;
		if( resolution == null )
			res = new double[]{1, 1, 1};
		else
			res = Arrays.stream(resolution.split(",")).mapToDouble(Double::parseDouble).toArray();

		final Scale inputScale;
		if (inputResolution == null) {
			inputScale = new Scale(new double[]{1, 1, 1});
		} else {
			final double[] ptres = Arrays.stream(inputResolution.split(","))
					.mapToDouble(Double::parseDouble).toArray();
			inputScale = new Scale(ptres);
		}

		final List< UnsignedShortType > vals = Stream.iterate( new UnsignedShortType( 1 ), x -> x ).limit( pts.size() )
				.collect( Collectors.toList() );

		System.out.println("radius: " + radius);
		// build renderer
		KDTreeRendererZarr< UnsignedShortType, RealPoint > treeRenderer = new KDTreeRendererZarr< UnsignedShortType, RealPoint >( 
				vals, pts, radius );


		final double dSqr = treeRenderer.searchDistSqr;
		final double idSqr = treeRenderer.invSquareSearchDistance;
		RealRandomAccessible< UnsignedShortType > source = treeRenderer.getRealRandomAccessible( 
				radius, x -> rbf(x, dSqr, idSqr ));

		final String name = input.substring(input.lastIndexOf('/'));
		BdvStackSource<UnsignedShortType> bdv = BdvFunctions.show(source, itvl, name, opts);
		bdv.setDisplayRange(0, 1);
		
		return bdv;
	}
	
	public RandomAccessibleInterval render(boolean flipZ)
	{
		final List<RealPoint> pts = loadZarr(input, 1, flipZ);
		final Interval itvl = boundingMaxInterval(pts);
		System.out.println(Intervals.toString(itvl));
		
		double[] res;
		if (resolution == null)
			res = new double[]{1, 1, 1};
		else
			res = Arrays.stream(resolution.split(",")).mapToDouble(Double::parseDouble).toArray();

		final Scale inputScale;
		if (inputResolution == null) {
			inputScale = new Scale(new double[]{1, 1, 1});
		} else {
			final double[] ptres = Arrays.stream(inputResolution.split(","))
					.mapToDouble(Double::parseDouble).toArray();
			inputScale = new Scale(ptres);
		}

		final List< UnsignedShortType > vals = Stream.iterate( new UnsignedShortType( 1 ), x -> x ).limit( pts.size() )
				.collect( Collectors.toList() );

		System.out.println("radius: " + radius);
		// build renderer
		KDTreeRendererZarr< UnsignedShortType, RealPoint > treeRenderer = new KDTreeRendererZarr< UnsignedShortType, RealPoint >( 
				vals, pts, radius );

		final double dSqr = treeRenderer.searchDistSqr;
		final double idSqr = treeRenderer.invSquareSearchDistance;
		RealRandomAccessible< UnsignedShortType > source = treeRenderer.getRealRandomAccessible( 
				radius, x -> rbf(x, dSqr, idSqr ));

		return Views.interval( Views.raster( source ), itvl );
	}

	public static Interval boundingMaxInterval(List<RealPoint> points) {

		long[] max = new long[3];
		Arrays.fill(max, Long.MIN_VALUE);
		
		for( RealPoint p : points ) {
			
			final double x = p.getDoublePosition(0);
			final double y = p.getDoublePosition(1);
			final double z = p.getDoublePosition(2);

			if (x > max[0])
				max[0] = (long)x;

			if (y > max[1])
				max[1] = (long)y;

			if (z > max[2])
				max[2] = (long)z;
		}
		
		// increment so that the interval includes the max value
		
		max[0]++;
		max[1]++;
		max[2]++;

		return new FinalInterval(max);
	}
	
	public static Interval boundingInterval(List<RealPoint> points) {
		
		long[] min = new long[3];
		Arrays.fill(min, Long.MAX_VALUE);

		long[] max = new long[3];
		Arrays.fill(max, Long.MIN_VALUE);
		
		for( RealPoint p : points ) {
			
			final double x = p.getDoublePosition(0);
			final double y = p.getDoublePosition(1);
			final double z = p.getDoublePosition(2);

			if (x < min[0])
				min[0] = (long)x;
			
			if (y < min[1])
				min[1] = (long)y;

			if (z < min[2])
				min[2] = (long)z;


			if (x > max[0])
				max[0] = (long)x;

			if (y > max[1])
				max[1] = (long)y;

			if (z > max[2])
				max[2] = (long)z;
		}

		return new FinalInterval(min, max);
	}

	public static double[] strToDouble( final String[] s )
	{
		final double[] out = new double[ s.length ];
		for( int i = 0; i < s.length; i++ )
			out[ i ] = Double.parseDouble( s[ i ]);

		return out;
	}

	static boolean[] parseBooleans( final String string )
	{

		if (string == null)
			return new boolean[0];

		final String[] split = string.split(",");
		final boolean[] out = new boolean[split.length];
		int i = 0;
		for (String s : split) {
			if (s.isEmpty() || s.equals("0"))
				out[i++] = false;
			else
				out[i++] = true;
		}

		return out;
	}
	
	static AffineTransform3D parseAffine(final String stringArg )
	{

		final String string;
		final boolean inv = stringArg.startsWith("i");
		if (inv)
			string = stringArg.substring(1);
		else
			string = stringArg;

		final AffineTransform3D affine = new AffineTransform3D();
		if (string == null)
			return affine;

		final double[] params = Arrays.stream(string.split(",")).mapToDouble(Double::parseDouble).toArray();
		affine.set(params);

		if (inv)
			return affine.inverse();
		else
			return affine;
	}

	final static int DIM = 0;
	final static int IDX = 1;

	public static <T extends RealType<T>> List<RealPoint> loadZarr( String path, int subsamplingFactor, boolean flipZ) {

		// use identity affine
		return loadZarr(path, subsamplingFactor, flipZ, new AffineTransform3D());
	}

	public static <T extends RealType<T>> List<RealPoint> loadZarr( String path, int subsamplingFactor, boolean flipZ, AffineTransform3D affine ) {

		System.out.println("load with affine: " + affine);
		final N5Reader zarr = new N5Factory().openReader(StorageFormat.ZARR, path);
		
		@SuppressWarnings("unchecked")
		final Img<T> pointImg = (Img<T>)N5Utils.open(zarr, "");
		final RandomAccess<T> ra = pointImg.randomAccess();

		final int nd = (int)pointImg.dimension(DIM);
		final int N = (int)pointImg.dimension(IDX);
		System.out.println("points: " + Intervals.toString(pointImg));
		
		ArrayList<RealPoint> points = new ArrayList<>();
		for( int i = 0; i < N; i+= subsamplingFactor ) {
			final RealPoint pt = new RealPoint(getPoint(nd, i, flipZ, ra));
			affine.apply(pt, pt);
			points.add(pt);
		}

		return points;
	}

	private static <T extends RealType<T>>  double[] getPoint(int nd, int i, boolean flipZ, RandomAccess<T> ra) {

		final double[] pt = new double[nd];
		ra.setPosition(0, DIM);
		ra.setPosition(i, IDX);

		pt[0] = ra.get().getRealDouble();

		ra.fwd(DIM);
		pt[1] = ra.get().getRealDouble();

		ra.fwd(DIM);
		pt[2] = ra.get().getRealDouble();

		if (flipZ)
			pt[2] *= -1;

		return pt;
	}
	
	public static <T extends RealType<T>> ImagePlus copyToImagePlus( 
			final RandomAccessibleInterval< T > img,
			final int nThreads )
	{
		ShortImagePlus<UnsignedShortType> outImg = ImagePlusImgs.unsignedShorts(Intervals.dimensionsAsLongArray( img ));

		LoopBuilder< BiConsumer< T, UnsignedShortType > > loop;
		if( nThreads == 1)
			loop = LoopBuilder.setImages( img, outImg );
		else
			loop = LoopBuilder.setImages( img, outImg ).multiThreaded( TaskExecutors.fixedThreadPool( nThreads ) );

		loop.forEachPixel( (x,y) -> y.setReal( x.getRealDouble() ));
		return outImg.getImagePlus();
	}


}
