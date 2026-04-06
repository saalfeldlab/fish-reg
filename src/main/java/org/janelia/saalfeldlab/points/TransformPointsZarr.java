package org.janelia.saalfeldlab.points;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.blosc.BloscCompression;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;

import bigwarp.landmarks.LandmarkTableModel;
import bigwarp.transforms.BigWarpTransform;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.realtransform.InvertibleRealTransform;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Intervals;
import picocli.CommandLine;
import picocli.CommandLine.Option;

public class TransformPointsZarr implements Runnable {

	@Option( names = { "-i", "--zarr-points" }, required = true,
			description = "Path to the input Zarr array of 3D points (shape: [3, N])." )
	String zarrPath;

	@Option( names = { "-a", "--affine" }, required = false,
			description = "Affine transform as 12 comma-separated values (row-major 3x4 matrix). Mutually exclusive with --landmarks." )
	String affineTransformArg;

	@Option( names = { "-l", "--landmarks" }, required = false,
			description = "Path to a BigWarp landmarks CSV file defining a transform. Mutually exclusive with --affine." )
	File landmarksArg;

	@Option( names = { "-t", "--type" }, required = false,
			description = "Transform type to derive from the landmarks (e.g. 'Thin Plate Spline', 'Affine'). Only used with --landmarks." )
	String transformType;

	@Option( names = { "--inv"}, required = false,
			description = "Invert the transform before applying it." )
	boolean invert;

	@Option( names = { "-o", "--output" }, required = true,
			description = "Path for the output Zarr array of transformed 3D points." )
	String outputZarrPath;

	public static void main(String[] args) {
		int exitCode = new CommandLine(new TransformPointsZarr()).execute(args);
		System.exit(exitCode);
	}

	final static int DIM = 0;
	final static int IDX = 1;

	int nd;
	int N;

	private InvertibleRealTransform transform;

	private Map<Integer, Integer> correspondenceIds;

	public <T extends RealType<T>> Img<T> loadZarr(String path) {

		final N5Reader zarr = new N5Factory().openReader(StorageFormat.ZARR, path);

		@SuppressWarnings("unchecked")
		final Img<T> pointImg = (Img<T>)N5Utils.open(zarr, "");

		nd = (int)pointImg.dimension(DIM);
		N = (int)pointImg.dimension(IDX);
		System.out.println("points: " + Intervals.toString(pointImg));
		return pointImg;
	}

	public <T extends RealType<T> & NativeType<T>> void saveZarr(String path, Img<T> points) {

		final N5Writer zarr = new N5Factory().openWriter(StorageFormat.ZARR, outputZarrPath);
		int[] blkSz = Arrays.stream(points.dimensionsAsLongArray()).mapToInt(x -> (int)x).toArray();
		N5Utils.save(points, zarr, "points", blkSz, new BloscCompression());

		if (correspondenceIds != null) {
			final int[] correspondences = new int[(int)points.dimension(IDX)];
			Arrays.fill(correspondences, -1);
			for (Entry<Integer, Integer> ids : correspondenceIds.entrySet()) {
				correspondences[ids.getKey()] = ids.getValue();
			}
			ArrayImg<IntType, IntArray> corrImg = ArrayImgs.ints(correspondences, N);
			N5Utils.save(corrImg, zarr, "correspondences", new int[]{N}, new BloscCompression());
		}
	}

	public <S extends RealType<S>,T extends RealType<T>> void transform( Img<S> src, Img<T> dst ) {

		final double[] x = new double[3];
		final double[] y = new double[3];

		RandomAccess<S> srcRa = src.randomAccess();
		RandomAccess<T> dstRa = dst.randomAccess();

		for(int i = 0; i < N; i++) {

			srcRa.setPosition(0, DIM);
			srcRa.setPosition(i, IDX);
			x[0] = srcRa.get().getRealDouble();

			srcRa.fwd(DIM);
			x[1] = srcRa.get().getRealDouble();

			srcRa.fwd(DIM);
			x[2] = srcRa.get().getRealDouble();

			transform.apply(x, y);

			dstRa.setPosition(0, DIM);
			dstRa.setPosition(i, IDX);
			dstRa.get().setReal(y[0]);

			dstRa.fwd(DIM);
			dstRa.get().setReal(y[1]);

			dstRa.fwd(DIM);
			dstRa.get().setReal(y[2]);
		}

	}

	private static <T extends RealType<T>> AffineTransform3D parseAffine(String arg) {

		final double[] params = Arrays.stream(arg.split(",")).map(String::trim)
				.mapToDouble(Double::parseDouble) .toArray();

		final AffineTransform3D result = new AffineTransform3D();
		result.set(params);
		return result;
	}

	static final Pattern ID_PATTERN = Pattern.compile("\\((\\d+),(\\d+)\\)");

	private InvertibleRealTransform parseLandmarks() {

		LandmarkTableModel ltm = new LandmarkTableModel(3);
		try {
			ltm.load(landmarksArg);

			final ArrayList<String> names = ltm.getNames();
			final boolean hasIds = names.stream().allMatch(n -> ID_PATTERN.matcher(n).find());
			if (hasIds) {
				correspondenceIds = new HashMap<>();
				for (final String name : names) {

					final Matcher m = ID_PATTERN.matcher(name);
					m.find(); // advance the matcher
					if (invert)
						correspondenceIds.put(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)));
					else
						correspondenceIds.put(Integer.parseInt(m.group(2)), Integer.parseInt(m.group(1)));
				}
			}

			BigWarpTransform bwt = new BigWarpTransform(ltm, transformType);
			return bwt.getTransformation();
		} catch (IOException e) {
			System.err.println("Error: failed to load landmarks file: " + landmarksArg);
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public void run() {

		if (affineTransformArg != null && landmarksArg != null) {
			System.err.println("Error: only one of --affine or --tps may be specified.");
			return;
		}
		if (affineTransformArg == null && landmarksArg == null) {
			System.err.println("Error: one of --affine or --tps must be specified.");
			return;
		}
		if (landmarksArg != null && !landmarksArg.exists()) {
			System.err.println("Error: landmarks file does not exist: " + landmarksArg);
			return;
		}


		if( affineTransformArg!= null)
			transform = parseAffine(affineTransformArg);
		else
			transform = parseLandmarks();

		if (invert) {
			System.out.println("inverting");
			transform = transform.inverse();
		}

		// sum of point positions
		Img<RealType> points = loadZarr(zarrPath);
		ArrayImg<DoubleType, DoubleArray> transformedPoints = ArrayImgs.doubles(points.dimensionsAsLongArray());

		transform(points, transformedPoints);
		saveZarr(outputZarrPath, transformedPoints);
		System.out.println("done");
	}

}
