package org.janelia.saalfeldlab.points;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.blosc.BloscCompression;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;

import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import picocli.CommandLine;
import picocli.CommandLine.Option;

@CommandLine.Command(
		name = "ElastixPointConverter", mixinStandardHelpOptions = true,
		description = "Converts 3D points between a Zarr array (shape [nd, N]) and the transformix text format. "
				+ "Direction is inferred automatically: if -i is a directory it is treated as Zarr input; "
				+ "otherwise it is treated as a transformix text file."
	)
public class ElastixPointConverter implements Runnable {

	@Option(names = {"-i"}, required = true)
	public String inputPath;

	@Option(names = {"-o"}, required = true)
	public String outputPath;

	static final int DIM = 0;
	static final int IDX = 1;

	private Img<? extends RealType<?>> inputImg = null;
	private boolean outputToZarr = false;

	public <T extends RealType<T> & NativeType<T>> void setInputImg(final Img<T> img) {
		this.inputImg = img;
	}

	public void setOutputToZarr(final boolean outputToZarr) {
		this.outputToZarr = outputToZarr;
	}

	public static void main(String[] args) {
		int exitCode = new CommandLine(new ElastixPointConverter()).execute(args);
		System.exit(exitCode);
	}

	@Override
	public void run() {
		if (inputImg != null) {
			if (outputToZarr)
				toZarr(inputImg);
			else
				toTransformix(inputImg);
		} else if (new File(inputPath).isDirectory()) {
			zarrToTransformix();
		} else {
			transformixToZarr();
		}
	}

	private <T extends RealType<T> & NativeType<T>> void zarrToTransformix() {

		try (final N5Reader zarr = new N5Factory().openReader(StorageFormat.ZARR, inputPath)) {
			final Img<T> img = N5Utils.open(zarr, "");
			toTransformix(img);
		}
	}

	@SuppressWarnings("unchecked")
	private <T extends RealType<T> & NativeType<T>> void toTransformix(final Img<?> img) {
		zarrToTransformixTyped((Img<T>) img);
	}

	@SuppressWarnings("unchecked")
	private <T extends RealType<T> & NativeType<T>> void toZarr(final Img<?> img) {
		toZarrTyped((Img<T>) img);
	}

	private <T extends RealType<T> & NativeType<T>> void toZarrTyped(final Img<T> img) {
		final int[] blkSz = Arrays.stream(img.dimensionsAsLongArray()).mapToInt(x -> (int) x).toArray();
		try (final N5Writer zarr = new N5Factory().openWriter(StorageFormat.ZARR, outputPath)) {
			N5Utils.save(img, zarr, "", blkSz, new BloscCompression());
		}
	}

	private <T extends RealType<T>> void zarrToTransformixTyped(final Img<T> img) {

		final int nd = (int) img.dimension(DIM);
		final int N = (int) img.dimension(IDX);
		final RandomAccess<T> ra = img.randomAccess();

		try (final PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
			writer.println("point");
			writer.println(N);
			for (int i = 0; i < N; i++) {
				ra.setPosition(i, IDX);
				final StringBuilder sb = new StringBuilder();
				for (int d = 0; d < nd; d++) {
					ra.setPosition(d, DIM);
					if (d > 0)
						sb.append(' ');
					sb.append(ra.get().getRealDouble());
				}
				writer.println(sb);
			}
			System.out.println("Wrote " + N + " points to " + outputPath);
		} catch (IOException e) {
			System.err.println("Error writing transformix file: " + outputPath);
			e.printStackTrace();
		}
	}

	private void transformixToZarr() {

		try (final BufferedReader reader = new BufferedReader(new FileReader(inputPath))) {

			final String header = reader.readLine();
			if (header == null || !header.trim().equals("point")) {
				System.err.println("Error: expected 'point' header, got: " + header);
				return;
			}

			final int N = Integer.parseInt(reader.readLine().trim());

			// read first line to determine dimensionality
			final String firstLine = reader.readLine();
			if (firstLine == null) {
				System.err.println("Error: no points found after header.");
				return;
			}
			final double[] firstCoords = parseCoords(firstLine);
			final int nd = firstCoords.length;

			final ArrayImg<DoubleType, DoubleArray> img = ArrayImgs.doubles(nd, N);
			final RandomAccess<DoubleType> ra = img.randomAccess();

			writeCoordsToImg(ra, nd, 0, firstCoords);

			for (int i = 1; i < N; i++) {
				final String line = reader.readLine();
				if (line == null) {
					System.err.println("Warning: expected " + N + " points, found only " + i);
					break;
				}
				writeCoordsToImg(ra, nd, i, parseCoords(line));
			}

			toZarrTyped(img);

		} catch (IOException e) {
			System.err.println("Error reading transformix file: " + inputPath);
			e.printStackTrace();
		}
	}

	private static double[] parseCoords(final String line) {
		return Arrays.stream(line.trim().split("\\s+"))
				.mapToDouble(Double::parseDouble)
				.toArray();
	}

	private static void writeCoordsToImg(
			final RandomAccess<DoubleType> ra, final int nd, final int i, final double[] coords) {
		ra.setPosition(i, IDX);
		for (int d = 0; d < nd; d++) {
			ra.setPosition(d, DIM);
			ra.get().set(coords[d]);
		}
	}
}
