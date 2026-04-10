package org.janelia.saalfeldlab.points;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;

public class ElastixPointConverterTest {

	@Rule
	public TemporaryFolder tmp = new TemporaryFolder();

	static final int DIM = ElastixPointConverter.DIM;
	static final int IDX = ElastixPointConverter.IDX;

	// Synthetic points: 4 points in 3D, stored as [nd=3, N=4] Img
	static final ArrayImg<DoubleType, DoubleArray> POINTS = makePointImg();

	private static ArrayImg<DoubleType, DoubleArray> makePointImg() {

		final int nd = 3;
		final int N = 4;
		final double[] data = {
			  1.0,   2.0,   3.0,
			 10.5,  20.5,  30.5,
			 -4.0,   0.0,  99.9,
			  0.25, -0.5,   1.5,
		};
		return ArrayImgs.doubles(data, nd, N);
	}

	// -- helpers --

	private static <T extends RealType<T> & NativeType<T>> Img<T> readZarr(String path) throws IOException {
		try (final N5Reader zarr = new N5Factory().openReader(StorageFormat.ZARR, path)) {
			return N5Utils.open(zarr, "");
		}
	}

	private static double[][] readTransformixFile(File f) throws IOException {
		try (final BufferedReader r = new BufferedReader(new FileReader(f))) {
			r.readLine(); // "point"
			final int N = Integer.parseInt(r.readLine().trim());
			final double[][] pts = new double[N][];
			for (int i = 0; i < N; i++) {
				final String[] parts = r.readLine().trim().split("\\s+");
				pts[i] = new double[parts.length];
				for (int d = 0; d < parts.length; d++)
					pts[i][d] = Double.parseDouble(parts[d]);
			}
			return pts;
		}
	}

	private static double[][] toDoubleArray(final Img<? extends RealType<?>> img) {
		final int nd = (int) img.dimension(DIM);
		final int N = (int) img.dimension(IDX);
		final RandomAccess<? extends RealType<?>> ra = img.randomAccess();
		final double[][] pts = new double[N][nd];
		for (int i = 0; i < N; i++) {
			for (int d = 0; d < nd; d++) {
				ra.setPosition(d, DIM);
				ra.setPosition(i, IDX);
				pts[i][d] = ra.get().getRealDouble();
			}
		}
		return pts;
	}

	private static ElastixPointConverter converter(String input, String output) {
		final ElastixPointConverter c = new ElastixPointConverter();
		c.inputPath = input;
		c.outputPath = output;
		return c;
	}

	private static void assertPointsEqual(double[][] expected, double[][] actual) {
		assertEquals("number of points", expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) {
			assertEquals("dimensionality of point " + i, expected[i].length, actual[i].length);
			for (int d = 0; d < expected[i].length; d++)
				assertEquals("point " + i + " dim " + d, expected[i][d], actual[i][d], 1e-9);
		}
	}

	// -- tests --

	@Test
	public void testZarrToTransformixAndBack() throws IOException {
		final Path dir = tmp.newFolder().toPath();
		final String zarrA = dir.resolve("points_a.zarr").toString();
		final String txtB  = dir.resolve("points_b.txt").toString();
		final String zarrC = dir.resolve("points_c.zarr").toString();

		// write POINTS img directly to Zarr
		final ElastixPointConverter toZarr = converter(null, zarrA);
		toZarr.setInputImg(POINTS);
		toZarr.setOutputToZarr(true);
		toZarr.run();

		// Zarr → transformix text → Zarr
		converter(zarrA, txtB).run();
		converter(txtB, zarrC).run();

		assertPointsEqual(toDoubleArray(POINTS), toDoubleArray(readZarr(zarrC)));
	}

	@Test
	public void testTransformixToZarrAndBack() throws IOException {
		final Path dir = tmp.newFolder().toPath();
		final File   txtA  = dir.resolve("points_a.txt").toFile();
		final String zarrB = dir.resolve("points_b.zarr").toString();
		final File   txtC  = dir.resolve("points_c.txt").toFile();

		// write POINTS img directly to transformix text
		final ElastixPointConverter toTxt = converter(null, txtA.toString());
		toTxt.setInputImg(POINTS);
		toTxt.run();

		// transformix text → Zarr → transformix text
		converter(txtA.toString(), zarrB).run();
		converter(zarrB, txtC.toString()).run();

		assertPointsEqual(toDoubleArray(POINTS), readTransformixFile(txtC));
	}
}
