using System;
using System.Linq;

namespace NDimensionalClusterExecution
{
    class Program
    {
        static void Main()
        {
            // === INPUT PLACEHOLDERS ===
            double[][] rawData = {
                new[] { 1.0, 2.0 },
                new[] { 1.5, 1.8 },
                new[] { 1.2, 2.2 },
                new[] { 5.0, 8.0 },
                new[] { 6.0, 8.5 },
                new[] { 5.5, 7.5 },
                new[] { 2.0, 2.0 },
                new[] { 2.2, 2.1 },
                new[] { 1.8, 1.9 }
            };
            int K = 3;                  // number of clusters
            double densityRadius = 0.2; // radius for density check
            int maxIter = 100;          // max K-means iterations

            // === RUN THE CLUSTER PIPELINE ===
            // Returns K base vertices + 1 apex vertex in ℝⁿ⁺¹
            var vertices = ComputeDensityNormalizedKMeansPipeline(
                rawData, K, densityRadius, maxIter);

            // === COMPUTE & PRINT TENSOR, VELOCITY, AND THEIR INVERSES ===
            ComputeTensorAndVelocity(vertices);
        }

        static double[][] ComputeDensityNormalizedKMeansPipeline(
            double[][] raw, int K, double radius, int maxIter)
        {
            int n = raw.Length;
            int dim = raw[0].Length;

            Console.WriteLine("=== RAW DATA ===");
            for (int i = 0; i < n; i++)
                Console.WriteLine($"Point {i}: ({string.Join(", ", raw[i].Select(x => x.ToString("F16")))})");

            // 1) Min–Max normalize to [0,1]
            double[] minVals = new double[dim], maxVals = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                minVals[d] = raw.Min(p => p[d]);
                maxVals[d] = raw.Max(p => p[d]);
            }

            Console.WriteLine($"\n=== NORMALIZATION PARAMETERS ===");
            Console.WriteLine($"Min values: ({string.Join(", ", minVals.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Max values: ({string.Join(", ", maxVals.Select(x => x.ToString("F16")))})");

            double[][] data = new double[n][];
            for (int i = 0; i < n; i++)
            {
                data[i] = new double[dim];
                for (int d = 0; d < dim; d++)
                    data[i][d] = (raw[i][d] - minVals[d]) / (maxVals[d] - minVals[d]);
            }

            Console.WriteLine("\n=== NORMALIZED DATA ===");
            for (int i = 0; i < n; i++)
                Console.WriteLine($"Norm Point {i}: ({string.Join(", ", data[i].Select(x => x.ToString("F16")))})");

            // 2) Local density count
            int[] density = new int[n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (i != j && Euclid(data[i], data[j]) <= radius)
                        density[i]++;

            Console.WriteLine("\n=== DENSITY VALUES ===");
            for (int i = 0; i < n; i++)
                Console.WriteLine($"Point {i} density: {density[i]}");

            // 3) Seed centroids at top-density points
            var seeds = density
                .Select((d, idx) => new { d, idx })
                .OrderByDescending(x => x.d)
                .Take(K)
                .Select(x => x.idx)
                .ToArray();

            Console.WriteLine($"\n=== DENSITY-BASED SEEDS ===");
            Console.WriteLine($"Top {K} density points selected as seeds: {string.Join(", ", seeds)}");

            double[][] centroids = new double[K][];
            for (int c = 0; c < K; c++)
                centroids[c] = (double[])data[seeds[c]].Clone();

            int[] labels = new int[n];
            bool moved;
            int iter = 0;

            // 4) K-means on normalized data
            do
            {
                moved = false;

                // Assign step
                for (int i = 0; i < n; i++)
                {
                    double best = Euclid(data[i], centroids[0]);
                    int cls = 0;
                    for (int c = 1; c < K; c++)
                    {
                        double d = Euclid(data[i], centroids[c]);
                        if (d < best) { best = d; cls = c; }
                    }
                    if (labels[i] != cls)
                    {
                        labels[i] = cls;
                        moved = true;
                    }
                }

                // Update step
                var sums = new double[K][];
                var counts = new int[K];
                for (int c = 0; c < K; c++)
                    sums[c] = new double[dim];
                for (int i = 0; i < n; i++)
                {
                    int cls = labels[i];
                    counts[cls]++;
                    for (int d = 0; d < dim; d++)
                        sums[cls][d] += data[i][d];
                }
                for (int c = 0; c < K; c++)
                    if (counts[c] > 0)
                        for (int d = 0; d < dim; d++)
                            centroids[c][d] = sums[c][d] / counts[c];

            } while (moved && ++iter < maxIter);

            Console.WriteLine($"\n=== K-MEANS RESULTS (after {iter} iterations) ===");
            for (int c = 0; c < K; c++)
            {
                Console.WriteLine($"\nCluster {c}:");
                Console.WriteLine($"  Centroid: ({string.Join(", ", centroids[c].Select(x => x.ToString("F16")))})");
                Console.WriteLine($"  Member points:");
                for (int i = 0; i < n; i++)
                {
                    if (labels[i] == c)
                    {
                        Console.WriteLine($"    Point {i}: ({string.Join(", ", data[i].Select(x => x.ToString("F16")))}), density={density[i]}");
                    }
                }
            }

            // 5) Refine centroids via density-weighted mean of high-density points
            double[][] refined = new double[K][];
            for (int c = 0; c < K; c++)
            {
                var pts = data
                    .Select((pt, idx) => new { Pt = pt, D = density[idx], L = labels[idx], Idx = idx })
                    .Where(x => x.L == c)
                    .ToArray();
                double avgD = pts.Average(x => x.D);
                var high = pts.Where(x => x.D >= avgD).ToArray();
                double wsum = high.Sum(x => x.D);
                refined[c] = new double[dim];

                Console.WriteLine($"\n=== REFINEMENT FOR CLUSTER {c} ===");
                Console.WriteLine($"  Average density: {avgD:F16}");
                Console.WriteLine($"  High-density points (density >= avg):");

                if (wsum > 0)
                {
                    for (int d = 0; d < dim; d++)
                        refined[c][d] = high.Sum(x => x.Pt[d] * x.D) / wsum;

                    foreach (var pt in high)
                    {
                        Console.WriteLine($"    Point {pt.Idx}: ({string.Join(", ", pt.Pt.Select(x => x.ToString("F16")))}), density={pt.D}, weight={pt.D / wsum:F16}");
                    }
                    Console.WriteLine($"  Density-weighted sum: {wsum:F16}");
                    Console.WriteLine($"  Refined centroid: ({string.Join(", ", refined[c].Select(x => x.ToString("F16")))})");
                }
                else
                {
                    refined[c] = centroids[c];
                    Console.WriteLine($"  No high-density points; using original centroid");
                }
            }

            // 6) Compute 2D relative centroid μ_rel
            double[] rel = new double[dim];
            for (int c = 0; c < K; c++)
                for (int d = 0; d < dim; d++)
                    rel[d] += refined[c][d];
            for (int d = 0; d < dim; d++)
                rel[d] /= K;

            Console.WriteLine($"\n=== RELATIVE CENTROID CALCULATION ===");
            Console.WriteLine($"Contributing refined centroids:");
            for (int c = 0; c < K; c++)
                Console.WriteLine($"  Refined {c}: ({string.Join(", ", refined[c].Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Sum: ({string.Join(", ", rel.Select(x => (x * K).ToString("F16")))})");
            Console.WriteLine($"Average (μ_rel): ({string.Join(", ", rel.Select(x => x.ToString("F16")))})");

            // 7) Lift into ℝⁿ⁺¹: base at z=0, apex z=‖μ_rel‖
            double relNorm = Math.Sqrt(rel.Sum(v => v * v));

            Console.WriteLine($"\n=== APEX Z-COORDINATE CALCULATION ===");
            Console.WriteLine($"μ_rel components: ({string.Join(", ", rel.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"μ_rel squared components: ({string.Join(", ", rel.Select(x => (x * x).ToString("F16")))})");
            Console.WriteLine($"Sum of squares: {rel.Sum(v => v * v):F16}");
            Console.WriteLine($"‖μ_rel‖ (apex z-coordinate): {relNorm:F16}");

            double[][] vertices = new double[K + 1][];
            for (int c = 0; c < K; c++)
                vertices[c] = refined[c].Concat(new[] { 0.0 }).ToArray();
            vertices[K] = rel.Concat(new[] { relNorm }).ToArray();

            Console.WriteLine($"\n=== FINAL 3D VERTICES (ℝⁿ⁺¹) ===");
            for (int c = 0; c < K; c++)
                Console.WriteLine($"Base vertex {c}: ({string.Join(", ", vertices[c].Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Apex vertex: ({string.Join(", ", vertices[K].Select(x => x.ToString("F16")))})");

            return vertices;
        }

        static void ComputeTensorAndVelocity(double[][] verts)
        {
            int m = verts[0].Length;      // dimension (n+1)
            int K = verts.Length - 1;     // number of base vertices
            double[] apex = verts[K];

            // 1) base-centroid b = (1/K) ∑ baseᵢ
            double[] b = new double[m];
            for (int i = 0; i < K; i++)
                for (int d = 0; d < m; d++)
                    b[d] += verts[i][d];
            for (int d = 0; d < m; d++)
                b[d] /= K;

            Console.WriteLine($"\n=== BASE CENTROID CALCULATION ===");
            Console.WriteLine($"Base vertices contributing:");
            for (int i = 0; i < K; i++)
                Console.WriteLine($"  V{i}: ({string.Join(", ", verts[i].Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Base centroid b: ({string.Join(", ", b.Select(x => x.ToString("F16")))})");

            // 2) direction vector d = apex − b
            double[] dir = new double[m];
            for (int d = 0; d < m; d++)
                dir[d] = apex[d] - b[d];

            Console.WriteLine($"\n=== DIRECTION VECTOR CALCULATION ===");
            Console.WriteLine($"Apex: ({string.Join(", ", apex.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Base centroid: ({string.Join(", ", b.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Direction d = apex - b: ({string.Join(", ", dir.Select(x => x.ToString("F16")))})");

            // 3) magnitude = ‖d‖
            double mag = Math.Sqrt(dir.Sum(x => x * x));

            // 4) unit direction u = d / mag
            double[] u = dir.Select(x => x / mag).ToArray();

            // 5) tensor T = u ⊗ u (outer product)
            double[,] T = new double[m, m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    T[i, j] = u[i] * u[j];

            // 6) velocity v = mag · u
            double[] v = u.Select(x => x * mag).ToArray();

            // ==== Inverses with NaN Resolution ====
            const double EPSILON = 1e-10;  // Small value to handle near-zero cases

            // Inverse of direction: component-wise reciprocal or special handling
            double[] invDir = new double[m];
            for (int i = 0; i < m; i++)
            {
                if (Math.Abs(dir[i]) < EPSILON)
                    invDir[i] = double.PositiveInfinity;  // Use infinity for zero components
                else
                    invDir[i] = 1.0 / dir[i];
            }

            // Inverse of magnitude: reciprocal
            double invMag = mag > EPSILON ? 1.0 / mag : double.PositiveInfinity;

            // Inverse unit direction: component-wise reciprocal
            double[] invU = new double[m];
            for (int i = 0; i < m; i++)
            {
                if (Math.Abs(u[i]) < EPSILON)
                    invU[i] = double.PositiveInfinity;
                else
                    invU[i] = 1.0 / u[i];
            }

            // Inverse velocity: component-wise reciprocal
            double[] invV = new double[m];
            for (int i = 0; i < m; i++)
            {
                if (Math.Abs(v[i]) < EPSILON)
                    invV[i] = double.PositiveInfinity;
                else
                    invV[i] = 1.0 / v[i];
            }

            // For rank-1 tensor, use Moore-Penrose pseudoinverse
            // For T = u⊗u, the pseudoinverse is T⁺ = (1/‖u‖²) * u⊗u when u≠0
            double[,] invT = null;
            double uNormSq = u.Sum(x => x * x);
            if (uNormSq > EPSILON)
            {
                // Pseudoinverse exists and equals T/‖u‖²
                invT = new double[m, m];
                double scale = 1.0 / uNormSq;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < m; j++)
                        invT[i, j] = scale * T[i, j];
            }

            // === OUTPUT ===
            Console.WriteLine("\n=== VERTICES (ℝⁿ⁺¹) ===");
            for (int i = 0; i < verts.Length; i++)
                Console.WriteLine($"V{i}: ({string.Join(", ", verts[i].Select(x => x.ToString("F3")))})");

            Console.WriteLine("\n=== DIRECTION & INVERSE ===");
            Console.WriteLine($"d   = ({string.Join(", ", dir.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"d⁻¹ = ({string.Join(", ", invDir.Select(x => FormatInverse(x)))})");

            Console.WriteLine("\n=== MAGNITUDE & INVERSE ===");
            Console.WriteLine($"|d|   = {mag:F3}");
            Console.WriteLine($"|d|⁻¹ = {invMag:F3}");

            Console.WriteLine("\n=== UNIT-DIR & INVERSE ===");
            Console.WriteLine($"u   = ({string.Join(", ", u.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"u⁻¹ = ({string.Join(", ", invU.Select(x => FormatInverse(x)))})");

            Console.WriteLine("\n=== VELOCITY & INVERSE ===");
            Console.WriteLine($"v   = ({string.Join(", ", v.Select(x => x.ToString("F3")))})");
            Console.WriteLine($"v⁻¹ = ({string.Join(", ", invV.Select(x => x.ToString("F3")))})");

            Console.WriteLine("\n=== TENSOR T & PSEUDOINVERSE T⁺ ===");
            Console.WriteLine("T:");
            for (int i = 0; i < m; i++)
            {
                Console.Write(" [ ");
                for (int j = 0; j < m; j++)
                    Console.Write($"{T[i, j]:F16} ");
                Console.WriteLine("]");
            }
            if (invT != null)
            {
                Console.WriteLine("T⁺ (Moore-Penrose pseudoinverse):");
                for (int i = 0; i < m; i++)
                {
                    Console.Write(" [ ");
                    for (int j = 0; j < m; j++)
                        Console.Write($"{invT[i, j]:F16} ");
                    Console.WriteLine("]");
                }
            }
            else
            {
                Console.WriteLine("T is zero matrix; no pseudoinverse exists.");
            }
        }

        // Helper to format inverse values (handles infinity nicely)
        static string FormatInverse(double val)
        {
            if (double.IsPositiveInfinity(val))
                return "∞";
            else if (double.IsNegativeInfinity(val))
                return "-∞";
            else
                return val.ToString("F16");
        }

        // Euclidean distance in any dimension
        static double Euclid(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            return Math.Sqrt(sum);
        }

        // General Gauss-Jordan inversion for any square matrix
        static double[,] InvertMatrix(double[,] A)
        {
            int n = A.GetLength(0);
            if (n != A.GetLength(1))
                return null; // not square

            // Create augmented matrix [A | I]
            double[,] aug = new double[n, 2 * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    aug[i, j] = A[i, j];
                aug[i, n + i] = 1.0;
            }

            // Forward elimination
            for (int i = 0; i < n; i++)
            {
                // Pivot
                double pivot = aug[i, i];
                if (Math.Abs(pivot) < 1e-12)
                    return null; // singular

                // Scale row i
                for (int j = 0; j < 2 * n; j++)
                    aug[i, j] /= pivot;

                // Eliminate other rows
                for (int k = 0; k < n; k++)
                {
                    if (k == i) continue;
                    double factor = aug[k, i];
                    for (int j = 0; j < 2 * n; j++)
                        aug[k, j] -= factor * aug[i, j];
                }
            }

            // Extract inverse from augmented
            double[,] inv = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    inv[i, j] = aug[i, n + j];

            return inv;
        }
    }
}