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

            // === DISPLAY COMPLETE MATHEMATICAL FORMULA ===
            PrintCompleteFormula();

            // === RUN THE CLUSTER PIPELINE ===
            // Formula: ND(x₁,…,xₙ) = densityNormalizedKMeans → tensorVelocity
            // Returns K base vertices + 1 apex vertex in ℝⁿ⁺¹
            var vertices = ComputeDensityNormalizedKMeansPipeline(
                rawData, K, densityRadius, maxIter);

            // === COMPUTE & PRINT TENSOR, VELOCITY, AND THEIR INVERSES ===
            ComputeTensorAndVelocity(vertices);
        }

        /// <summary>
        /// Contains the complete compiled mathematical formula for the N-Dimensional Cluster Execution pipeline.
        /// This can be copy-pasted for documentation, papers, or other implementations.
        /// </summary>
        static void PrintCompleteFormula()
        {
            string compiledFormula = GetCompiledFormula();

            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("N-DIMENSIONAL CLUSTER EXECUTION - COMPLETE MATHEMATICAL FORMULA");
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine();
            Console.WriteLine("ALGORITHM DESCRIPTION:");
            Console.WriteLine("This algorithm implements a density-normalized K-means clustering pipeline that:");
            Console.WriteLine("1. Normalizes input data to [0,1] hypercube");
            Console.WriteLine("2. Computes local density for intelligent cluster seeding");
            Console.WriteLine("3. Performs K-means clustering with density-based initialization");
            Console.WriteLine("4. Refines centroids using density-weighted averaging");
            Console.WriteLine("5. Lifts 2D clusters into 3D geometric structure (simplex)");
            Console.WriteLine("6. Computes directional tensors and velocity vectors with inverses");
            Console.WriteLine();
            Console.WriteLine("MATHEMATICAL FOUNDATION:");
            Console.WriteLine("The algorithm transforms raw data X ∈ ℝⁿˣᵈ into a geometric structure");
            Console.WriteLine("in ℝᵈ⁺¹ with associated tensor fields, enabling topological analysis");
            Console.WriteLine("of cluster relationships and directional flow computations.");
            Console.WriteLine();
            Console.WriteLine("COPY-PASTE FORMULA:");
            Console.WriteLine("─".PadRight(80, '─'));
            Console.WriteLine(compiledFormula);
            Console.WriteLine("─".PadRight(80, '─'));
            Console.WriteLine();

            // === N-DIMENSIONAL EXPRESSION GENERATION ===
            PrintNDimensionalExpressions();
        }

        /// <summary>
        /// Generates and displays N-dimensional expressions for different pipeline components
        /// </summary>
        static void PrintNDimensionalExpressions()
        {
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("N-DIMENSIONAL EXPRESSIONS - PIPELINE COMPONENTS");
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine();

            // Generate expressions for different pipeline keys
            string[] logicKeys = { "densityNormalizedKMeans", "tensorVelocity", "completePipeline" };

            foreach (string logicKey in logicKeys)
            {
                // 1) Convert the logic key into a regex "pattern"
                string regexPattern = ConvertLogicToRegex(logicKey);

                // 2) Convert that regex pattern into an n-dimensional expression
                string ndExpression = ConvertRegexToNDimensionalExpression(regexPattern);

                // === OUTPUT TO CONSOLE ===
                Console.WriteLine($"Logic Key:       {logicKey}");
                Console.WriteLine($"Regex Pattern:   {regexPattern}");
                Console.WriteLine($"ND Expression:{ndExpression}");
                Console.WriteLine();
            }

            Console.WriteLine("─".PadRight(80, '─'));
            Console.WriteLine();
        }

        /// <summary>
        /// Maps a high-level clustering logic identifier into a regex pattern
        /// that "captures" the key operations.
        /// </summary>
        static string ConvertLogicToRegex(string logic)
        {
            if (logic == "densityNormalizedKMeans")
            {
                // density -> kmeans -> refine
                return @"(density\([^\)]+\))\s*->\s*(kmeans\([^\)]+\))\s*->\s*(refine\([^\)]+\))";
            }
            if (logic == "tensorVelocity")
            {
                // tensor + velocity
                return @"(tensor\([^\)]+\))\s*\+\s*(velocity\([^\)]+\))";
            }
            if (logic == "completePipeline")
            {
                // complete pipeline with all phases
                return @"(normalize\([^\)]+\))\s*->\s*(density\([^\)]+\))\s*->\s*(kmeans\([^\)]+\))\s*->\s*(refine\([^\)]+\))\s*->\s*(lift\([^\)]+\))\s*->\s*(tensor\([^\)]+\))\s*\+\s*(velocity\([^\)]+\))";
            }
            // fallback for anything else
            return @"(\w+)";
        }

        /// <summary>
        /// Converts a regex‐style "capture" of our clustering pipeline
        /// into a single, compute-safe n-dimensional expression string.
        /// </summary>
        static string ConvertRegexToNDimensionalExpression(string regexPattern)
        {
            // complete pipeline with all phases
            if (System.Text.RegularExpressions.Regex.IsMatch(regexPattern, @"normalize\([^\)]+\)\s*->\s*density\([^\)]+\)\s*->\s*kmeans\([^\)]+\)\s*->\s*refine\([^\)]+\)\s*->\s*lift\([^\)]+\)\s*->\s*tensor\([^\)]+\)\s*\+\s*velocity"))
            {
                return GetCompiledFormula();
            }
            // density‐normalized K-means pipeline
            if (System.Text.RegularExpressions.Regex.IsMatch(regexPattern, @"density\([^\)]+\)\s*->\s*kmeans"))
            {
                return @"
ND(x₁,…,xₙ) =
  // 1) Density seeding: pick top‐density points
  μⱼ⁽⁰⁾ ← argmaxₓ ∑_{y:‖x−y‖≤r} 1
  // 2) K-means loop:
  μⱼ⁽ᵗ⁺¹⁾ ← (1/|Cⱼ|) ∑_{x∈Cⱼ} x
  // 3) Refinement: density‐weighted mean
  μⱼ′ ← (∑_{x∈Cⱼ, d(x)≥d̄ⱼ} d(x)·x) / ∑_{…} d(x)
  // 4) Relative centroid:
  μ_rel = (1/K) ∑ⱼ μⱼ′
  // 5) Apex lift to ℝⁿ⁺¹:
  apex = (μ_rel, ‖μ_rel‖)
  // 6) Unit‐normalize all K+1 vertices
";
            }
            // tensor + velocity pipeline
            if (System.Text.RegularExpressions.Regex.IsMatch(regexPattern, @"tensor\([^\)]+\)\s*\+\s*velocity"))
            {
                return @"
ND(x₁,…,xₙ) =
  // Base centroid:
  b = (1/K) ∑ baseᵢ
  // Direction & magnitude:
  d = apex − b, ‖d‖ = mag
  u = d / mag
  // Tensor:
  T = u ⊗ u
  // Velocity:
  v = mag · u
";
            }
            // generic fallback
            return @"
ND(x₁,…,xₙ) = x₁ + x₂ + … + xₙ
";
        }

        /// <summary>
        /// Returns the complete compiled mathematical formula as a string for easy copy-pasting.
        /// </summary>
        static string GetCompiledFormula()
        {
            return @"
NDimensionalClusterExecution(X, K, r, maxIter) = 
{
  // === PHASE 1: DATA PREPROCESSING ===
  // Input: X = {x₁, x₂, ..., xₙ} ∈ ℝⁿˣᵈ
  
  // 1.1) Min-Max Normalization to [0,1]ᵈ
  x'ᵢ = (xᵢ - min(X)) / (max(X) - min(X))
  
  // 1.2) Local Density Computation
  d(xᵢ) = ∑_{j≠i} 𝟙[‖xᵢ - xⱼ‖₂ ≤ r]
  
  // === PHASE 2: DENSITY-NORMALIZED K-MEANS ===
  
  // 2.1) Density-Based Seeding
  Seeds = argmax_{K} {d(xᵢ) : i ∈ {1,...,n}}
  μⱼ⁽⁰⁾ = x_{Seeds[j]} for j ∈ {1,...,K}
  
  // 2.2) K-Means Iteration Loop
  repeat until convergence or maxIter:
    // Assignment: Cⱼ = {xᵢ : ‖xᵢ - μⱼ‖₂ ≤ ‖xᵢ - μₖ‖₂ ∀k}
    // Update: μⱼ⁽ᵗ⁺¹⁾ = (1/|Cⱼ|) ∑_{x∈Cⱼ} x
  
  // 2.3) Density-Weighted Refinement
  d̄ⱼ = (1/|Cⱼ|) ∑_{x∈Cⱼ} d(x)                    // average density
  Hⱼ = {x ∈ Cⱼ : d(x) ≥ d̄ⱼ}                       // high-density subset
  μⱼ' = (∑_{x∈Hⱼ} d(x)·x) / (∑_{x∈Hⱼ} d(x))       // density-weighted centroid
  
  // === PHASE 3: GEOMETRIC LIFTING TO ℝᵈ⁺¹ ===
  
  // 3.1) Relative Centroid
  μᵣₑₗ = (1/K) ∑ⱼ μⱼ'
  
  // 3.2) Simplex Construction
  baseⱼ = (μⱼ', 0) ∈ ℝᵈ⁺¹                         // base vertices at z=0
  apex = (μᵣₑₗ, ‖μᵣₑₗ‖₂) ∈ ℝᵈ⁺¹                   // apex at z=‖μᵣₑₗ‖
  
  // === PHASE 4: TENSOR & VELOCITY COMPUTATION ===
  
  // 4.1) Base Centroid & Direction
  b = (1/K) ∑ⱼ baseⱼ                              // geometric center of bases
  d = apex - b                                    // direction vector
  
  // 4.2) Magnitude & Unit Direction
  mag = ‖d‖₂                                     // magnitude
  u = d / mag                                     // unit direction vector
  
  // 4.3) Tensor Field (Rank-1)
  T = u ⊗ u ∈ ℝ⁽ᵈ⁺¹⁾ˣ⁽ᵈ⁺¹⁾                        // outer product tensor
  
  // 4.4) Velocity Field
  v = mag · u                                     // velocity vector
  
  // === PHASE 5: INVERSE COMPUTATIONS ===
  
  // 5.1) Component-wise Inverses (with ∞ for zeros)
  d⁻¹ᵢ = {1/dᵢ  if |dᵢ| > ε, ∞ otherwise}
  u⁻¹ᵢ = {1/uᵢ  if |uᵢ| > ε, ∞ otherwise}
  v⁻¹ᵢ = {1/vᵢ  if |vᵢ| > ε, ∞ otherwise}
  mag⁻¹ = 1/mag
  
  // 5.2) Tensor Pseudoinverse (Moore-Penrose)
  T⁺ = (1/‖u‖₂²) · T  when u ≠ 0
  
  // === OUTPUT ===
  return {
    vertices: {base₁, base₂, ..., baseₖ, apex},
    direction: d, magnitude: mag, unit: u,
    tensor: T, tensor_inverse: T⁺,
    velocity: v, inverses: {d⁻¹, u⁻¹, v⁻¹, mag⁻¹}
  }
}

// COMPLEXITY: O(n² + K·n·maxIter) time, O(n·d + K·d) space
// APPLICATIONS: Topological data analysis, cluster flow dynamics, 
//               geometric machine learning, n-dimensional pattern recognition
";
        }

        /// <summary>
        /// Implements the density-normalized K-means pipeline:
        /// Formula: μⱼ⁽⁰⁾ ← argmaxₓ ∑_{y:‖x−y‖≤r} 1 → μⱼ⁽ᵗ⁺¹⁾ ← (1/|Cⱼ|) ∑_{x∈Cⱼ} x → 
        ///          μⱼ′ ← (∑_{x∈Cⱼ, d(x)≥d̄ⱼ} d(x)·x) / ∑_{…} d(x) → apex = (μ_rel, ‖μ_rel‖)
        /// </summary>
        static double[][] ComputeDensityNormalizedKMeansPipeline(
            double[][] raw, int K, double radius, int maxIter)
        {
            int n = raw.Length;
            int dim = raw[0].Length;

            Console.WriteLine("=== RAW DATA ===");
            for (int i = 0; i < n; i++)
                Console.WriteLine($"Point {i}: ({string.Join(", ", raw[i].Select(x => x.ToString("F16")))})");

            // === STEP 1: MIN-MAX NORMALIZATION ===
            // Formula: x'ᵢ = (xᵢ - min(x)) / (max(x) - min(x))
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

            // === STEP 2: LOCAL DENSITY COMPUTATION ===
            // Formula: d(xᵢ) = ∑_{j≠i} 𝟙[‖xᵢ - xⱼ‖ ≤ r]
            int[] density = new int[n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (i != j && Euclid(data[i], data[j]) <= radius)
                        density[i]++;

            Console.WriteLine("\n=== DENSITY VALUES ===");
            for (int i = 0; i < n; i++)
                Console.WriteLine($"Point {i} density: {density[i]}");

            // === STEP 3: DENSITY-BASED SEEDING ===
            // Formula: μⱼ⁽⁰⁾ ← argmaxₓ ∑_{y:‖x−y‖≤r} 1 (top K density points)
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

            // === STEP 4: K-MEANS CLUSTERING ===
            // Formula: μⱼ⁽ᵗ⁺¹⁾ ← (1/|Cⱼ|) ∑_{x∈Cⱼ} x
            do
            {
                moved = false;

                // Assignment step: Cⱼ = {x : ‖x - μⱼ‖ ≤ ‖x - μₖ‖ ∀k}
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

                // Update step: μⱼ⁽ᵗ⁺¹⁾ ← (1/|Cⱼ|) ∑_{x∈Cⱼ} x
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

            // === STEP 5: DENSITY-WEIGHTED REFINEMENT ===
            // Formula: μⱼ′ ← (∑_{x∈Cⱼ, d(x)≥d̄ⱼ} d(x)·x) / (∑_{x∈Cⱼ, d(x)≥d̄ⱼ} d(x))
            double[][] refined = new double[K][];
            for (int c = 0; c < K; c++)
            {
                var pts = data
                    .Select((pt, idx) => new { Pt = pt, D = density[idx], L = labels[idx], Idx = idx })
                    .Where(x => x.L == c)
                    .ToArray();
                double avgD = pts.Average(x => x.D);  // d̄ⱼ = average density in cluster j
                var high = pts.Where(x => x.D >= avgD).ToArray();  // high-density subset
                double wsum = high.Sum(x => x.D);  // ∑ d(x) for normalization
                refined[c] = new double[dim];

                Console.WriteLine($"\n=== REFINEMENT FOR CLUSTER {c} ===");
                Console.WriteLine($"  Average density: {avgD:F16}");
                Console.WriteLine($"  High-density points (density >= avg):");

                if (wsum > 0)
                {
                    // Density-weighted centroid: μⱼ′ = (∑ d(x)·x) / (∑ d(x))
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

            // === STEP 6: RELATIVE CENTROID COMPUTATION ===
            // Formula: μ_rel = (1/K) ∑ⱼ μⱼ′
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

            // === STEP 7: GEOMETRIC LIFT TO ℝⁿ⁺¹ ===
            // Formula: apex = (μ_rel, ‖μ_rel‖), baseᵢ = (μᵢ′, 0)
            double relNorm = Math.Sqrt(rel.Sum(v => v * v));  // ‖μ_rel‖

            Console.WriteLine($"\n=== APEX Z-COORDINATE CALCULATION ===");
            Console.WriteLine($"μ_rel components: ({string.Join(", ", rel.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"μ_rel squared components: ({string.Join(", ", rel.Select(x => (x * x).ToString("F16")))})");
            Console.WriteLine($"Sum of squares: {rel.Sum(v => v * v):F16}");
            Console.WriteLine($"‖μ_rel‖ (apex z-coordinate): {relNorm:F16}");

            double[][] vertices = new double[K + 1][];
            for (int c = 0; c < K; c++)
                vertices[c] = refined[c].Concat(new[] { 0.0 }).ToArray();  // baseᵢ = (μᵢ′, 0)
            vertices[K] = rel.Concat(new[] { relNorm }).ToArray();  // apex = (μ_rel, ‖μ_rel‖)

            Console.WriteLine($"\n=== FINAL 3D VERTICES (ℝⁿ⁺¹) ===");
            for (int c = 0; c < K; c++)
                Console.WriteLine($"Base vertex {c}: ({string.Join(", ", vertices[c].Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Apex vertex: ({string.Join(", ", vertices[K].Select(x => x.ToString("F16")))})");

            return vertices;
        }

        /// <summary>
        /// Computes tensor and velocity from the geometric structure:
        /// Formula: b = (1/K) ∑ baseᵢ, d = apex − b, u = d/‖d‖, T = u ⊗ u, v = ‖d‖ · u
        /// </summary>
        static void ComputeTensorAndVelocity(double[][] verts)
        {
            int m = verts[0].Length;      // dimension (n+1)
            int K = verts.Length - 1;     // number of base vertices
            double[] apex = verts[K];

            // === STEP 1: BASE CENTROID ===
            // Formula: b = (1/K) ∑ᵢ baseᵢ
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

            // === STEP 2: DIRECTION VECTOR ===
            // Formula: d = apex − b
            double[] dir = new double[m];
            for (int d = 0; d < m; d++)
                dir[d] = apex[d] - b[d];

            Console.WriteLine($"\n=== DIRECTION VECTOR CALCULATION ===");
            Console.WriteLine($"Apex: ({string.Join(", ", apex.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Base centroid: ({string.Join(", ", b.Select(x => x.ToString("F16")))})");
            Console.WriteLine($"Direction d = apex - b: ({string.Join(", ", dir.Select(x => x.ToString("F16")))})");

            // === STEP 3: MAGNITUDE & UNIT DIRECTION ===
            // Formula: ‖d‖ = magnitude, u = d / ‖d‖
            double mag = Math.Sqrt(dir.Sum(x => x * x));  // ‖d‖
            double[] u = dir.Select(x => x / mag).ToArray();  // u = d / ‖d‖

            // === STEP 4: TENSOR COMPUTATION ===
            // Formula: T = u ⊗ u (outer product)
            double[,] T = new double[m, m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    T[i, j] = u[i] * u[j];

            // === STEP 5: VELOCITY COMPUTATION ===
            // Formula: v = ‖d‖ · u
            double[] v = u.Select(x => x * mag).ToArray();

            // === INVERSE COMPUTATIONS WITH NaN RESOLUTION ===
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

            // === TENSOR PSEUDOINVERSE ===
            // Formula: For T = u⊗u, T⁺ = (1/‖u‖²) * u⊗u when u≠0 (Moore-Penrose)
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

        /// <summary>
        /// Euclidean distance in any dimension
        /// Formula: ‖a - b‖ = √(∑ᵢ (aᵢ - bᵢ)²)
        /// </summary>
        static double Euclid(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            return Math.Sqrt(sum);
        }

        /// <summary>
        /// General Gauss-Jordan inversion for any square matrix
        /// Formula: A⁻¹ such that A·A⁻¹ = I
        /// </summary>
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