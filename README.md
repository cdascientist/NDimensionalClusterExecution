# NDimensionalClusterExecution

# N-Dimensional Matrix Vertex Implementation
## NDimensionalClusterExecution as Vertex-Based Computation

### Overview
The N-Dimensional Cluster Execution algorithm can be conceptualized as a distributed computation occurring across the vertices of an n-dimensional matrix, where each vertex acts as a computational node capable of storing data, performing local calculations, and communicating with neighboring vertices.

---

## Phase 1: Matrix Initialization and Vertex Normalization

### **Vertex Grid Setup**
```
Matrix M ∈ ℝⁿˣⁿˣ...ˣⁿ (d-dimensional hypercube)
Each vertex V(i₁,i₂,...,iₐ) contains:
  - position: (x₁, x₂, ..., xₐ)
  - data_value: raw input value
  - normalized_value: [0,1] scaled value
  - neighbors: set of adjacent vertices
  - density_count: local density metric
  - cluster_id: assigned cluster identifier
  - centroid_flag: boolean marking cluster centers
```

### **Normalization Process**
Each vertex V performs local normalization:
```
V.normalized_value = (V.data_value - global_min) / (global_max - global_min)
```
- **Global min/max** propagated through matrix via broadcast
- Each vertex updates its position to normalized coordinates
- Vertices maintain both original and normalized coordinate systems

---

## Phase 2: Vertex-Based Density Computation

### **Local Density Calculation**
Each vertex V computes its density by querying neighbor vertices:
```python
def compute_vertex_density(vertex_V, radius_r):
    density_count = 0
    for vertex_U in matrix.all_vertices():
        if vertex_U != vertex_V:
            distance = euclidean_distance(V.position, U.position)
            if distance <= radius_r:
                density_count += 1
    vertex_V.density_count = density_count
    return density_count
```

### **Neighborhood Communication**
- Each vertex broadcasts its position to all other vertices
- Vertices within radius `r` respond with acknowledgment
- Dense regions emerge as vertices with high neighbor counts
- **Matrix topology**: vertices form communication graphs based on distance

---

## Phase 3: Distributed K-Means with Vertex Centroids

### **Centroid Vertex Selection**
```python
# Select K vertices with highest density as initial centroids
centroid_vertices = select_top_k_density_vertices(K)
for cv in centroid_vertices:
    cv.centroid_flag = True
    cv.cluster_id = assign_unique_id()
```

### **Vertex Assignment Phase**
Each non-centroid vertex finds its nearest centroid:
```python
def assign_vertex_to_cluster(vertex_V):
    min_distance = infinity
    assigned_cluster = None
    
    for centroid_vertex in matrix.get_centroid_vertices():
        distance = euclidean_distance(V.position, centroid_vertex.position)
        if distance < min_distance:
            min_distance = distance
            assigned_cluster = centroid_vertex.cluster_id
    
    vertex_V.cluster_id = assigned_cluster
```

### **Centroid Update Phase**
Centroid vertices recompute their positions:
```python
def update_centroid_position(centroid_vertex):
    cluster_members = matrix.get_vertices_in_cluster(centroid_vertex.cluster_id)
    
    # Compute mean position of all member vertices
    sum_position = [0] * dimensions
    for member_vertex in cluster_members:
        for d in range(dimensions):
            sum_position[d] += member_vertex.position[d]
    
    # Update centroid vertex position
    for d in range(dimensions):
        centroid_vertex.position[d] = sum_position[d] / len(cluster_members)
```

---

## Phase 4: Density-Weighted Vertex Refinement

### **High-Density Vertex Filtering**
Each centroid vertex refines its position using only high-density members:
```python
def refine_centroid_with_density(centroid_vertex):
    cluster_members = matrix.get_vertices_in_cluster(centroid_vertex.cluster_id)
    average_density = mean([v.density_count for v in cluster_members])
    
    # Filter for high-density vertices
    high_density_vertices = [v for v in cluster_members 
                           if v.density_count >= average_density]
    
    # Density-weighted position update
    weighted_sum = [0] * dimensions
    total_weight = sum([v.density_count for v in high_density_vertices])
    
    for vertex in high_density_vertices:
        weight = vertex.density_count / total_weight
        for d in range(dimensions):
            weighted_sum[d] += vertex.position[d] * weight
    
    # Update centroid vertex to refined position
    centroid_vertex.position = weighted_sum
```

---

## Phase 5: Matrix Dimensional Lifting

### **Vertex Elevation to (n+1)D Matrix**
The entire vertex matrix is lifted into higher-dimensional space:
```python
def lift_matrix_to_higher_dimension():
    # Compute relative centroid from all centroid vertices
    centroid_vertices = matrix.get_centroid_vertices()
    relative_centroid = compute_mean_position(centroid_vertices)
    apex_height = euclidean_norm(relative_centroid)
    
    # Create new (n+1)D matrix
    lifted_matrix = Matrix(dimensions + 1)
    
    # Lift all vertices
    for vertex in matrix.all_vertices():
        new_vertex = Vertex(dimensions + 1)
        new_vertex.position = vertex.position + [0.0]  # Add z=0
        
        if vertex.centroid_flag:
            new_vertex.vertex_type = "base_vertex"
        
        lifted_matrix.add_vertex(new_vertex)
    
    # Create apex vertex
    apex_vertex = Vertex(dimensions + 1)
    apex_vertex.position = relative_centroid + [apex_height]
    apex_vertex.vertex_type = "apex_vertex"
    lifted_matrix.add_vertex(apex_vertex)
    
    return lifted_matrix
```

---

## Phase 6: Tensor Field Computation Across Vertices

### **Inter-Vertex Directional Computation**
```python
def compute_tensor_field_at_vertices(lifted_matrix):
    base_vertices = lifted_matrix.get_base_vertices()
    apex_vertex = lifted_matrix.get_apex_vertex()
    
    # Compute base centroid vertex
    base_centroid = compute_mean_position(base_vertices)
    
    # Direction vector from base centroid to apex
    direction_vector = apex_vertex.position - base_centroid.position
    magnitude = euclidean_norm(direction_vector)
    unit_direction = direction_vector / magnitude
    
    # Compute tensor field at each vertex
    for vertex in lifted_matrix.all_vertices():
        # Rank-1 tensor: T = u ⊗ u
        vertex.tensor_field = outer_product(unit_direction, unit_direction)
        vertex.velocity_field = magnitude * unit_direction
        vertex.direction_vector = direction_vector
        vertex.magnitude = magnitude
```

### **Vertex-to-Vertex Tensor Propagation**
```python
def propagate_tensor_fields():
    for vertex in lifted_matrix.all_vertices():
        # Each vertex computes tensor relationships with neighbors
        for neighbor_vertex in vertex.get_neighbors():
            # Compute relative tensor between vertices
            relative_tensor = vertex.tensor_field - neighbor_vertex.tensor_field
            vertex.neighbor_tensors[neighbor_vertex.id] = relative_tensor
```

---

## Phase 7: Inverse Field Computation

### **Per-Vertex Inverse Calculations**
```python
def compute_inverse_fields_at_vertices():
    for vertex in lifted_matrix.all_vertices():
        # Component-wise inverse of direction vector
        vertex.inverse_direction = [
            1.0/d if abs(d) > EPSILON else float('inf') 
            for d in vertex.direction_vector
        ]
        
        # Inverse magnitude
        vertex.inverse_magnitude = 1.0 / vertex.magnitude if vertex.magnitude > EPSILON else float('inf')
        
        # Moore-Penrose pseudoinverse of tensor
        if vertex.tensor_field.is_invertible():
            vertex.inverse_tensor = vertex.tensor_field.pseudoinverse()
        else:
            vertex.inverse_tensor = None
```

---

## Implementation Architecture

### **Matrix Vertex Class Structure**
```python
class NDimensionalMatrixVertex:
    def __init__(self, position, dimensions):
        self.position = position  # n-dimensional coordinates
        self.dimensions = dimensions
        self.data_value = None
        self.normalized_value = None
        self.density_count = 0
        self.cluster_id = None
        self.centroid_flag = False
        self.vertex_type = "data_vertex"  # "base_vertex", "apex_vertex"
        
        # Tensor and velocity fields
        self.tensor_field = None
        self.velocity_field = None
        self.direction_vector = None
        self.magnitude = None
        
        # Inverse fields
        self.inverse_direction = None
        self.inverse_magnitude = None
        self.inverse_tensor = None
        
        # Neighbor relationships
        self.neighbors = set()
        self.neighbor_tensors = {}
    
    def compute_local_density(self, radius):
        # Implementation as shown above
        pass
    
    def assign_to_cluster(self, centroids):
        # Implementation as shown above
        pass
    
    def update_as_centroid(self):
        # Implementation as shown above
        pass
```

### **Matrix Computation Parallelization**
- **Phase Parallelism**: Each phase can be parallelized across all vertices
- **Vertex Independence**: Most computations are local to individual vertices
- **Communication Patterns**: 
  - Broadcast for global statistics (min/max, centroid positions)
  - Local neighborhood communication for density computation
  - Cluster-based communication for centroid updates

### **Memory Architecture**
```
Matrix Memory Layout:
├── Vertex Storage: O(n^d) vertices in d-dimensional matrix
├── Position Data: Each vertex stores d-dimensional coordinates
├── Field Data: Tensor, velocity, and inverse fields per vertex
├── Communication Buffers: For inter-vertex message passing
└── Computation Scratch Space: For local calculations
```

### **Scalability Properties**
- **Space Complexity**: O(n^d × field_size) for d-dimensional matrix
- **Time Complexity**: O(n^d × K × iterations) for distributed K-means
- **Communication Complexity**: O(n^d) for density computation, O(K) for centroid updates
- **Parallelization**: Nearly embarrassingly parallel with minimal synchronization points

This vertex-based implementation transforms the algorithm into a distributed computation where each vertex in the n-dimensional matrix acts as an autonomous computational agent, making the entire clustering and tensor analysis process scalable across massively parallel architectures.


Generated markdown
# N-Dimensional Cluster Execution - Complete Mathematical Formula

## Algorithm Description

This algorithm implements a density-normalized K-means clustering pipeline that:

1.  Normalizes input data to a `[0,1]` hypercube.
2.  Computes local density for intelligent cluster seeding.
3.  Performs K-means clustering with density-based initialization.
4.  Refines centroids using density-weighted averaging.
5.  Lifts D-dimensional clusters (from original NxD data, usually visualized in 2D or 3D projections of centroids) into a (D+1)-dimensional geometric structure (simplex).
6.  Computes directional tensors and velocity vectors with inverses.

## Mathematical Foundation

The algorithm transforms raw data X ∈ ℝ<sup>N×D</sup> (N samples, D dimensions) into a geometric structure in ℝ<sup>D+1</sup> with associated tensor fields, enabling topological analysis of cluster relationships and directional flow computations.

---

## Algorithm Pseudocode

```math
\text{NDimensionalClusterExecution}(X, K, r, \text{maxIter}) =
\{
  // === PHASE 1: DATA PREPROCESSING ===
  // Input: X = \{x_1, x_2, ..., x_N\} \subset \mathbb{R}^D

  // 1.1) Min-Max Normalization to [0,1]^D
  x'_i = (x_i - \min_{\text{col}}(X)) / (\max_{\text{col}}(X) - \min_{\text{col}}(X)) \quad \text{// component-wise}

  // 1.2) Local Density Computation
  // \mathbb{I} is the indicator function
  d(x'_i) = \sum_{j \neq i} \mathbb{I}[\|x'_i - x'_j\|_2 \le r]

  // === PHASE 2: DENSITY-NORMALIZED K-MEANS ===
  // Using normalized data x'

  // 2.1) Density-Based Seeding
  // Select K points with highest density as initial seeds
  \text{Seeds} = \text{argmax}_{K} \{d(x'_i) : i \in \{1,...,N\}\}
  \mu_j^{(0)} = x'_{\text{Seeds}[j]} \quad \text{for } j \in \{1,...,K\}

  // 2.2) K-Means Iteration Loop
  \text{repeat until convergence or maxIter}:
    // Assignment: Assign each x'_i to the cluster C_j with the nearest centroid \mu_j
    C_j^{(t)} = \{x'_i : \|x'_i - \mu_j^{(t)}\|_2^2 \le \|x'_i - \mu_k^{(t)}\|_2^2 \quad \forall k \neq j\}
    // Update: Recalculate centroids
    \mu_j^{(t+1)} = \frac{1}{|C_j^{(t)}|} \sum_{x' \in C_j^{(t)}} x'

  // Let \mu_j be the converged centroids from K-Means

  // 2.3) Density-Weighted Refinement
  \bar{d}_j = \frac{1}{|C_j|} \sum_{x' \in C_j} d(x')                    // average density of cluster j
  H_j = \{x' \in C_j : d(x') \ge \bar{d}_j\}                       // high-density subset of cluster j
  \mu'_j = \frac{\sum_{x' \in H_j} d(x') \cdot x'}{\sum_{x' \in H_j} d(x')}       // density-weighted centroid

  // === PHASE 3: GEOMETRIC LIFTING TO ℝ^(D+1) ===

  // 3.1) Relative Centroid (Centroid of refined centroids)
  \mu_{\text{rel}} = \frac{1}{K} \sum_{j=1}^K \mu'_j

  // 3.2) Simplex Construction
  // base_j are vertices in the z=0 plane of ℝ^(D+1)
  \text{base}_j = (\mu'_j, 0) \in \mathbb{R}^{D+1}
  // apex is lifted along the (D+1)-th dimension by its magnitude
  \text{apex} = (\mu_{\text{rel}}, \|\mu_{\text{rel}}\|_2) \in \mathbb{R}^{D+1}

  // === PHASE 4: TENSOR & VELOCITY COMPUTATION ===

  // 4.1) Base Centroid & Direction
  b = \frac{1}{K} \sum_{j=1}^K \text{base}_j                              // geometric center of bases in ℝ^(D+1)
  \vec{d} = \text{apex} - b                                    // direction vector from base center to apex

  // 4.2) Magnitude & Unit Direction
  \text{mag} = \|\vec{d}\|_2                                     // magnitude of direction vector
  \vec{u} = \vec{d} / \text{mag}                                     // unit direction vector (if mag != 0)

  // 4.3) Tensor Field (Rank-1 Outer Product)
  T = \vec{u} \otimes \vec{u}^T \in \mathbb{R}^{(D+1) \times (D+1)}      // outer product tensor

  // 4.4) Velocity Field
  \vec{v} = \text{mag} \cdot \vec{u}                                     // velocity vector (same as \vec{d})

  // === PHASE 5: INVERSE COMPUTATIONS ===

  // 5.1) Component-wise Inverses (with ∞ for zeros, ε is a small tolerance)
  \vec{d}^{\,-1}_k = \{1/d_k  \text{ if } |d_k| > \epsilon, \infty \text{ otherwise}\} // for each component k of \vec{d}
  \vec{u}^{\,-1}_k = \{1/u_k  \text{ if } |u_k| > \epsilon, \infty \text{ otherwise}\} // for each component k of \vec{u}
  \vec{v}^{\,-1}_k = \{1/v_k  \text{ if } |v_k| > \epsilon, \infty \text{ otherwise}\} // for each component k of \vec{v}
  \text{mag}^{-1} = 1/\text{mag} \text{ if mag } > \epsilon, \infty \text{ otherwise}

  // 5.2) Tensor Pseudoinverse (Moore-Penrose)
  // For a rank-1 tensor T = u u^T, if u is a non-zero column vector, T^+ = (1/(u^T u)^2) T or (1/\|u\|_2^4) u u^T
  // Simpler: if T = u u^T and \|u\|_2 = 1, then T^+ = T.
  // Here, u is already a unit vector, so \|u\|_2 = 1.
  // T^\dagger = (1/(\|\vec{u}\|_2^2)^2) \cdot T = (1/1^2) \cdot T = T \quad \text{when } \vec{u} \neq \vec{0}
  // More generally, T^\dagger = \frac{1}{\text{trace}(T)^2} T \text{ if T is rank 1 and trace(T) != 0}
  // Since T = u u^T and u is unit vector, trace(T) = u^T u = \|u\|_2^2 = 1.
  T^\dagger = T \quad \text{if } \vec{u} \neq \vec{0} \text{ (since } \|\vec{u}\|_2=1 \text{)}
  // Or, if using the given formula: T_dagger = (1/\|u\|_2^2) * T
  // If \|u\|_2 = 1 (as it should be for a unit vector), then T_dagger = T.
  // If u is the zero vector, T is the zero matrix, and its pseudoinverse is also the zero matrix.
  T^\dagger = (1/\|\vec{u}\|_2^2) \cdot T \quad \text{if } \vec{u} \neq \vec{0}, \text{ else } 0 \text{ matrix}


  // === OUTPUT ===
  \text{return} \{
    \text{vertices: } \{\text{base}_1, \text{base}_2, ..., \text{base}_K, \text{apex}\},
    \text{direction: } \vec{d}, \text{magnitude: } \text{mag}, \text{unit_direction: } \vec{u},
    \text{tensor: } T, \text{tensor_inverse: } T^\dagger,
    \text{velocity: } \vec{v}, \text{inverses: } \{\vec{d}^{\,-1}, \vec{u}^{\,-1}, \vec{v}^{\,-1}, \text{mag}^{-1}\}
  \}
\}

// COMPLEXITY: O(N^2 * D + K*N*D*maxIter) time for dense local density, O(N*D + K*D) space
// (N^2*D for density, K*N*D*maxIter for K-Means)
// APPLICATIONS: Topological data analysis, cluster flow dynamics,
//               geometric machine learning, n-dimensional pattern recognition
Use code with caution.
Markdown
N-Dimensional Expressions - Pipeline Components
Logic Key: densityNormalizedKMeans
Regex Pattern:
Generated regex
(density\([^\)]+\))\s*->\s*(kmeans\([^\)]+\))\s*->\s*(refine\([^\)]+\))
Use code with caution.
Regex
ND Expression:
Generated code
ND(x1, ..., xN) = x1 + x2 + ... + xN
Use code with caution.
(Note: This ND Expression seems generic and might need context. Assuming it's a placeholder for a sum operation.)
Logic Key: tensorVelocity
Regex Pattern:
Generated regex
(tensor\([^\)]+\))\s*\+\s*(velocity\([^\)]+\))
Use code with caution.
Regex
ND Expression:
Generated code
ND(x1, ..., xN) = x1 + x2 + ... + xN
Use code with caution.
(Note: This ND Expression seems generic and might need context.)
Logic Key: completePipeline
Regex Pattern:
Generated regex
(normalize\([^\)]+\))\s*->\s*(density\([^\)]+\))\s*->\s*(kmeans\([^\)]+\))\s*->\s*(refine\([^\)]+\))\s*->\s*(lift\([^\)]+\))\s*->\s*(tensor\([^\)]+\))\s*\+\s*(velocity\([^\)]+\))
Use code with caution.
Regex
ND Expression:
Generated code
ND(x1, ..., xN) = x1 + x2 + ... + xN
Use code with caution.
(Note: This ND Expression seems generic and might need context.)
Example Execution Trace
Raw Data
Generated code
Point 0: (1.0000000000000000, 2.0000000000000000)
Point 1: (1.5000000000000000, 1.8000000000000000)
Point 2: (1.2000000000000000, 2.2000000000000002)
Point 3: (5.0000000000000000, 8.0000000000000000)
Point 4: (6.0000000000000000, 8.5000000000000000)
Point 5: (5.5000000000000000, 7.5000000000000000)
Point 6: (2.0000000000000000, 2.0000000000000000)
Point 7: (2.2000000000000002, 2.1000000000000001)
Point 8: (1.8000000000000000, 1.8999999999999999)
Use code with caution.
Normalization Parameters
Generated code
Min values: (1.0000000000000000, 1.8000000000000000)
Max values: (6.0000000000000000, 8.5000000000000000)
Use code with caution.
Normalized Data
Generated code
Norm Point 0: (0.0000000000000000, 0.0298507462686567)
Norm Point 1: (0.1000000000000000, 0.0000000000000000)
Norm Point 2: (0.0400000000000000, 0.0597014925373135)
Norm Point 3: (0.8000000000000000, 0.9253731343283582)
Norm Point 4: (1.0000000000000000, 1.0000000000000000)
Norm Point 5: (0.9000000000000000, 0.8507462686567164)
Norm Point 6: (0.2000000000000000, 0.0298507462686567)
Norm Point 7: (0.2400000000000000, 0.0447761194029851)
Norm Point 8: (0.1600000000000000, 0.0149253731343283)
Use code with caution.
Density Values (assuming r was chosen appropriately)
Generated code
Point 0 density: 4
Point 1 density: 5
Point 2 density: 4
Point 3 density: 1
Point 4 density: 1
Point 5 density: 2
Point 6 density: 5
Point 7 density: 3
Point 8 density: 5
Use code with caution.
Density-Based Seeds (K=3)
Generated code
Top 3 density points selected as seeds: 1, 6, 8
(Using their normalized coordinates)
Use code with caution.
K-Means Results (after 2 iterations)
Cluster 0:
Centroid: (0.0466666666666667, 0.0298507462686567)
Member points:
Point 0: (0.0000000000000000, 0.0298507462686567), density=4
Point 1: (0.1000000000000000, 0.0000000000000000), density=5
Point 2: (0.0400000000000000, 0.0597014925373135), density=4
Cluster 1:
Centroid: (0.9000000000000000, 0.9253731343283582)
Member points:
Point 3: (0.8000000000000000, 0.9253731343283582), density=1
Point 4: (1.0000000000000000, 1.0000000000000000), density=1
Point 5: (0.9000000000000000, 0.8507462686567164), density=2
Cluster 2:
Centroid: (0.2000000000000000, 0.0298507462686567)
Member points:
Point 6: (0.2000000000000000, 0.0298507462686567), density=5
Point 7: (0.2400000000000000, 0.0447761194029851), density=3
Point 8: (0.1600000000000000, 0.0149253731343283), density=5
Refinement for Cluster 0
Generated code
Average density: 4.3333333333333330
High-density points (density >= avg):
  Point 1: (0.1000000000000000, 0.0000000000000000), density=5
Weighted sum of densities in H_0: 5.0
Refined centroid µ'_0: (0.1000000000000000, 0.0000000000000000)
Use code with caution.
Refinement for Cluster 1
Generated code
Average density: 1.3333333333333333
High-density points (density >= avg):
  Point 5: (0.9000000000000000, 0.8507462686567164), density=2
Weighted sum of densities in H_1: 2.0
Refined centroid µ'_1: (0.9000000000000000, 0.8507462686567164)
Use code with caution.
Refinement for Cluster 2
Generated code
Average density: 4.3333333333333330
High-density points (density >= avg):
  Point 6: (0.2000000000000000, 0.0298507462686567), density=5
  Point 8: (0.1600000000000000, 0.0149253731343283), density=5
Weighted sum of densities in H_2: 10.0
Refined centroid µ'_2: (0.1800000000000000, 0.0223880597014925)
Use code with caution.
Relative Centroid Calculation (µ<sub>rel</sub>)
Contributing refined centroids:
µ'_0: (0.1000000000000000, 0.0000000000000000)
µ'_1: (0.9000000000000000, 0.8507462686567164)
µ'_2: (0.1800000000000000, 0.0223880597014925)
Sum: (1.1799999999999999, 0.8731343283582089)
Average (µ<sub>rel</sub>): (0.3933333333333333, 0.2910447761194030)
Apex Z-Coordinate Calculation (||µ<sub>rel</sub>||<sub>2</sub>)
µ<sub>rel</sub> components: (0.3933333333333333, 0.2910447761194030)
µ<sub>rel</sub> squared components: (0.1547111111111111, 0.0847070617063934)
Sum of squares: 0.2394181728175045
||µ<sub>rel</sub>||<sub>2</sub> (apex z-coordinate): 0.4893037633387919
Final 3D Vertices (ℝ<sup>D+1</sup>, here D=2, so ℝ<sup>3</sup>)
Base vertex 0 (base<sub>0</sub>): (0.1000000000000000, 0.0000000000000000, 0.0000000000000000)
Base vertex 1 (base<sub>1</sub>): (0.9000000000000000, 0.8507462686567164, 0.0000000000000000)
Base vertex 2 (base<sub>2</sub>): (0.1800000000000000, 0.0223880597014925, 0.0000000000000000)
Apex vertex: (0.3933333333333333, 0.2910447761194030, 0.4893037633387919)
Base Centroid Calculation (b)
Base vertices contributing:
V0: (0.1000000000000000, 0.0000000000000000, 0.0000000000000000)
V1: (0.9000000000000000, 0.8507462686567164, 0.0000000000000000)
V2: (0.1800000000000000, 0.0223880597014925, 0.0000000000000000)
Base centroid b: (0.3933333333333333, 0.2910447761194030, 0.0000000000000000)
Direction Vector Calculation (d⃗)
Apex: (0.3933333333333333, 0.2910447761194030, 0.4893037633387919)
Base centroid b: (0.3933333333333333, 0.2910447761194030, 0.0000000000000000)
Direction d⃗ = apex - b: (0.0000000000000000, 0.0000000000000000, 0.4893037633387919)
Vertices (ℝ<sup>D+1</sup>, truncated for display)
Generated code
V0: (0.100, 0.000, 0.000)
V1: (0.900, 0.851, 0.000)
V2: (0.180, 0.022, 0.000)
V_apex: (0.393, 0.291, 0.489) // Renamed V3 to V_apex for clarity
Use code with caution.
Direction & Inverse
Generated code
d⃗   = (0.0000000000000000, 0.0000000000000000, 0.4893037633387919)
d⃗⁻¹ = (∞, ∞, 2.0437202305096602)
Use code with caution.
Magnitude & Inverse
Generated code
mag (|d⃗|)   = 0.4893037633387919
mag⁻¹       = 2.0437202305096602
Use code with caution.
(Note: Displayed as 0.489 and 2.044 in original, using full precision here.)
Unit-Direction & Inverse
Generated code
u⃗   = (0.0000000000000000, 0.0000000000000000, 1.0000000000000000)
u⃗⁻¹ = (∞, ∞, 1.0000000000000000)
Use code with caution.
Velocity & Inverse
Generated code
v⃗   = (0.0000000000000000, 0.0000000000000000, 0.4893037633387919)
v⃗⁻¹ = (∞, ∞, 2.0437202305096602)
Use code with caution.
(Note: Displayed as 0.489 and 2.044 in original, using full precision here.)
Tensor T & Pseudoinverse T<sup>†</sup>
T = u⃗ ⊗ u⃗<sup>T</sup>:
Generated code
[ 0.0000000000000000  0.0000000000000000  0.0000000000000000 ]
 [ 0.0000000000000000  0.0000000000000000  0.0000000000000000 ]
 [ 0.0000000000000000  0.0000000000000000  1.0000000000000000 ]
Use code with caution.
T<sup>†</sup> (Moore-Penrose pseudoinverse):
(As per formula T<sup>†</sup> = (1/||u||<sub>2</sub><sup>2</sup>) * T, and since ||u||<sub>2</sub>=1, T<sup>†</sup> = T)
Generated code
[ 0.0000000000000000  0.0000000000000000  0.0000000000000000 ]
 [ 0.0000000000000000  0.0000000000000000  0.0000000000000000 ]
 [ 0.0000000000000000  0.0000000000000000  1.0000000000000000 ]
