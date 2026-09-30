# Topological Quantum Codes, the Surface Code, and Union-Find Decoding

A guide from quantum error correction to a practical classical decoding algorithm.

## How to read this document

Here, **topological codes** means topological quantum error-correcting codes. **Union-find** refers both to the classical disjoint-set data structure and to the quantum decoder that uses it. The guide explains both and connects them through graph parity.

The prerequisites are basic linear algebra, binary arithmetic, and familiarity with graphs. Quantum concepts are introduced as needed. Binary vectors and edge sets are added modulo two. The examples are illustrative, and the pseudocode describes algorithms rather than a complete production implementation.

### Contents

1. [Why quantum error correction is needed](#1-why-quantum-error-correction-is-needed)
2. [Stabilizers and syndromes](#2-stabilizers-and-syndromes)
3. [What makes a code topological](#3-what-makes-a-code-topological)
4. [Constructing the surface code](#4-constructing-the-surface-code)
5. [Error chains and logical failure](#5-error-chains-and-logical-failure)
6. [The binary algebra](#6-the-binary-algebra)
7. [Repeated measurements and spacetime](#7-repeated-measurements-and-spacetime)
8. [The decoding problem](#8-the-decoding-problem)
9. [The union-find data structure](#9-the-union-find-data-structure)
10. [The union-find decoder](#10-the-union-find-decoder)
11. [Peeling a forest into a correction](#11-peeling-a-forest-into-a-correction)
12. [Worked examples](#12-worked-examples)
13. [Complexity and limitations](#13-complexity-and-limitations)
14. [Implementation and evaluation](#14-implementation-and-evaluation)
15. [Common misconceptions](#15-common-misconceptions)
16. [Glossary and further reading](#16-glossary-and-further-reading)

## 1. Why quantum error correction is needed

A qubit can occupy a superposition

$$
|\psi\rangle=a|0\rangle+b|1\rangle,
\qquad |a|^2+|b|^2=1.
$$

Measuring it in the computational basis reveals an outcome and generally destroys the original superposition. We cannot protect an unknown state simply by reading its value repeatedly, and we cannot make arbitrary independent copies of it.

Instead, quantum error correction encodes information into a joint state of several physical qubits. Measurements reveal changes in relationships among those qubits without revealing the encoded amplitudes.

The basic error operators are the Pauli matrices:

$$
X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix}=iXZ.
$$

- **X** exchanges the computational-basis amplitudes: a bit flip.
- **Z** changes their relative sign: a phase flip.
- **Y** combines the two, up to an overall phase.

The identity and these three matrices form a basis for single-qubit operators. That explains why correcting Pauli errors is fundamental even when physical noise is more general. A stochastic Pauli model remains a modeling choice: coherent noise, leakage, and correlated faults can require additional treatment.

A code denoted $[[n,k,d]]$ uses $n$ physical data qubits to encode $k$ logical qubits. The **distance** $d$ is the smallest weight of a nontrivial logical Pauli operator. Weight counts the qubits on which the operator acts nontrivially.

A distance-$d$ code admits recovery from every unknown Pauli error of weight at most

$$
t=\left\lfloor\frac{d-1}{2}\right\rfloor.
$$

This is a property of the code with a suitable recovery procedure. A particular approximate decoder needs its own correctness argument.

## 2. Stabilizers and syndromes

### 2.1 The code space

A stabilizer code is specified by commuting Pauli operators $S_1,\ldots,S_r$. Code states satisfy

$$
S_i|\psi_L\rangle=|\psi_L\rangle
\quad\text{for every }i.
$$

The stabilizer group must exclude $-I$. With $r$ independent generators on $n$ qubits, the code encodes $k=n-r$ logical qubits.

For example, $Z_1Z_2$ and $Z_2Z_3$ stabilize the space spanned by $|000\rangle$ and $|111\rangle$. Their measurement reveals agreement between neighboring bits without revealing the amplitudes of $a|000\rangle+b|111\rangle$.

That repetition code protects against bit flips, but a single Z changes the encoded relative phase. Full quantum protection requires both kinds of checks.

### 2.2 Turning errors into classical information

For a Pauli error $E$, each check either commutes or anticommutes with it:

$$
S_iE=(-1)^{s_i}ES_i.
$$

The syndrome bit $s_i$ is zero for commutation and one for anticommutation. Measuring the check on the errored code state returns $(-1)^{s_i}$.

The essential identity is $XZ=-ZX$. An X-type check detects a Z component, and a Z-type check detects an X component. A Y error can affect both check families.

The **syndrome** is the classical vector of check violations. It identifies parity constraints that have changed, rather than directly identifying the physical error.

### 2.3 What successful recovery means

Let $E$ be the actual error and $R$ the selected correction. Recovery succeeds if

$$
RE\in\mathcal S
$$

up to an irrelevant overall phase, where $\mathcal S$ is the stabilizer group. A stabilizer acts as identity on every code state, so identifying the exact physical error is unnecessary.

If $R$ and $E$ have the same syndrome, $RE$ commutes with all checks. It can still be a nontrivial logical operator. In that case the syndrome is cleared, but the encoded information has changed.

This distinction is essential throughout decoding: **syndrome consistency is necessary but does not establish logical success.**

## 3. What makes a code topological

In surface codes, checks are local on a lattice while logical operators depend on its global geometry. Small neighborhoods provide error information; extended paths carry logical information.

Imagine loops drawn on a tiled surface. A small loop surrounding one face can be a stabilizer. Multiplying by adjacent face stabilizers deforms it. On a torus, a loop winding around the surface cannot be contracted to a point by those local deformations. Such a loop can represent a logical operator.

**Homology** formalizes this: closed paths are equivalent when their difference is the boundary of a collection of faces. Planar codes allow paths to end on specified boundaries, leading to relative homology.

The connection between anyons, topology, and quantum information originates in foundational constructions such as [Kitaev, *Fault-tolerant quantum computation by anyons*](https://arxiv.org/abs/quant-ph/9707021).

### 3.1 Defects as anyons

A violated check can be interpreted as an excitation, often called an anyon. A local bulk error creates a pair. Extending an error string moves its endpoints. Closing a string removes the endpoint syndrome, although a topologically nontrivial closed string may still act logically.

This language is useful intuition. A practical surface-code memory still requires repeated measurements and classical decoding; topology does not cause ordinary hardware noise to repair itself automatically.

### 3.2 The wider family

Color codes and higher-dimensional topological codes have different check structures, excitations, and logical operators. Locality alone does not make every code identical to a surface code.

The ordinary graph decoder below relies on a fault component having one or two relevant syndrome endpoints. More general fault structures may require graph transformations, hypergraphs, or different decoding algorithms.

## 4. Constructing the surface code

### 4.1 Fixing an operator convention

Start with a square lattice and put a data qubit on each edge. Define

$$
A_v=\prod_{e\ni v}X_e,
\qquad
B_f=\prod_{e\in\partial f}Z_e.
$$

$A_v$ is a vertex or star check. $B_f$ is a face or plaquette check. Each bulk check has weight four.

A vertex and adjacent face overlap on two edges. Each shared X/Z pair contributes a minus sign when exchanged; the two signs cancel. Checks with no overlap commute, as do checks of the same Pauli type. Thus all these checks commute.

```text
vertex --- q --- vertex
   |               |
   q     face      q
   |               |
vertex --- q --- vertex
```

Here `q` denotes an edge data qubit. Z errors form chains on this primal lattice. X errors form chains on its dual lattice, whose vertices are face centers.

### 4.2 The toric code

Identify opposite sides of an $L\times L$ square lattice. For the standard periodic construction,

$$
n=2L^2,\qquad k=2,\qquad d=L.
$$

There are $L^2$ star checks and $L^2$ plaquette checks. Two global dependencies are

$$
\prod_v A_v=I,\qquad\prod_f B_f=I.
$$

Every edge appears twice in each relevant product. There are therefore $2L^2-2$ independent checks, leaving two logical qubits. Noncontractible paths along the two independent directions of the torus supply logical operators.

### 4.3 Planar patches and boundaries

A planar surface code replaces periodic connections with boundaries. A standard patch with two boundaries of each appropriate type encodes one logical qubit.

This document calls a boundary **rough** when a Z string can end there without an endpoint X-check defect. A **smooth** boundary similarly absorbs endpoints of X strings. When reading another source, identify its operator convention explicitly.

A logical Z connects opposite rough boundaries. A logical X connects opposite smooth boundaries on the dual graph. Representatives crossing on one data qubit anticommute:

$$
\overline X\,\overline Z=-\overline Z\,\overline X.
$$

Boundary checks must be designed consistently. Deleting arbitrary checks from a torus does not automatically create the intended planar code.

### 4.4 Rotated patches

A standard rotated square patch has $d^2$ data qubits, one logical qubit, and $d^2-1$ independent checks. With one dedicated measurement ancilla for each check, its data-plus-ancilla count is $2d^2-1$. Extra hardware resources are outside that count.

For distance three, this gives nine data qubits and eight check ancillas. One arbitrary data-qubit Pauli error is correctable with suitable decoding and ideal syndrome measurements.

The edge-lattice picture and rotated-patch layout use different drawings and check supports. Before interpreting a diagram, locate its data qubits, X checks, Z checks, and boundaries. A detailed operational introduction is [Fowler et al., *Surface codes: Towards practical large-scale quantum computation*](https://arxiv.org/abs/1208.0928).

## 5. Error chains and logical failure

### 5.1 Why endpoints appear

Consider two Z errors on consecutive primal edges:

```text
u ----Z---- v ----Z---- w
```

The checks at $u$ and $w$ each overlap one error and flip. The check at $v$ overlaps two errors, so its two anticommutation signs cancel. Only the endpoints are visible.

Represent an edge set by a binary chain $e$. Its boundary $\partial e$ consists of vertices incident to an odd number of selected edges:

$$
s=\partial e.
$$

Addition of chains is symmetric difference, written $\oplus$: shared edges cancel.

### 5.2 Residual cycles

A correction chain $r$ must satisfy $\partial r=s$. Consequently,

$$
\partial(e\oplus r)=\partial e\oplus\partial r=0.
$$

On a torus, the residual is a union of cycles. If it is a boundary of faces, its operator is a stabilizer. If it has nontrivial winding, it can produce a logical error.

For planar patches the analogous statement uses relative boundaries. A path joining appropriate opposite boundaries can have no measured endpoints while acting logically.

The decoder therefore needs a correction in the actual error's logical equivalence class, rather than merely one with matching endpoints.

### 5.3 The distance argument

Suppose two errors share a syndrome and each has weight at most $t$. Their product has zero syndrome and weight at most $2t$. If $2t<d$, it cannot be a nontrivial logical Pauli. It must be a stabilizer up to phase.

Thus no two errors in that weight range belong to different logical classes while sharing a syndrome. This explains why a distance-$d$ code admits correction of up to $\lfloor(d-1)/2\rfloor$ arbitrary errors.

## 6. The binary algebra

Surface codes are CSS codes: generators separate into X-only and Z-only families. Let $H_X$ and $H_Z$ be their binary support matrices. Rows identify the qubits participating in each check.

Commutation requires

$$
H_XH_Z^T=0\pmod 2.
$$

Write a Pauli error, ignoring phase, as $E\propto X^xZ^z$. Then

$$
s_Z=H_Zx,\qquad s_X=H_Xz.
$$

Here the syndrome subscript names the measured check type. To recover, estimates $r_X,r_Z$ must satisfy $H_Zr_X=s_Z$ and $H_Xr_Z=s_X$.

For the Z sector, genuine success means

$$
z+r_Z\in\operatorname{rowspan}(H_Z).
$$

The weaker condition $H_X(z+r_Z)=0$ only clears the syndrome. Logical Z classes are described by

$$
\ker(H_X)/\operatorname{rowspan}(H_Z).
$$

Geometrically, this quotient distinguishes closed chains from face boundaries.

### A small matrix example

For the path $a-b-c-d$, label the three edges in order. Its incidence matrix is

$$
D=\begin{pmatrix}
1&0&0\\
1&1&0\\
0&1&1\\
0&0&1
\end{pmatrix}.
$$

Selecting all three gives $D(1,1,1)^T=(1,0,0,1)^T$: two endpoints and no interior defects. This simple parity calculation is the algebra behind graph decoding.

## 7. Repeated measurements and spacetime

### 7.1 Extracting a check with an ancilla

A Z-parity check can be measured by preparing an ancilla in $|0\rangle$, applying CNOTs from each participating data qubit to the ancilla, and measuring the ancilla in Z.

For an X-parity check, prepare the ancilla in $|+\rangle$, use it as CNOT control with participating data qubits as targets, and measure it in X.

These are ideal check circuits. In a real schedule, an ancilla fault can propagate onto several data qubits. Such hook errors make gate ordering relevant to the circuit's effective distance.

### 7.2 Detection events

An incorrect measurement can imitate a data error when only one round is examined. Repeated rounds provide temporal information. If $m_v(t)$ is the binary result of check $v$ in round $t$, an interior detection event is

$$
\delta_v(t)=m_v(t)\oplus m_v(t-1).
$$

Initialization determines the reference for the first comparison. Final data measurements can provide closing parity information for the measured basis. These temporal boundaries need explicit treatment.

In a simple model, data errors produce detection events separated in space, while readout errors produce events separated in time. The resulting decoding graph has two spatial coordinates and a time coordinate. The quantum hardware can remain a two-dimensional array.

### 7.3 Noise models

| Model | Included noise | Typical graph |
|---|---|---|
| Code capacity | Data errors; perfect checks | A spatial graph |
| Phenomenological | Data errors and faulty check results | A spacetime graph |
| Circuit level | Faulty gates, preparation, idle periods, and measurements | Detectors derived from the circuit |

Circuit faults can create diagonal edges, correlated edges, or more than two detection events. A plain graph may need an approximation or a richer decoding method. For the foundational treatment of noisy syndrome histories, see [Dennis et al., *Topological quantum memory*](https://arxiv.org/abs/quant-ph/0110143).

## 8. The decoding problem

### 8.1 Probability and weight

Under independent binary edge faults with probabilities $p_e<1/2$, a candidate edge set $F$ has probability

$$
P(F)=\prod_{e\in F}p_e\prod_{e\notin F}(1-p_e).
$$

Taking its negative logarithm and removing a constant produces the objective

$$
\sum_{e\in F}w_e,\qquad w_e=\log\frac{1-p_e}{p_e}.
$$

Thus the most probable individual configuration minimizes additive weight subject to the syndrome constraints. Equal edge probabilities reduce this to minimum cardinality.

However, a logical class contains many stabilizer-equivalent configurations. Its total probability is their sum. The most probable individual error need not belong to the most probable logical class. This distinction matters in degenerate quantum codes.

### 8.2 Matching versus clustering

Minimum-weight perfect matching pairs defects, or connects them to compatible boundaries, using graph path costs. Union-find decoding grows regions around defects until each region can explain its syndrome, then constructs a correction within those regions.

Basic union-find does not generally return the global minimum-weight correction. Its appeal is efficient connectivity management and a simple growth-and-recovery structure. Practical comparisons depend on the noise model, graph, implementation, and execution hardware.

### 8.3 X and Z components

CSS structure allows separate X and Z decoding problems. Under independent component noise this is natural. Depolarizing noise includes Y errors, which correlate those components. Separate decoders discard such information unless an additional procedure shares it between the two sectors.

## 9. The union-find data structure

Union-find maintains a partition of objects into disjoint sets:

- `make_set(v)` creates a singleton.
- `find(v)` returns its current set representative.
- `union(u, v)` merges the sets containing two objects.

Starting with `{a}, {b}, {c}, {d}`, the operations `union(a, b)` and `union(c, d)` produce `{a, b}` and `{c, d}`. After `union(b, c)`, all four objects share a representative.

### 9.1 Parent pointers and efficient merging

Store each object in a parent tree. Roots point to themselves. Finding a representative means following parent pointers.

Two optimizations are standard:

1. **Union by size or rank:** attach the smaller or shallower tree to the other root.
2. **Path compression:** redirect visited nodes toward the root during lookup.

```python
# Initially: parent[v] = v and size[v] = 1 for every vertex.
def find(v):
    while parent[v] != v:
        parent[v] = parent[parent[v]]  # Path halving.
        v = parent[v]
    return v

def union(u, v):
    ru, rv = find(u), find(v)
    if ru == rv:
        return ru
    if size[ru] < size[rv]:
        ru, rv = rv, ru
    parent[rv] = ru
    size[ru] += size[rv]
    return ru
```

Across a sequence of operations, the amortized cost per operation is $O(\alpha(N))$, where $\alpha$ is the inverse Ackermann function. It grows extremely slowly.

### 9.2 Decoder metadata

A decoder additionally stores cluster parity, allowed boundary contact, an active flag, and a growth frontier. When disjoint clusters merge, their parity combines by XOR and their boundary-contact flags by logical OR.

A union-find parent tree represents membership. Its pointers can connect distant graph vertices. **Those pointers are not physical correction edges.** Recovery needs a separate forest made from actual decoding-graph edges.

## 10. The union-find decoder

Union-find decoding combines growing connected regions with an erasure-style recovery stage. The algorithm and its analyzed implementation are introduced in [Delfosse and Nickerson, *Almost-linear time decoding algorithm for topological codes*](https://arxiv.org/abs/1709.06218).

The following explanation derives the core mechanism from graph parity. It works on one Pauli sector at a time.

### 10.1 When is a cluster valid?

For a cluster $C$, define

$$
q(C)=\bigoplus_{v\in C}s_v.
$$

If a connected cluster has no absorbing boundary, every selected internal edge contributes two endpoints. Its total syndrome parity must therefore be even. An odd cluster cannot explain its own syndrome using only internal edges.

Even parity is also sufficient: pair the defects arbitrarily, connect each pair by a path inside the cluster, and XOR the paths. Their boundary is exactly the requested syndrome.

If the cluster reaches an allowed absorbing boundary, an unmatched defect can connect to that boundary. Hence

$$
\operatorname{valid}(C)
\iff q(C)=0\;\text{or}\;C\text{ touches an allowed boundary}.
$$

Validity means that a syndrome-consistent solution exists in the cluster. It does not establish that the solution preserves the logical state.

### 10.2 Initialization and merging

Without known erasures, initialize singleton sets at graph vertices. Defect vertices begin as active odd clusters; zero-syndrome singletons are inactive.

Grow active clusters outward. When growth completes an edge connection between distinct clusters, merge them and update metadata.

| Before merging | Resulting parity |
|---|---|
| Odd + odd | Even |
| Odd + even | Odd |
| Even + even | Even |

An even cluster stops initiating growth but can still be absorbed. If an odd cluster reaches an even cluster, their union is odd and may need to grow further.

### 10.3 Half-edge growth

For unit-length edges, half-edge steps let active clusters approaching from opposite ends meet midway. The implementation must track growth progress along each edge, rather than merely marking its endpoints as occupied.

A clear synchronous round is:

1. Collect distinct active roots at the start of the round.
2. Advance their frontiers by half an edge.
3. Collect newly completed edges.
4. Merge the corresponding endpoint clusters.
5. Refresh metadata and frontier lists; remove duplicate roots.

This prevents iteration order from giving a newly merged cluster an extra growth step.

### 10.4 Known erasures

An erasure marks a known unreliable location without specifying its Pauli error. Initialize erased edges as already included in the support and union their endpoints before growth.

Growth then enlarges that support until it can explain the syndrome. This is a mathematical support construction: growing an edge does not physically erase another qubit.

### 10.5 Conceptual pseudocode

```text
decode_one_sector(graph, syndrome, erased_edges):
    initialize singleton sets and metadata
    support = erased_edges
    union endpoints of erased edges
    compute cluster parity, boundary contact, and frontiers

    while an invalid cluster exists:
        active = distinct roots of invalid clusters
        completed_edges = grow_frontiers_half_step(active)
        add completed_edges to support
        union endpoint clusters of completed_edges
        update metadata and frontiers

    forest = spanning_forest_using_actual_edges(support)
    return peel(forest, syndrome, allowed_boundaries)
```

The input syndrome must be feasible under the graph's boundary rules. A closed connected graph cannot explain an odd total syndrome. Detecting an impossible input is preferable to growing indefinitely.

## 11. Peeling a forest into a correction

Growth identifies support; peeling selects correction edges.

Choose a spanning tree of each support component. Root a component at an absorbing boundary vertex if it has one. Otherwise choose any vertex.

Process leaves toward the root. If the current bit at a non-root vertex is one, select its edge to its parent and toggle the parent's bit. Then remove that vertex from consideration.

```text
correction = empty edge set
remaining = copy of syndrome

for each tree:
    choose an absorbing boundary root if available
    otherwise choose any root

    for v in postorder, excluding root:
        p = tree_parent[v]
        if remaining[v] == 1:
            toggle edge (v, p) in correction
            remaining[p] ^= 1
        remaining[v] = 0

    if root is not an absorbing boundary:
        require remaining[root] == 0
```

For this construction, assign initial bit zero to non-root boundary vertices. Enforcing zero there is a permissible choice; the selected boundary root absorbs unconstrained residual parity.

### Why it works

A leaf has only one remaining incident edge. Its bit therefore forces whether that edge is selected. Removing the satisfied leaf produces a smaller instance of the same problem.

The root's final bit is the XOR of all initial tree bits. It vanishes for an even component. An absorbing boundary root is exempt from that parity constraint.

A spanning tree retains enough connectivity to solve the equations, though different trees can yield different corrections. Peeling as an erasure recovery method is developed in [Delfosse and Zémor, *Linear-Time Maximum Likelihood Decoding of Surface Codes over the Quantum Erasure Channel*](https://arxiv.org/abs/1703.01517).

## 12. Worked examples

### 12.1 Two endpoint defects

```text
a ----- b ----- c ----- d
1       0       0       1     syndrome
```

Odd clusters at $a$ and $d$ grow toward each other. When their support connects, their union has two defects and becomes even.

Rooting the path at $d$ gives:

| Removed vertex | Current bit | Selected edge | Parent update |
|---|---:|---|---|
| a | 1 | a–b | b becomes 1 |
| b | 1 | b–c | c becomes 1 |
| c | 1 | c–d | d becomes 0 |

All three edges are selected. Interior vertices have two incident correction edges and zero syndrome; the endpoints each have one.

### 12.2 One defect at a boundary

```text
absorbing boundary B ----- u ----- v*
```

The star marks a defect. Growth from $v$ reaches $B$, making the odd cluster valid. Rooting at $B$ and peeling selects both edges. The endpoint at $B$ is allowed, so every measured constraint is satisfied.

An implementation that requires even parity everywhere mishandles this example.

### 12.3 An even cluster becomes odd again

Suppose defects $a$ and $b$ have already merged. Their cluster is even and stops growing. An odd cluster containing defect $c$ later reaches it.

The new parity is $1\oplus1\oplus1=1$. Unless it touches an absorbing boundary, the merged cluster resumes growth. Inactive does not mean permanently disconnected from future processing.

### 12.4 Same syndrome, logical failure

On a torus, an actual error can follow a short path between two defects. An alternative path between them can wrap around the periodic direction.

Both have the same endpoints. If the alternative is chosen as correction, its symmetric difference with the actual error can wind around the torus. The final syndrome is zero, but a logical operator remains.

### 12.5 A readout fault

A check with true value zero can have one incorrectly reported result:

```text
round:             0  1  2  3  4
reported result:   0  0  1  0  0
detection event:      0  1  1  0
```

Two neighboring detection events can be explained by one time-direction edge. A decoder restricted to round two lacks the later observation that helps identify the measurement fault.

## 13. Complexity and limitations

### 13.1 The size of the decoding instance

The analyzed union-find decoder has complexity $O(N\alpha(N))$ with suitable bookkeeping. Its specified algorithm also has a recovery guarantee for $t$ known erasures and $s$ unknown errors when $t+2s<d$ under the paper's assumptions. These statements do not automatically extend to arbitrary growth heuristics. [Delfosse and Nickerson](https://arxiv.org/abs/1709.06218).

Here $N$ should mean the size of the actual decoding instance. A bounded-degree spatial graph scales as $O(d^2)$. A spacetime block of $T$ rounds typically has size $O(d^2T)$; taking $T=O(d)$ gives $O(d^3)$.

Fast union-find operations alone do not establish fast decoding. Repeatedly rescanning entire clusters to reconstruct their frontiers can dominate runtime. Initialization, growth bookkeeping, support construction, and peeling must also have controlled total work.

### 13.2 Different meanings of weighting

| Mechanism | Purpose |
|---|---|
| Union by size/rank | Keeps internal parent trees shallow |
| Cluster growth priority | Chooses which active region grows next |
| Probability-based edge weights | Changes the cost or growth distance of edges |

These are distinct choices. Changing growth priorities or edge costs can change the selected correction. A variant needs its own analysis before inheriting another algorithm's guarantees.

### 13.3 Thresholds require context

A threshold describes whether increasing code size improves logical reliability under a specified family of experiments. It is not a universal constant attached to the term surface code.

Any numerical threshold should identify code geometry, noise model, decoder variant, number of rounds, and failure metric. A perfect-measurement threshold cannot be directly compared with a gate error threshold from a noisy circuit.

Distance and threshold answer different questions. Distance gives a finite-weight guarantee. Threshold behavior concerns probabilities as the code grows, including configurations with many separated errors that remain recoverable.

### 13.4 Practical limits

Basic union-find makes local growth decisions. It can select a less probable correction than a global optimization procedure. Correlated noise and inaccurate graph construction can further reduce performance.

Algorithmic work, wall-clock latency, memory consumption, and throughput are separate measurements. An almost-linear bound alone does not establish that a particular implementation keeps pace with a device's measurement cycle.

## 14. Implementation and evaluation

### 14.1 Essential representations

| Object | Information to retain |
|---|---|
| Check/detector vertex | Stable identifier and syndrome bit |
| Decoding edge | Endpoints and associated fault/correction action |
| Boundary endpoint | Which syndrome sector it can absorb |
| Edge support | Unoccupied, partially grown, completed, or erased |
| Union-find root | Size/rank, parity, contact flags, activity |
| Frontier | Locations from which growth can continue |
| Recovery forest | Actual decoding edges used for peeling |

For a spacetime graph, retain the edge-to-correction mapping. A measurement-fault edge can explain detectors without implying a data-qubit Pauli update. Logical-observable effects also need tracking.

Virtual boundary representations must preserve the intended connectivity and parity rules. Accidentally connecting distant boundary locations through a zero-cost growth region changes the algorithm.

### 14.2 Useful invariants

Root parity should equal the XOR of original syndrome bits in its component. Boundary metadata should reflect actual allowed contact. Every recovery-forest edge must belong to the completed support.

After recovery, verify $\partial r=s$ on constrained vertices. In simulation, also evaluate the logical action of $e\oplus r$.

For a full CSS simulation, test residual components against their stabilizer row spaces, or check their commutation with a complete set of logical operators. Syndrome clearing alone is an inadequate success metric.

### 14.3 A progression of tests

Begin with an empty syndrome, each single-qubit X/Z/Y error, simple paths, plaquette loops, and boundary terminations. Add odd clusters absorbing even clusters.

On a small torus, inject a noncontractible loop. It has no syndrome, so the decoder may return no correction, but the evaluator must recognize its logical effect.

For repeated measurements, include isolated readout faults and faults near the initial and final time boundaries. Then compare randomized small-code trials with an independent reference calculation.

### 14.4 Performance reporting

Useful measurements include logical failure probability, average latency, tail latency, peak memory, and throughput. State whether failure is measured per round or per experiment, and include sampling uncertainty. Zero observed failures in a finite sample does not establish zero failure probability.

### 14.5 Pauli frames and streaming

A system can store inferred Pauli corrections in a classical **Pauli frame** instead of applying physical gates immediately. The frame changes the interpretation of later measurements and must be propagated through subsequent operations.

A streaming decoder must distinguish committed history from its active time window. Discarding information or finalizing corrections too early can introduce artificial temporal-boundary effects.

## 15. Common misconceptions

**A syndrome identifies the faulty qubits.** It identifies violated constraints. Multiple error chains can produce the same observations.

**Zero syndrome means no error occurred.** Stabilizers and nontrivial logical operators both have zero syndrome.

**Union-find alone supplies a correction.** The data structure maintains membership. Growth rules and a recovery stage provide decoding behavior.

**Every valid cluster has even parity.** An absorbing boundary can resolve an odd cluster.

**The parent tree is a correction path.** Parent pointers can connect non-neighboring vertices. Physical recovery uses actual graph edges.

**Independent X and Z decoding preserves all noise information.** It can discard correlations, including those associated with Y errors.

**Distance three corrects any three errors.** It admits guaranteed correction of one arbitrary unknown Pauli error with suitable recovery and ideal syndrome information.

**Topological protection removes active correction.** Practical surface-code memories repeatedly measure checks and process the results.

## 16. Glossary and further reading

### Glossary

| Term | Meaning |
|---|---|
| Physical qubit | A device-level data qubit or measurement ancilla |
| Logical qubit | Quantum information encoded in a code space |
| Stabilizer | An operator acting as identity on every code state |
| Syndrome | Measured check violations |
| Detection event | An unexpected check relationship, often a change across rounds |
| Defect | A nonzero syndrome or detection-event location |
| Distance | Minimum weight of a nontrivial logical Pauli |
| Error chain | Graph edges representing an error component |
| Homology class | Paths equivalent under addition of local face boundaries |
| Erasure | Unreliable information at a known location |
| Union-find | A structure for maintaining merging disjoint sets |
| Peeling | Solving tree parity constraints from leaves toward roots |
| Pauli frame | A classical record of inferred Pauli corrections |

### Primary references

1. **A. Y. Kitaev, “Fault-tolerant quantum computation by anyons.”** Foundational connection between topology, anyons, and quantum computation. [Paper](https://arxiv.org/abs/quant-ph/9707021).
2. **E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, “Topological quantum memory.”** Homological error correction and noisy syndrome histories. [Paper](https://arxiv.org/abs/quant-ph/0110143).
3. **A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, “Surface codes: Towards practical large-scale quantum computation.”** Detailed introduction to surface-code operation and architecture. [Paper](https://arxiv.org/abs/1208.0928).
4. **N. Delfosse and N. H. Nickerson, “Almost-linear time decoding algorithm for topological codes.”** Union-find decoding, implementation, guarantees, and variants. [Paper](https://arxiv.org/abs/1709.06218).
5. **N. Delfosse and G. Zémor, “Linear-Time Maximum Likelihood Decoding of Surface Codes over the Quantum Erasure Channel.”** Erasure decoding and peeling. [Paper](https://arxiv.org/abs/1703.01517).

### The central picture

Local checks turn quantum errors into classical parity constraints. Surface-code geometry represents those constraints as endpoints of paths. Union-find decoding grows and merges regions until each can explain its defects; peeling produces a correction. Recovery succeeds when the actual error and that correction differ only by stabilizers.
