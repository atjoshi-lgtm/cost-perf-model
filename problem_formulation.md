# Joint Cost–Performance Optimization of CDN Disk Capacity

## 1. Setting

A content-delivery network (CDN) serves traffic to end-users through a global set of **edge metros** — points of presence (PoPs) each identified by an airport code (e.g., `BOS`, `LHR`, `SIN`). Each metro runs one or more **edge caches**. When a request arrives at an edge cache, it is either:

- **a hit** — the content is already on disk and is served immediately, or  
- **a miss** — the edge cache must fetch the content from a **mid-tier cache hub (MCH)**, incurring extra round-trip time and bandwidth cost.

Each metro belongs to a **geo** (e.g., North America, EMEA, APAC) and may serve traffic originating from many neighboring metros. The question we answer is:

> **How much disk should be provisioned at each edge metro to jointly minimize infrastructure cost and end-user latency, given the available content footprint at each site?**

---

## 2. Inputs

### 2.1 Footprint Descriptor (FDS)

For each metro $m$ we have a **footprint descriptor** — an empirically measured curve:

$$h_m : s \mapsto h_m(s) \in [0, 100]$$

where $s$ is the cache disk allocation (in MB) and $h_m(s)$ is the resulting cache **hit-rate** (%). This curve is monotonically non-decreasing and saturates at a metro-specific maximum (e.g., Boston's FDS saturates at ~77%). The FDS captures the working-set structure of the traffic mix served at that metro.

### 2.2 Traffic

A **traffic matrix** $T_{ij}$ (Mbps) records how much traffic from end-user metro $i$ is served by edge metro $j$. From this we derive:

- **Incoming traffic** $\lambda_j = \sum_i T_{ij}$: total Mbps flowing into edge metro $j$.
- **Neighborhoods**: metro $j$ is an active serving neighbor of $i$ if $T_{ij}$ exceeds a configurable threshold (e.g., 5 Gbps). Only these edges are considered during optimization.

### 2.3 Latency Distributions

Latency at each hop is modeled as a **probability density function (PDF)** over milliseconds:

| Component | Notation | Meaning |
|---|---|---|
| Edge RTT | $f^{\text{rtt}}_{ij}$ | Round-trip time from user metro $i$ to edge metro $j$ |
| Edge TAT (hit) | $f^{\text{tat-hit}}_j$ | Turn-around time at edge on a cache hit |
| Edge TAT (miss) | $f^{\text{tat-miss}}_j$ | Turn-around time at edge on a cache miss (includes MCH fetch) |

PDFs are either loaded from empirical measurements or approximated as Gaussians. Convolution of independent latency components is performed via FFT.

### 2.4 Cost Model

The monthly cost of operating edge metro $j$ with disk allocation $s_j$ and incoming traffic $\lambda_j$ is decomposed into:

$$C_j(s_j, \lambda_j) = r_j \cdot \left[ C^{\mathrm{dep}}(s_j) + C^{\mathrm{colo}}(s_j) + C^{\mathrm{mid}}(\lambda_j, h_j(s_j)) + C^{\mathrm{parent}}(\lambda_j, h_j(s_j)) \right]$$

where:
- $r_j \in \{2, 3, 5\}$ is the **replication factor** (determined by metro tier: Tier-2 = 2×, Tier-1 = 3×, Tier-0 = 5×),
- $C^{\text{dep}}$ is disk **depreciation** cost (capital amortization),
- $C^{\text{colo}}$ is **colocation** cost (rack space, power),
- $C^{\text{mid}}$ is **midgress** cost (bandwidth from edge to MCH on misses),
- $C^{\text{parent}}$ is **parent service** cost (MCH serving the miss traffic).

Traffic-proportional costs ($C^{\text{mid}}, C^{\text{parent}}$) scale with the miss fraction $(1 - h_j(s_j)/100)$ and are computed on the per-replica split traffic $\lambda_j / r_j$.

---

## 3. Performance Model

### 3.1 Time-to-First-Byte (TTFB)

For a user at metro $i$ served by edge metro $j$ with hit-rate $h$, the TTFB is a random variable whose distribution is:

$$\text{TTFB}_{ij}(h) = f^{\text{rtt}}_{ij} \;\circledast\; \left[\frac{h}{100} \cdot f^{\text{tat-hit}}_j + \left(1 - \frac{h}{100}\right) \cdot f^{\text{tat-miss}}_j\right]$$

where $\circledast$ denotes convolution. In words: every request pays the RTT; then it additionally pays either the hit TAT or the miss TAT according to the current hit-rate.

### 3.2 Aggregate Performance per Metro

Traffic from multiple user metros is blended at edge metro $j$:

$$F_j = \sum_i \frac{T_{ij}}{\sum_k T_{kj}} \cdot \text{TTFB}_{ij}(h_j)$$

The **P50** and **P95** of $F_j$ are the key performance metrics, denoted $\tau^{50}_j$ and $\tau^{95}_j$ respectively.

---

## 4. Objective Function

We seek a disk allocation $\mathbf{s} = (s_1, \ldots, s_M)$ for all $M$ active metros that minimizes the combined cost and latency penalty:

$$\boxed{
\min_{\mathbf{s}} \;\; \mathcal{L}(\mathbf{s}) = \underbrace{\sum_{j=1}^{M} C_j\!\left(s_j,\, \lambda_j\right)}_{\text{Infrastructure cost}} + \underbrace{\sum_{j=1}^{M} \Psi_j\!\left(\tau^{50}_j(\mathbf{s}),\, \tau^{95}_j(\mathbf{s})\right)}_{\text{Latency penalty}}
}$$

### 4.1 Penalty Function

The latency penalty for metro $j$ is:

$$\Psi_j(\tau^{50}_j,\, \tau^{95}_j) = 2\,\lambda_j^{\text{GB}} \cdot \max\!\left(\tau^{50}_j - \bar{\tau}^{50}_j,\; 0\right)^2 + 2\,\lambda_j^{\text{GB}} \cdot \max\!\left(\tau^{95}_j - \bar{\tau}^{95}_j,\; 0\right)$$

where:
- $\lambda_j^{\text{GB}} = \lambda_j / 1000$ is the outbound traffic in Gbps (used as a traffic-weighted importance scalar),
- $\bar{\tau}^{50}_j,\; \bar{\tau}^{95}_j$ are **regional latency targets** for metro $j$.

**Asymmetry by design:** the P50 penalty is **quadratic** in the excess — small violations near the target cost little, but large deviations are penalized aggressively. The P95 penalty is **linear** — every millisecond of tail-latency excess costs proportionally regardless of magnitude. This reflects the engineering intuition that P95 tail control is non-negotiable while P50 has diminishing marginal value once already near target.

### 4.2 Regional Targets

Latency targets differ by geographic region, reflecting realistic network physics:

| Region | $\bar{\tau}^{50}$ (ms) | $\bar{\tau}^{95}$ (ms) |
|---|---|---|
| Europe | 24 | 105 |
| Middle East | 45 | 180 |
| Africa | 75 | 220 |
| South Africa | 35 | 200 |
| North America (default) | 24 | 105 |

### 4.3 Coupling Between Metros

The objective is **not separable** across metros. The disk allocation $s_j$ at metro $j$ affects $h_j(s_j)$, which changes $\tau^{50}_j$ and $\tau^{95}_j$, which in turn affects $\Psi_j$. However, $\Psi_j$ is itself weighted by the traffic arriving from *all neighboring user metros*, so the gradient of the penalty at $j$ depends on the full neighborhood traffic structure. This coupling necessitates a global optimization procedure.

---

## 5. Optimization Algorithm

### 5.1 Initialization

Each metro starts at its **cost-optimal point** — the disk allocation $s_j^*$ that minimizes $C_j$ alone (ignoring latency). This is found by scanning the FDS curve and evaluating cost at each integer hit-rate percentile.

### 5.2 Gradient Computation

At each iteration, the **marginal gradient** for metro $j$ is computed by probing a fixed disk increment $\Delta s$ (5 TB):

$$g_j = \left[C_j(s_j + \Delta s) - C_j(s_j)\right]_{\mathrm{cost}} + \left[\Psi_j(s_j + \Delta s) - \Psi_j(s_j)\right]_{\mathrm{perf}}$$

A **negative gradient** ($g_j < 0$) means adding disk reduces the combined objective — i.e., the latency savings outweigh the infrastructure cost increase.

Gradients are computed for all metros **in parallel** using a thread pool.

### 5.3 Budget-Proportional Update

Rather than a fixed step size, a **budget** is allocated each iteration:

$$B = N^{-}_{\text{iter}} \times \Delta_{\text{TB}}$$

where $N^{-}_{\text{iter}}$ is the number of metros with a negative gradient and $\Delta_{\text{TB}} = 10\,\text{TB}$ is the per-metro budget quantum. The budget is distributed across metros proportionally to the magnitude of their gradient:

$$\delta s_j = \text{round\_up\_TB}\!\left(\frac{|g_j|}{\sum_k |g_k|} \cdot B\right)$$

Metros with $g_j < 0$ receive additional disk; metros with $g_j > 0$ lose disk (down to their FDS minimum). This implements a soft form of **Frank-Wolfe / conditional gradient** descent adapted to the discrete FDS structure.

### 5.4 Convergence

The process iterates until the objective $\mathcal{L}$ stops improving. At each iteration the per-metro state (disk, hit-rate, cost, penalty, P50, P95, gradient components) and the summary-level objective are logged to disk for post-hoc analysis.

---

## 6. Summary of Structural Assumptions

| Assumption | Justification |
|---|---|
| FDS curve is fixed and pre-measured | Cache hit-rate depends on content working set, not on optimizer decisions made here |
| Latency distributions are independent across hops | Standard queueing approximation; validated empirically |
| Cost model is separable across metros | Each PoP is billed independently |
| Replication factor is fixed per metro tier | Determined by reliability SLAs, not by cost optimization |
| Traffic matrix is stationary | Optimization runs on a representative traffic snapshot; re-run periodically as traffic shifts |
| P50 penalty quadratic, P95 penalty linear | Engineering policy choice reflecting tail-latency importance |
