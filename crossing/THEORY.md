# Crossing a class-score boundary under softened prototype attraction

Consider any fixed finite collection of labeled prototypes $a_i\in\mathbb R^d$ with positive weights $w_i$, symmetric positive-definite precision matrices $P_i$, and bandwidth variance $\beta>0$. Each class has at least one prototype. Set

$$
u_i(z)=w_i\exp\left[-\frac{(z-a_i)^\top P_i(z-a_i)}{2\beta}\right],
\quad S_c=\sum_{i:y_i=c}u_i,\quad S=\sum_c S_c,\quad g_c=\beta\log S_c.
$$

The readout is $C(z)=\arg\max_c g_c(z)$ with a fixed rule for exact ties. At horizon $t$, use this same readout at $\Phi_t(z)$, where $\Phi_t$ is the exact flow of $F=\beta\nabla\log S$. This is the unpatched potential in the pilot implementation. The results below concern the finite-horizon score readout. Convergence to a labeled source and first entry into a core ball are separate readouts. [Pilot field and readouts](../gdm/dynamics.py)

## 1. The direction at a class tie

Write

$$
r_i=P_i(a_i-z),\quad \pi_c=S_c/S,\quad
v_c=\mathbb E_{i\mid c} r_i,\qquad
\Pr(i\mid c)=u_i/S_c\quad (y_i=c).
$$

Differentiating the class log sum gives $\nabla g_c=v_c$ and $F=\sum_c\pi_c v_c$. In particular $F$ has at most linear growth and is smooth, so its flow exists for all finite real times. For a margin $m_{ba}=g_b-g_a$, put $D_Fm=\nabla m\cdot F$. The boundary from $a$ to $b$ at a strict regular top tie has unit normal $n_{ab}=(v_b-v_a)/\|v_b-v_a\|$.

**Proposition 1.** At a tie of $a$ and $b$ with $\pi=\pi_a=\pi_b$,

$$
D_Fm_{ba}
=\pi(\|v_b\|^2-\|v_a\|^2)
+\sum_{c\notin\{a,b\}}\pi_c(v_b-v_a)\cdot v_c .
$$

This follows by expanding $(v_b-v_a)\cdot F$. At a strict regular top tie with positive derivative, a small flow segment through the tie has $a$ immediately before the tie and $b$ immediately after it. Smoothness and the strict gaps to other classes give an open set of trajectories with the same directed crossing. For a binary tie the expression reduces to $(\|v_b\|^2-\|v_a\|^2)/2$. In a multiclass model the additional term can decide the direction even when the two norms coincide.

For example, with unit precision, unit bandwidth variance and equal weights, place one source for each of $a,b,c$ at $-1,1,2$ on the line. At $z=0$, $a$ and $b$ tie strictly above $c$, with $\|v_a\|=\|v_b\|=1$. The positive third-class term gives $D_Fm_{ba}=4\pi_c>0$.

For common precision $P$ with arbitrarily many sources, write $\mu_c=\mathbb E_{i\mid c}a_i$, $d_{ba}=\mu_b-\mu_a$ and $o_{ba}=(\mu_a+\mu_b)/2$. At a tie the exact expression becomes

$$
D_Fm_{ba}=-(P^2d_{ba})^\top(z-o_{ba})
+\sum_{c\notin\{a,b\}}\pi_c(P^2d_{ba})^\top(\mu_c-o_{ba}).
$$

The conditional means may vary along the boundary. A full affine description is available for **binary, one source per class, common precision**. Let $d_0=a_b-a_a\ne0$, $o=(a_a+a_b)/2$, and $\ell=\beta\log(w_b/w_a)$. Then

$$
m_{ba}(z)=\ell+(P d_0)^\top(z-o),\qquad
F(z)=-P(z-o)+\tfrac12(\pi_b-\pi_a)P d_0.
$$

On the tie hyperplane $D_Fm_{ba}=-(P^2d_0)^\top(z-o)$. With $n=P d_0/\|P d_0\|$ and the decomposition $z-o=u-\ell n/\|P d_0\|$, $u\perp n$, this is

$$
D_Fm_{ba}=\ell\,n^\top P n-(P^2d_0)^\top u .
$$

With equal weights the entire hyperplane is invariant if and only if $P d_0$ is parallel to $d_0$. In that case uniqueness preserves both strict sides at every finite horizon. If the vectors are unaligned, both crossing directions occur locally on the hyperplane, although a given testing distribution may cover only one direction or miss them both. With $P d_0=\lambda d_0$ and unequal weights the tie flux is the constant $\lambda\ell$, pointing toward the heavier source. A zero at just one tie point, such as the equal-weight midpoint for unaligned $P$, gives no invariance conclusion. At singular ties where $v_b=v_a$, a normal is undefined. One cannot infer invariance from sampling zero flux only on parts of the tie set.

## 2. A computable finite-horizon certificate

Take a query $x$ and radius $R>0$. For any two nonempty source index sets $I,J$, the quantity

$$
D_{IJ}
=\max_{i\in I,\ j\in J}
\left(\|r_i(x)-r_j(x)\|+\|P_i-P_j\|_{\mathrm{op}}R\right)
$$

bounds the corresponding component separation throughout the ball $B(x,R)$. Use $A$ for all sources and identify classes with their index sets. The following constants need only frozen geometry and the query:

$$
\begin{aligned}
\Lambda&=\max_i\|P_i\|_{\mathrm{op}},&
V&=\max_i(\|r_i(x)\|+\|P_i\|_{\mathrm{op}} R),\\
L&=\max\left(\Lambda,\frac{D_{AA}^2}{4\beta}\right),&
B&=\min\bigl(V,\|F(x)\|+LR\bigr),\\
G_{ba}&=D_{ba},&
H_{ba}&=\max_{i\in b,j\in a}\|P_i-P_j\|_{\mathrm{op}}
 +\frac{\max(D_{bb}^2,D_{aa}^2)}{4\beta},\\
M_{ba}&=H_{ba} B^2+G_{ba} L B .
\end{aligned}
$$

To verify the bounds, differentiation of responsibilities gives

$$
Dv_c=-\mathbb E[P_i\mid c]+\beta^{-1}\operatorname{Cov}(r_i\mid c),
\qquad DF=-\mathbb E[P_i]+\beta^{-1}\operatorname{Cov}(r_i).
$$

Every directional variance in the covariance bound is at most one quarter of its squared directional range. The component diameter bounds that range. A difference of two positive-semidefinite matrices has norm at most the maximum of their norms. Thus $\|DF\|\le L$ and $\|F\|\le B$. The pairwise difference of class means is a convex average of cross-class component differences, so $\|\nabla m_{ba}\|\le G_{ba}$. Applying the same argument to the class precision means and covariance difference yields $\|\nabla^2m_{ba}\|\le H_{ba}$. Under common precision the precision-difference term vanishes, as does the covariance term for a pair with one source in each class.

Precision norms and pairwise precision-difference norms can be precomputed. Frobenius norms are inexpensive upper bounds, and diagonal matrices have exact entrywise operator norms. A local refinement is useful when remote sources determine the diameter but have very small responsibilities. On the same ball, set $E_i=(\|r_i(x)\|R+\|P_i\|_{\mathrm{op}}R^2/2)/\beta$. Then $\log u_i$ lies between $\log u_i(x)-E_i$ and $\log u_i(x)+E_i$. Within any class or over all sources, upper-bound responsibility $\bar p_i$ by using the upper log weight for $i$ and the lower log weights for its competitors in the softmax. The exact covariance identity $\operatorname{Cov}(r)=\sum_{i<j}p_ip_j(r_i-r_j)(r_i-r_j)^\top$ gives the alternative upper bound $\sum_{i<j}\bar p_i\bar p_jD_{\{i\},\{j\}}^2$. Take the smaller of this and the diameter-based bound.

**Proposition 2.** If $h B<R$, then for any $0\le t\le h$,

$$
\left|m_{ba}(\Phi_t x)-m_{ba}(x)-t D_Fm_{ba}(x)\right|
\le \tfrac12 M_{ba} t^2.
$$

The speed bound rules out a first exit from the ball by $h$. The second time derivative is $F^\top\nabla^2m_{ba}F+\nabla m_{ba}^\top DF\, F$ and has absolute value at most $M_{ba}$ along the path. Integrating twice proves the inequality. Write $L_{ba}(t)$ for the linear prediction minus the remainder and $U_{ba}(t)$ for the linear prediction plus the remainder.

- If $m_{ba}(x)<0$ and $L_{ba}(h)>0$, the pair reverses order by $h$.
- If $C(x)=a$ strictly and $L_{bc}(h)>0$ for **every** $c\ne b$, then $C(\Phi_h x)=b$ strictly. This endpoint statement permits intervening classes.
- If in addition both initial margins $m_{ac}(x),m_{bc}(x)$ exceed $Bh G_{ac},Bh G_{bc}$ respectively for every third class $c$, the two classes remain above all third classes throughout the segment. If also $D_Fm_{ba}(x)>M_{ba}h$, the pair changes order exactly once on the segment.
- If $C(x)=a$ strictly and every lower bound $L_{ac}(t)$ stays positive on $[0,h]$, the winner is constant on the segment. Each such lower bound is concave in $t$, so checking both endpoints suffices.

Failure of a sufficient inequality leaves the query uncertified. In particular, a positive local direction or a correct linear prediction at one short horizon does not specify the winner at a later horizon. Radii and horizons can be chosen using the frozen geometry. In floating-point experiments, record small slack as unresolved unless a roundoff bound is included. Ordinary solver-refinement agreement is empirical validation rather than an enclosure.

### Continuing a bound over several steps

Suppose a reference Euler path starts the $k$th step at $x_k$ and the exact state is known to be within $e_k$ of it. Compute $B_k,L_k$ on a fresh ball $B(x_k,R_k)$ and take a step $s_k$ with $e_k+B_k s_k<R_k$. Then both the reference-start and exact-start flow segments stay in the ball until the end of the step. For the Euler update $x_{k+1}=x_k+s_kF(x_k)$,

$$
e_{k+1}=\exp(L_k s_k)e_k+\tfrac12 L_k B_k s_k^2
$$

is a valid next error radius in exact arithmetic. The first term bounds dependence on the start by Grönwall's inequality. The second integrates the bound on $F(\Phi_t x_k)-F(x_k)$. A signed margin at $x_{k+1}$ is certified for the exact endpoint when its magnitude exceeds $G_{ba}e_{k+1}$ for a margin bound on that endpoint ball. Use all relevant pair margins for a multiclass winner. This continuation can become loose, particularly at late times. Report the share of queries and horizons certified.

### A tighter certificate for shared identity precision

The implemented follow-up uses $P_i=I$ in its feature coordinates. Removing the common term $-\|z\|^2/2$ from every class score leaves the margins unchanged. The remaining logits are affine with slopes $a_i/\beta$. For a source subset $A$ that is either a class or the entire mixture, write $p_i^A$ for its normalized responsibilities, $D_A$ for its prototype diameter and $C_A(x)$ for the covariance of its prototypes under $p^A(x)$. For $\delta=z-x$,

$$
\frac{p_i^A(z)}{p_i^A(x)}
=\frac{\exp(a_i^\top\delta/\beta)}
{\sum_{j\in A}p_j^A(x)\exp(a_j^\top\delta/\beta)}
\le \exp\!\left(\frac{D_A\|\delta\|}{\beta}\right).
$$

For any direction $u$, variance at $z$ minimizes the expectation of $(u^\top a_i-m)^2$ over $m$. Evaluating that expectation at the mean under $p^A(x)$ and applying the ratio bound gives $u^\top C_A(z)u\le e^{D_A R/\beta}u^\top C_A(x)u$ on $B(x,R)$. The directional range bound then gives

$$
\|C_A(z)\|_{\mathrm{op}}\le \kappa_A(R)
=\min\!\left(\frac{D_A^2}{4},
e^{D_A R/\beta}\operatorname{tr} C_A(x)\right).
$$

For all sources together, $F(z)=\mathbb E a_i-z$ and $DF=C_{\mathrm{all}}/\beta-I$. Hence $DF$ is symmetric, its eigenvalues are at least $-1$, and their upper bound on the ball is $\kappa_{\mathrm{all}}(R)/\beta-1$.

Take a horizon $h$ and $v_0=\|F(x)\|$. Define

$$
I_\lambda(t)=\int_0^t e^{\lambda s}\,ds,\qquad
K_\lambda(t)=\int_0^t(t-s)e^{\lambda s}\,ds.
$$

These are positive for $t>0$ and all real $\lambda$. At $\lambda=0$ they are $t$ and $t^2/2$. For $\lambda\ne0$, they equal $(e^{\lambda t}-1)/\lambda$ and $(e^{\lambda t}-1-\lambda t)/\lambda^2$. Let $\ell_0=D_{\mathrm{all}}^2/(4\beta)-1$. For any trajectory segment with $\lambda_{\max}(DF)\le\ell$, differentiation of $\|F(\Phi_t x)\|^2$ and Grönwall's inequality give

$$
\|F(\Phi_t x)\|\le v_0 e^{\ell t},\qquad
\|\Phi_t x-x\|\le v_0 I_\ell(t).
$$

The global rate yields a containing radius $v_0 I_{\ell_0}(h)$ because $I_{\ell_0}(t)$ increases with $t$. A second global bound follows by solving the linear part of $\dot z=\mu(z)-z$ where $\mu$ is the mixture mean:

$$
\Phi_t x-x=\int_0^t e^{-(t-s)}\big(\mu(\Phi_s x)-x\big)\,ds.
$$

Every $\mu$ is a convex combination of the prototypes. Thus $(1-e^{-h})\max_i\|a_i-x\|$ also bounds displacement for the whole horizon. Start at the minimum $R_0$ of these two radii. The implementation makes two refinements. From any known containing ball $R_k$ and valid rate $\ell_k$, set

$$
\ell_{k+1}=\min\bigl(\ell_k,\ \kappa_{\mathrm{all}}(R_k)/\beta-1\bigr),\qquad
R_{k+1}=\min\bigl(R_k,\ v_0 I_{\ell_{k+1}}(h)\bigr).
$$

The local spectral bound holds along the trajectory already known to be in $B(x,R_k)$. The speed and displacement inequalities therefore give the smaller containing ball for all times up to $h$. This induction does not assume that the refined trajectory stays in the refined ball before proving containment. Both refinements remain valid for negative rates and for the zero-radius stationary case.

On the last ball of radius $R$, use $J=\max(1,\ell)$ so $\|DF\|_{\mathrm{op}}\le J$. For a class pair, the margin gradient is the difference of conditional means, and the Hessian is the difference of their covariance matrices divided by $\beta$. The operator norm of this difference is bounded by the larger norm of the two covariances. If $D_{ba}$ denotes the maximum center distance across the two classes, valid margin bounds are

$$
H=\frac{\max(\kappa_b(R),\kappa_a(R))}{\beta},\qquad
G=\min\bigl(D_{ba},\ \|\nabla m_{ba}(x)\|+HR\bigr).
$$

The second gradient bound uses the Hessian bound along each line segment inside the ball. Substituting the time-dependent speed into the exact identity for $m''$ and integrating twice gives, for every $0\le t\le h$,

$$
\left|m_{ba}(\Phi_t x)-m_{ba}(x)-tD_Fm_{ba}(x)\right|
\le E(t)
=H v_0^2 K_{2\ell}(t)+GJv_0 K_\ell(t).
$$

The lower margin $m_{ba}(x)+tD_Fm_{ba}(x)-E(t)$ is concave since $K_\lambda''(t)=e^{\lambda t}>0$ and all coefficients in $E$ are nonnegative. Consequently, the initial strict winning class is certified throughout $[0,h]$ if each of its lower margins is positive at both ends. A rival's lower margin over that class being positive at $h$ certifies at least one departure from the initial class. Certifying a particular terminal winner requires positivity against every other class at $h$. As a simpler comparison, $m_{ba}(x)-D_{ba}R_0>0$ preserves that pair throughout the global ball.

### Integrating the known linear term exactly

The same identity-precision structure allows a different approximation center. Along the exact path write $m(t)=m_{ba}(\Phi_t x)$ and let $c\ge0$ be the bound on $\|C_{\mathrm{all}}/\beta\|_{\mathrm{op}}$ that produced $\ell=c-1$. The known $-I$ in $DF$ gives

$$
m''(t)+m'(t)
=F^\top \nabla^2m_{ba} F+\nabla m_{ba}^\top(C_{\mathrm{all}}/\beta)F
=q(t).
$$

With the same radius and class bounds, $|q(t)|\le W(t)=H v_0^2 e^{2\ell t}+Gc v_0 e^{\ell t}$. Solving the scalar evolution of $m'$ by variation of constants and integrating gives

$$
m(t)
=m(0)+(1-e^{-t})m'(0)
+\int_0^t(1-e^{-(t-s)})q(s)\,ds.
$$

The kernel is nonnegative, so the approximation $m(0)+(1-e^{-t})m'(0)$ has absolute error bounded by

$$
E_{\mathrm{lin}}(t)=H v_0^2 Q_{2\ell}(t)+Gc v_0 Q_\ell(t),\qquad
Q_\lambda(t)=\int_0^t(1-e^{-(t-s)})e^{\lambda s}\,ds
=I_\lambda(t)-e^{-t}I_{\lambda+1}(t).
$$

This bound is useful where the posterior covariance is small. In particular, $\ell\le0$ gives $c\le1$, while the generic bound above has $J=1$. Also $Q_\lambda(t)\le K_\lambda(t)$ by $1-e^{-v}\le v$ for $v\ge0$. The two approximation centers differ. For positive $\ell$ the error bounds do not have a universal ordering.

The endpoint test still guarantees stability throughout the interval. Indeed, with $u=1-e^{-t}$, the new lower margin is $m(0)+u\,m'(0)-E_{\mathrm{lin}}(t(u))$. Since

$$
\frac{dE_{\mathrm{lin}}}{du}=\int_0^t e^s W(s)\,ds ,
\qquad
\frac{d^2E_{\mathrm{lin}}}{du^2}=e^{2t}W(t)\ge0 ,
$$

the lower margin is concave in $u$. Positive lower margins at $0$ and $h$ imply strict positivity throughout. The code takes the larger of the ordinary and linear-term lower bounds for each pair at $h$. For each certified pair, at least one of the two constituent lower margins is positive at $h$, and that constituent proves whole-interval positivity if the initial margin is positive. This uses concavity of the constituent in its own time coordinate. The combined endpoint bounds also support the terminal and forced-crossing tests. There may still be uncertified queries near ties or where the covariance bound is loose.

### Certifying an endpoint by scoring an enclosed state

A cheaper endpoint-only evaluation uses the shifted state

$$
q(t)=x+(1-e^{-t})F(x)=e^{-t}x+(1-e^{-t})\mu(x).
$$

Variation of constants for the state gives the exact defect

$$
\Phi_t x-q(t)
=\int_0^t e^{-(t-s)}\bigl(\mu(\Phi_s x)-\mu(x)\bigr)\,ds.
$$

The covariance bound gives $\|D\mu\|_{\mathrm{op}}\le c$ on the convex containing ball. Therefore $\|\mu(\Phi_s x)-\mu(x)\|\le c\|\Phi_s x-x\|\le c v_0 I_\ell(s)$. Interchanging the corresponding nonnegative integrals gives

$$
\|\Phi_t x-q(t)\|\le \epsilon(t)
=c v_0\int_0^t e^{-(t-s)}I_\ell(s)\,ds
=c v_0Q_\ell(t),\qquad \ell=c-1 .
$$

Every class margin is globally $D_{ba}$-Lipschitz because its gradient is the difference of two convex combinations of their respective prototypes. Evaluating the original nonlinear score margin at the shifted state yields

$$
m_{ba}(\Phi_t x)\ \ge\ m_{ba}(q(t))-D_{ba}\epsilon(t).
$$

Strict lower bounds versus every rival certify an endpoint winner. The full diagnostic can combine these with the two preceding lower bounds pair by pair. The fast version uses only the state enclosure, so it needs the global posterior mean and covariance trace at $x$, class scores at $q(t)$ and the precomputed cross-class diameters. It does not need classwise Hessian bounds. The function $m_{ba}(q(t))$ can be nonlinear along the segment. Endpoint positivity from this bound alone does not establish whole-interval stability, even when the initial and terminal winners agree.

For example, use $\beta=10$ in one dimension. Class $a$ has centers $-1,1$ with weights $1,2$. Class $b$ has center $0$ with weight $2\sqrt{2}e^{-1/20}\cosh(0.005)$. The winning region of $b$ is exactly the interval of radius $0.05$ around $x_*=-5\log 2$, by the affine logits of the two classes. Start in $a$ at $x_* - 0.1$. Since $\mu\ge-1$, $\Phi_t(x)\ge-1+(x+1)e^{-t}$, which exceeds $x_*+0.05$ at $t=0.15$. The trajectory has entered and left the open interval of class $b$. At that horizon $m_{ab}(q)\approx0.007713$ and even the global bound $c=0.1$ gives $D_{ab}\epsilon\approx0.003645$. The state enclosure certifies the terminal label $a$ in this example.

The shift is the exponential Euler step for the known linear part $-z$. It is also a relaxed mean-shift update for this field. Bounding a class margin by a Lipschitz constant times a state uncertainty is an established certification strategy. The proposed specialization composes the query-local posterior covariance enclosure with the cross-class diameter bound, then uses this test to decide which inputs need further integration. Scoring at the shift without the enclosure is an approximation control. Numerical and latency comparisons are still needed to measure the value of the certificate in a selective solver. [Hochbruck and Ostermann](https://publikationen.bibliothek.kit.edu/1000024914) · [Tsuzuku et al.](https://arxiv.org/abs/1802.04034)

These bounds are exact-arithmetic statements. The implementation uses positive comparison slack. Floating-point scores, covariance and integral evaluations are subject to roundoff. In particular, the direct subtraction for $Q_\lambda$ can lose relative accuracy at short times, where the integral starts as $t^2/2+(\lambda-1)t^3/6+O(t^4)$. When working with very small covariance caps, keep $c$ before the subtraction $\ell=c-1$. Large shifts, ill-scaled features and very small bandwidths may amplify numerical errors. Treat numerical certificates as guarded evaluations of the analytic conditions and use independent integration to check disagreements. A machine-checked enclosure would require explicit numerical error bounds. [Follow-up implementation](flow.py)

## 3. Boundary flux and the direction of a population accuracy change

For this section let test features have a continuous density $\rho$ of compact support. Work on a compact neighborhood containing that support and its short-time flow. Assume every top pair tie is regular on the neighborhood, including ties at junctions. At each top triple tie, assume the two differences $g_b-g_a$ and $g_c-g_a$ have linearly independent gradients. Assume there is no top triple tie in dimension one. These conditions imply that simultaneous two-margin bands of width $O(h)$ near top triple ties have volume $O(h^2)$, by a compact cover of submersion charts. The open strict top interface is

$$
\Sigma_{ab}=\{z:g_a(z)=g_b(z)>\max_{c\notin\{a,b\}}g_c(z)\}.
$$

**Proposition 3.** With $[s]_+=\max(s,0)$,

$$
\Pr\{C(X)=a,\ C(\Phi_h X)=b\}
= h\int_{\Sigma_{ab}}\rho(z)[F(z)\cdot n_{ab}(z)]_+\,d\mathcal H^{d-1}(z)+o(h) .
$$

To see this, the bounded score changes are $O(h)$ on the compact neighborhood. A switch away from the two-margin junction bands can involve only the two locally leading classes. In a signed-distance chart, a point initially a distance $r=O(h)$ on the $a$ side has distance $-r+h F(z)\cdot n_{ab}(z)+O(h^2)$ after the step, where $z$ is the normal projection onto the tie. The outgoing strip therefore has width $h[F(z)\cdot n_{ab}(z)]_++O(h^2)$. Integrate the density using the chart Jacobian $1+O(r)$. The tie faces have locally finite surface area under regularity. One can first use charts away from junctions and then let their exclusion shrink. The two-margin bands themselves contribute $O(h^2)$ probability because the density is bounded. The incoming $b$ to $a$ term uses the negative part of the normal velocity.

If each true class $y$ has a continuous compactly supported conditional density $p_y$ satisfying the same conditions, apply the transition law to each conditional distribution. For balanced accuracy $\mathrm{BA}(h)=C_{\mathrm{num}}^{-1}\sum_y\Pr(C(\Phi_h X)=y\mid Y=y)$, where $C_{\mathrm{num}}$ is the number of classes,

$$
\mathrm{BA}'(0+)=\frac{1}{C_{\mathrm{num}}}\sum_{a<b}
\int_{\Sigma_{ab}}(p_b-p_a)(F\cdot n_{ab})\,d\mathcal H^{d-1}.
$$

The sign depends on the true class densities on the tie, which core-only fitting generally does not identify. In particular, the derivative vanishes if the two densities agree on every model tie, as on regular boundaries of a balanced Bayes readout. If a density jumps there, the directed transition law instead uses its trace on the **originating** side, when those traces and the corresponding local integrability conditions hold. Atomic test sets, singular feature manifolds and nontransverse or coincident ties require different arguments. In particular, a finite set of queries away from ties has no switches for a sufficiently small horizon.

For inertial dynamics with $v_0=0$ the position is $x+\frac12 h^2F(x)+O(h^3)$ on compact sets. The corresponding directed leading term, under the same regularity, is $\frac12 h^2$ times the flux integral. The first-order term in physical time is zero. This expansion gives no destination guarantee for the inertial flow.

## 4. Second-order risk with Bayes-optimal initial scores

Let $p_c$ be weighted joint densities, normalized by $\sum_c\int p_c=1$. For ordinary misclassification risk these are the joint class densities with their population priors. For balanced error they are the class-conditional densities divided by the number of classes. Write

$$
R(t)=1-\int p_{C(\Phi_t x)}(x)\,dx.
$$

Suppose the initial decision $C(x)$ maximizes $p_c(x)$ on the test distribution. Require the score-face and triple-tie regularity of Section 3 and continuously differentiable $p_c$ with compact support. Sufficient conditions for removing compact support are given below.

**Proposition 4.** Under these conditions the second-order excess risk is

$$
R(t)-R(0)=\frac{t^2}{2}
\sum_{a<b}\int_{\Sigma_{ab}}
\|\nabla(p_b-p_a)(z)\|\,(F(z)\cdot n_{ab}(z))^2
\,d\mathcal H^{d-1}(z)
+o(t^2),\qquad t\downarrow0 .
$$

At a strict top interface, continuity and Bayes ordering on the two sides give $p_b-p_a=0$ on the face. Its tangential derivative is zero there. The normal derivative $\kappa_{ab}=\partial_{n_{ab}}(p_b-p_a)$ is nonnegative because $n_{ab}$ points toward $b$, so it equals $\|\nabla(p_b-p_a)\|$. A zero derivative gives a zero contribution.

Fix a normal chart away from junctions and write $\nu=F\cdot n_{ab}$ on the face. The switched layer has thickness $t|\nu|+O(t^2)$ by the signed-distance expansion in Section 3. If $\nu>0$ it originates on the $a$ side, with regret $p_a-p_b$. If $\nu<0$ it originates on the $b$ side, with regret $p_b-p_a$. On either side at normal distance $r$, the regret is $\kappa_{ab}r+o(r)$. Integrating in $r$ against the normal-coordinate Jacobian $1+O(r)$ yields $\kappa_{ab}t^2\nu^2/2+o(t^2)$ per unit surface area. If $\nu=0$ the shifted layer contributes $o(t^2)$. Uniform smoothness on a compact set and finite local surface area justify integration of the expansion.

The two-margin junction bands have volume $O(t^2)$. If a point switches from $a$ to $b$, ordering gives $p_a(x)\ge p_b(x)$ at its initial location and $p_a(\Phi_t x)\le p_b(\Phi_t x)$ at its terminal location. Lipschitz continuity of the gap and $O(t)$ displacement bound the regret per switched point by $O(t)$. The junction bands thus contribute $O(t^3)$, leaving the strict two-class face integrals in the formula. In the remaining face charts one may first exclude fixed junction neighborhoods and then shrink them. The strip regret is $O(t^2)$ per unit face area, and the relative area of the excluded junction neighborhoods tends to zero by transversality.

For unbounded support, it is enough that the normalized regret outside large compact sets vanishes uniformly as $t\downarrow0$ and the corresponding surface integrals converge. For example, suppose the field has at most linear growth, the densities and their gradients have Gaussian upper tails, and the volume of the $\varepsilon$-neighborhood of the top tie set, including junctions, in each unit radial shell $j\le\|z\|<j+1$ is bounded by $C(1+j)^m\varepsilon$ for all sufficiently small $\varepsilon$ with fixed constants. Linear growth gives an $O(t(1+j))$ displacement in that shell. Switching entails reaching a top tie along the path, so its initial point lies in the associated neighborhood. The Lipschitz gap argument with the local Gaussian gradient bound then bounds the shell regret by $t^2$ times a polynomial in $j$ times a Gaussian tail. Shells where $t(1+j)$ exceeds the small-width bound are themselves beyond distance of order $1/t$, so the density tail gives $o(t^2)$. A binary affine tie satisfies the shell geometry bound. Unbounded multimodal ties require a separate geometry or integrability argument.

### The coefficient for a Gaussian-mixture Bayes oracle

If the weighted joint densities satisfy $p_c(z)=A(z)S_c(z)$ with a common positive smooth factor $A$, their ordering is exactly the score ordering. For normalized Gaussian components of common precision this includes a single common normalization factor, with the model weights chosen to match the weighted component masses. With different component precisions, normalized Gaussian densities require absorbing the determinant factors into the effective prototype weights. On $\Sigma_{ab}$, $p_a=p_b=p_*$ and the derivative of the common factor cancels. Since $\nabla S_c=S_c v_c/\beta$, the coefficient in Proposition 4 becomes

$$
\lim_{t\downarrow0}\frac{R(t)-R(0)}{t^2}
=\frac{1}{2\beta}
\sum_{a<b}\int_{\Sigma_{ab}} p_*(z)
\frac{\big((v_b-v_a)\cdot F\big)^2}{\|v_b-v_a\|}
\,d\mathcal H^{d-1}(z).
$$

This quantifies the initial penalty for changing a true Bayes decision. It can be positive even when the two directed fluxes balance or the first-order risk derivative is zero. A field tangent to every regular tie gives a zero coefficient. A fitted core-only score generally lacks Bayes ordering on the external distribution. In that regime the signed first-order comparison remains relevant and the second-order sign above does not establish loss.

### Pointwise comparison for arbitrary mixtures of test measures

Take weighted class measures $\lambda_c\mu_c$ and marginal $\mu=\sum_c\lambda_c\mu_c$. This construction permits continuous, atomic and singular components. Define $\eta_c=d(\lambda_c\mu_c)/d\mu$. If $C(x)$ maximizes $\eta_c(x)$ $\mu$-almost everywhere, then for any horizon and any measurable displacement giving a classifier $\widetilde C$,

$$
R(\widetilde C)-R(C)
=\int\big(\max_c\eta_c(x)-\eta_{\widetilde C(x)}(x)\big)\,d\mu(x)\ \ge0.
$$

Conditioning on the original feature proves the identity. Regret is strictly positive exactly when a non-Bayes class is chosen on a set of positive marginal measure. In particular, switching between tied Bayes classes contributes zero. This argument has no smoothness or geometry requirement. The quadratic rate uses the density and regularity assumptions separately and need not apply to the atomic or singular components.

The pointwise inequality is the classical excess-risk identity. The smooth expansion is a second variation of a decision region at a stationary Bayes boundary. The application here supplies the Gaussian prototype coefficient through the same interface-normal velocity that governs directed switches, allowing a parameter-free prediction in a known-density oracle check. [Audibert and Tsybakov](https://arxiv.org/abs/0708.2321) · [Second variation in shape optimization](https://arxiv.org/abs/1410.2586)

## 5. Scope and relation to prior work

- The control is the **same score at time zero**. A label can switch inside the same field with all fitted parameters unchanged. One can falsify a proposed sufficient certificate against a known analytic path or a separate enclosed reference integration. A boundary derivative evaluated at one tie cannot replace the continuation bound.
- General tangency/invariance, Taylor remainder, Grönwall and normal transport arguments are classical. The specialization here is the exact margin identity, the precision alignment condition, and constants computed from the same prototypes used by both classifiers. Prior work is needed before claiming novelty for any component. [Nagumo](https://www.jstage.jst.go.jp/article/ppmsj1919/24/0/24_0_551/_article) · [Transport theorem](https://pmc.ncbi.nlm.nih.gov/articles/PMC4240036/)
- The unpatched field is a positive rescaling of the ascent direction $\nabla S$. With common isotropic precision $P=pI$, an Euler step of length $1/p$ is Gaussian mean shift on weighted prototypes. The finite-horizon class margin is a distinct quantity from finding density modes. [Comaniciu and Meer](https://pages.cs.wisc.edu/~dyer/ai-qual/comaniciu-pami02.pdf) · [Chacón](https://arxiv.org/abs/1408.1381)
- A stability patch keeps the static scores and changes the vector field inside its support. The chain rule and the general certificate or transport arguments still apply with bounds for the patched field. The responsibility-based speed and curvature bounds and the tie identity above apply directly only where the field is unpatched.
- The topological obstruction for a same-dimensional finite-time Neural ODE with a linear readout assumes a different terminal classifier. Its second-order extension also precedes this study. Here the final score can be nonlinear and the theorem is local to a specified score boundary. Neither annulus impossibility nor a general second-order advantage follows from these formulas. [Dupont et al.](https://arxiv.org/abs/1904.01681) · [Norcliffe et al.](https://arxiv.org/abs/2006.07220)

Test the direction and the guaranteed horizon before relating changes to accuracy. Analytic controls include an aligned equal-weight binary pair, a rotated anisotropic pair, an unequal-weight aligned pair, and the strict multiclass tie above. For held-out geometries, report the observed switches among certified and uncertified queries, the contribution of small margins and refinement, and the sign of net corrected versus damaged predictions independently of coverage. Compare small-horizon rates against flux quadrature only where the density and regularity assumptions are defensible.
