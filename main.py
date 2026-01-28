
import jax
import jax.numpy as jnp
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)

# ============================================================
# 1. Parameters (geometry, scales, phenomenology)
# ============================================================

@dataclass
class Params:
    # Geometry and baseline effective RS-like background
    k: float = 1.0
    L: float = 1.0
    rc: float = 12.0
    vUV: float = 0.1
    vIR: float = 0.01
    Ny: int = 3001
    kappa5_sq: float = 1.0

    # Grid shaping
    stretch: float = 0.35
    alpha: float = 4.0

    # Phenomenological IR-localized deformations of A'(y)
    eps_JT: float = 0.0
    eps_Sch: float = 0.0
    ir_window_center_frac: float = 0.95
    ir_window_width_frac: float = 0.02
    sch_sat: float = 1.0

    # UV counter-terms (phenomenological, not derived)
    delta_m2_UV: float = 0.0
    delta_lambda_UV: float = 0.0

    # Physical scales
    mH_bare: float = 125.0
    M5: float = 1.0e18

    # Numerical safety thresholds
    A_abs_max: float = 200.0
    Aprime_abs_max: float = 200.0


# ============================================================
# 0. Safe helpers (pure JAX)
# ============================================================

def safe_exp(x, low=-700.0, high=700.0):
    x_clipped = jnp.clip(x, low, high)
    return jnp.exp(x_clipped)

def safe_div(n, d, eps=1e-300):
    return n / (d + eps)

def pct(a, b):
    denom = jnp.where(jnp.abs(b) > 1e-30, b, jnp.array(1.0))
    return 100.0 * (a - b) / denom


# ============================================================
# 2. Grids
# ============================================================

def make_stretched_grid(p: Params):
    Ymax = jnp.pi * jnp.array(p.rc, dtype=jnp.float64)
    s = jnp.array(p.stretch, dtype=jnp.float64)
    xi = jnp.linspace(0.0, 1.0, p.Ny, dtype=jnp.float64)
    f = ((1.0 - s) * xi + s * (xi ** p.alpha)) / ((1.0 - s) + s)
    y = Ymax * f
    return y, Ymax

def make_uniform_grid(Ymax, Ny):
    return jnp.linspace(0.0, Ymax, Ny, dtype=jnp.float64)

def ir_window(y, Ymax, p: Params):
    y0 = p.ir_window_center_frac * Ymax
    w  = p.ir_window_width_frac * Ymax
    return 0.5 * (1.0 + jnp.tanh((y - y0) / w))


# ============================================================
# 3. Physics: effective RS-like superpotential system
# ============================================================

def W0(p: Params):
    # Constant effective superpotential (toy RS-like)
    return jnp.array(3.0 * p.k / p.kappa5_sq, dtype=jnp.float64)

def rhs_superpotential(p: Params, y, U, Ymax, c2):
    """
    Effective first-order system:
      dφ/dy = 2 c2 φ
      dA/dy = (κ5^2 / 3) [ W0 + c2 φ^2 ]
    This is a self-consistent toy model, not a full RS solution.
    """
    phi, A = U
    dphi = 2.0 * c2 * phi
    dA = (p.kappa5_sq / 3.0) * (W0(p) + c2 * (phi**2))
    return jnp.array([dphi, dA])


# ============================================================
# 4. Phenomenological deformations of A'(y)
# ============================================================

def rhs_superpotential_deformed(p: Params, y, U, Ymax, c2, vUV_eff):
    """
    Phenomenological deformation of the effective RS-like system:
      dφ/dy as in baseline
      dA/dy = A'_base + IR-localized deformations (eps_JT, eps_Sch)
    This is NOT derived from a superpotential; it is a controlled
    phenomenological experiment (off-shell background engineering).
    """
    phi, A = U
    dphi = 2.0 * c2 * phi

    # Baseline effective A'
    Aprime_base = (p.kappa5_sq / 3.0) * (W0(p) + c2 * (phi**2))

    # IR-localized window
    wIR = ir_window(y, Ymax, p)

    # Phenomenological JT-like shift
    dA_JT = p.eps_JT * wIR

    # Phenomenological "Schwarzian-like" saturation deformation
    sat = p.sch_sat
    dA_S = p.eps_Sch * wIR * (Aprime_base**2) / (1.0 + (Aprime_base**2) / (sat**2))

    dA = Aprime_base + dA_JT + dA_S
    return jnp.array([dphi, dA])


def uv_renormalized_params(p: Params, Ymax):
    """
    Phenomenological UV counter-terms: effective vUV and c2.
    Not a true renormalization calculation; just a parametrization.
    """
    c2_base = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)

    # Simple linear map from (delta_m2, delta_lambda) to (delta_c2, delta_vUV)
    a_m = jnp.array(0.5, dtype=jnp.float64)
    a_l = jnp.array(0.1, dtype=jnp.float64)
    b_m = jnp.array(0.25, dtype=jnp.float64)
    b_l = jnp.array(0.05, dtype=jnp.float64)

    delta_c2 = a_m * p.delta_m2_UV + a_l * p.delta_lambda_UV
    delta_vUV = b_m * p.delta_m2_UV + b_l * p.delta_lambda_UV

    vUV_eff = p.vUV * (1.0 + delta_vUV)
    c2_eff = c2_base + delta_c2
    return vUV_eff, c2_eff


# ============================================================
# 5. Integrator: fixed-step RK4 (pure JAX, stretched grid default)
# ============================================================

def integrate_fixed_rk4(fun, p: Params, y: jnp.ndarray, U0: jnp.ndarray, Ymax, c2):
    Ny = y.shape[0]
    h = (y[-1] - y[0]) / jnp.array(Ny - 1, dtype=jnp.float64)

    def step(Uc, i):
        yi = y[0] + i * h
        k1 = fun(p,   yi,           Uc,             Ymax, c2)
        k2 = fun(p,   yi + 0.5*h,   Uc + 0.5*h*k1,  Ymax, c2)
        k3 = fun(p,   yi + 0.5*h,   Uc + 0.5*h*k2,  Ymax, c2)
        k4 = fun(p,   yi + h,       Uc + h*k3,      Ymax, c2)
        Un = Uc + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return Un, Un

    _, states = jax.lax.scan(step, U0, jnp.arange(Ny-1))
    Y = jnp.vstack([U0, states])
    return Y


# ============================================================
# 6. Observables
# ============================================================

def redshift_outputs(A, p: Params):
    A_IR = A[-1]
    redshift = safe_exp(-A_IR)
    return {"A_IR": A_IR, "redshift": redshift}

def planck_mass_integral(A, y):
    integrand = safe_exp(2.0 * A)
    dx = jnp.diff(y)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    return jnp.sum(avg * dx)

def hierarchy_from_A(A: jnp.ndarray):
    logV = -4.0 * A
    V_eff = safe_exp(logV)
    dA = jnp.diff(A)
    eps_local = safe_exp(-4.0 * dA)
    eps_mean = jnp.mean(eps_local) if eps_local.size > 0 else jnp.nan
    return {"logV": logV, "V_eff": V_eff, "eps_local": eps_local, "eps_mean": eps_mean}


# ============================================================
# 7. Audits (physics + numerics)
# ============================================================

def audit_volume_ratio_pointwise(V_eff: jnp.ndarray, A: jnp.ndarray, tol: float = 1e-6):
    if V_eff.size < 2 or A.size < 2:
        return {"pass": jnp.array(False), "max_error": jnp.inf, "mean_error": jnp.inf}
    obs = V_eff[1:] / safe_div(V_eff[:-1], 1.0)
    dA  = jnp.diff(A)
    exp_ratio = safe_exp(-4.0 * dA)
    err_vec = jnp.abs(obs - exp_ratio)
    return {
        "max_error": jnp.max(err_vec),
        "mean_error": jnp.mean(err_vec),
        "pass": jnp.max(err_vec) < tol
    }

def audit_monotone_A(A: jnp.ndarray, tol: float = 0.0):
    V_eff = safe_exp(-4.0 * A)
    diffs = jnp.diff(V_eff)
    nonincreasing = jnp.all(diffs <= tol) if diffs.size else jnp.array(True)
    return {"nonincreasing": nonincreasing}

def audit_sign_Aprime(A: jnp.ndarray, tol: float = 1e-6):
    """
    Check that A'(y) has a consistent sign (mostly positive or mostly negative),
    allowing small numerical noise. Convention-agnostic.
    """
    dA = jnp.diff(A)
    if dA.size == 0:
        return {"Aprime_consistent_sign": jnp.array(True)}

    pos_frac = jnp.mean(dA >  tol)
    neg_frac = jnp.mean(dA < -tol)

    ok = jnp.logical_or(pos_frac > 0.99, neg_frac > 0.99)
    return {"Aprime_consistent_sign": ok}

def audit_phi_monotone(phi: jnp.ndarray, tol: float = 1e-5):
    dphi = jnp.diff(phi)
    ok = jnp.all(dphi <= tol) if dphi.size else jnp.array(True)
    return {"phi_nonincreasing": ok}

def audit_overflow_A_Aprime(A: jnp.ndarray, y: jnp.ndarray, p: Params):
    """
    Numerical safety audit: |A| and |A'| below user-defined thresholds.
    """
    A_abs_ok = jnp.max(jnp.abs(A)) <= p.A_abs_max
    dA = jnp.diff(A) / jnp.diff(y)
    Aprime_abs_ok = jnp.max(jnp.abs(dA)) <= p.Aprime_abs_max
    return {
        "A_abs_ok": A_abs_ok,
        "Aprime_abs_ok": Aprime_abs_ok,
    }


# ============================================================
# 8. Solvers: baseline effective vs phenomenological deformation
# ============================================================

def solve_superpotential_and_hierarchy(p: Params):
    # Stretched grid as default for physics
    y_str, Ymax = make_stretched_grid(p)

    c2 = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    U0 = jnp.array([p.vUV, 0.0], dtype=jnp.float64)

    Y = integrate_fixed_rk4(rhs_superpotential, p, y_str, U0, Ymax, c2)
    phi = Y[:, 0]
    A   = Y[:, 1]

    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "monotone_A":             audit_monotone_A(A),
        "sign_Aprime":            audit_sign_Aprime(A),
        "phi_monotone":           audit_phi_monotone(phi),
        "overflow":               audit_overflow_A_Aprime(A, y_str, p),
    }
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y_str)

    return {
        "y":        y_str,
        "phi":      phi,
        "A":        A,
        "c2":       c2,
        "audits":   audits,
        "redshift": redshift,
        "Mpl_eff":  Mpl_eff,
    }


def solve_superpotential_with_deformation(p: Params):
    """
    Phenomenological deformation of the effective RS-like background.
    All deviations from baseline are explicitly labeled as such.
    """
    y_str, Ymax = make_stretched_grid(p)

    vUV_eff, c2_eff = uv_renormalized_params(p, Ymax)
    U0 = jnp.array([vUV_eff, 0.0], dtype=jnp.float64)

    def fun_corr(p_, y_, U_, Ymax_, c2_):
        return rhs_superpotential_deformed(p_, y_, U_, Ymax_, c2_, vUV_eff)

    Y = integrate_fixed_rk4(fun_corr, p, y_str, U0, Ymax, c2_eff)
    phi = Y[:, 0]
    A   = Y[:, 1]

    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "monotone_A":             audit_monotone_A(A),
        "sign_Aprime":            audit_sign_Aprime(A),
        "phi_monotone":           audit_phi_monotone(phi),
        "overflow":               audit_overflow_A_Aprime(A, y_str, p),
    }
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y_str)

    # Effective IR Newton constant (phenomenological)
    G_IR = safe_div(1.0, Mpl_eff) * safe_exp(2.0 * A[-1])

    # Effective Higgs mass from redshift (pure kinematics)
    mH_eff = p.mH_bare * redshift["redshift"]

    return {
        "y":         y_str,
        "phi":       phi,
        "A":         A,
        "audits":    audits,
        "redshift":  redshift,
        "Mpl_eff":   Mpl_eff,
        "G_IR_eff":  G_IR,
        "mH_eff":    mH_eff,
    }


# ============================================================
# 9. Deformation comparison + “too-violent warp” audit
# ============================================================

def compare_deformed_vs_RS(p: Params,
                           max_abs_dA_IR=2.0,
                           max_abs_redshift_dev=90.0,
                           max_abs_dG_pct=200.0):
    """
    Compare deformed vs baseline and include a kinematic "too-violent warp"
    audit based on ΔA_IR, redshift deviation, and ΔG_IR.
    """
    base = solve_superpotential_and_hierarchy(p)
    corr = solve_superpotential_with_deformation(p)

    dA_IR = corr["redshift"]["A_IR"] - base["redshift"]["A_IR"]
    red_dev = pct(
        corr["redshift"]["redshift"],
        base["redshift"]["redshift"]
    )
    G_IR_base = safe_div(1.0, base["Mpl_eff"]) * safe_exp(2.0 * base["A"][-1])
    dG_pct = pct(
        corr["G_IR_eff"],
        G_IR_base
    )

    # Kinematic “too-violent warp” audit
    ok_dA   = jnp.abs(dA_IR) <= max_abs_dA_IR
    ok_red  = jnp.abs(red_dev) <= max_abs_redshift_dev
    ok_dG   = jnp.abs(dG_pct) <= max_abs_dG_pct
    warp_ok = ok_dA & ok_red & ok_dG

    audit_corr = corr["audits"]
    audit_base = base["audits"]

    audit_pass_corr = (
        audit_corr["monotone_A"]["nonincreasing"]
        & audit_corr["volume_ratio_pointwise"]["pass"]
        & audit_corr["sign_Aprime"]["Aprime_consistent_sign"]
        & audit_corr["phi_monotone"]["phi_nonincreasing"]
        & audit_corr["overflow"]["A_abs_ok"]
        & audit_corr["overflow"]["Aprime_abs_ok"]
        & warp_ok
    )

    audit_pass_base = (
        audit_base["monotone_A"]["nonincreasing"]
        & audit_base["volume_ratio_pointwise"]["pass"]
        & audit_base["sign_Aprime"]["Aprime_consistent_sign"]
        & audit_base["phi_monotone"]["phi_nonincreasing"]
        & audit_base["overflow"]["A_abs_ok"]
        & audit_base["overflow"]["Aprime_abs_ok"]
    )

    report = {
        "ΔA_IR": dA_IR,
        "redshift_pct_dev": red_dev,
        "ΔmH_eff_GeV": corr["mH_eff"] - (p.mH_bare * base["redshift"]["redshift"]),
        "ΔG_IR_eff_pct": dG_pct,
        "warp_ok": warp_ok,
        "audit_pass_corr": audit_pass_corr,
        "audit_pass_base": audit_pass_base,
    }
    return {"baseline": base, "deformed": corr, "report": report}


# ============================================================
# 10. Novelty metric (phenomenological diagnostic)
# ============================================================

def novelty_components_raw(p: Params, epsJT, epsSch, dm2=0.0, dl=0.0):
    """
    Phenomenological novelty metric: how much the deformed background
    deviates from effective RS-like baseline in a few observables.
    JAX-friendly.
    """
    p2 = Params(**{
        **p.__dict__,
        "eps_JT":          epsJT,
        "eps_Sch":         epsSch,
        "delta_m2_UV":     dm2,
        "delta_lambda_UV": dl,
    })
    rep = compare_deformed_vs_RS(p2)["report"]

    cA   = jnp.abs(rep["ΔA_IR"])
    cred = 0.1 * jnp.abs(rep["redshift_pct_dev"])
    cmH  = 0.01 * jnp.abs(rep["ΔmH_eff_GeV"])
    cG   = 0.1 * jnp.abs(rep["ΔG_IR_eff_pct"])

    total = cA + cred + cmH + cG
    return total, (cA, cred, cmH, cG), rep


def numerical_floors_from_convergence(p: Params, Ny_list=(1501, 3001, 6001)):
    """
    Estimate numerical floors for each novelty component via Ny-convergence
    at very small deformations (close to baseline).
    This is a driver-level routine (not jitted) because Ny_list is Python.
    """
    epsJT  = jnp.array(1e-6, dtype=jnp.float64)
    epsSch = jnp.array(2e-6, dtype=jnp.float64)

    comps_list = []
    for Ny in Ny_list:
        pN = Params(**{
            **p.__dict__,
            "Ny":      Ny,
            "eps_JT":  float(epsJT),
            "eps_Sch": float(epsSch),
        })
        _, comps, _ = novelty_components_raw(pN, epsJT, epsSch)
        cA, cred, cmH, cG = comps
        comps_list.append(jnp.array([cA, cred, cmH, cG], dtype=jnp.float64))

    comps_arr   = jnp.stack(comps_list, axis=0)
    mean_comps  = jnp.mean(comps_arr, axis=0, keepdims=True)
    floors      = jnp.max(jnp.abs(comps_arr - mean_comps), axis=0)
    floors      = floors + jnp.array([1e-9, 1e-9, 1e-12, 1e-9], dtype=jnp.float64)

    return {"floors": floors, "Ny_list": Ny_list, "samples": comps_arr}


def novelty_components_normalized(p: Params, epsJT, epsSch, floors,
                                  dm2=0.0, dl=0.0, norm="l1"):
    total, comps, rep = novelty_components_raw(p, epsJT, epsSch, dm2, dl)
    cA, cred, cmH, cG = comps
    fA, fred, fmH, fG = [jnp.array(x, dtype=jnp.float64) for x in floors]

    cA_n   = cA   / fA
    cred_n = cred / fred
    cmH_n  = cmH  / fmH
    cG_n   = cG   / fG

    if norm == "l2":
        total_n = jnp.sqrt(cA_n**2 + cred_n**2 + cmH_n**2 + cG_n**2)
    else:
        total_n = cA_n + cred_n + cmH_n + cG_n

    return total_n, (cA_n, cred_n, cmH_n, cG_n), rep, comps


# ============================================================
# 11. Gradient-based single trajectory in parameter space
# ============================================================

def gradient_ascent_normalized(
    p: Params,
    floors,
    init=(0.05, 0.10, 0.0, 0.0),
    lr=0.05,
    steps=20,
    bounds=((0.0, 0.5), (0.0, 0.8), (-0.2, 0.2), (-0.2, 0.2)),
    norm="l1",
):
    """
    Driver-level routine (Python loop) that internally uses JAX grad.
    """
    epsJT, epsSch, dm2, dl = [jnp.array(x, dtype=jnp.float64) for x in init]

    def metric(a, b, c, d):
        val_n, _, _, _ = novelty_components_normalized(p, a, b, floors, c, d, norm=norm)
        return val_n

    grad4 = jax.grad(lambda a, b, c, d: metric(a, b, c, d), argnums=(0, 1, 2, 3))

    traj = []

    for t in range(steps):
        g_tuple = grad4(epsJT, epsSch, dm2, dl)
        g_vec   = jnp.stack(g_tuple)
        g_norm  = jnp.linalg.norm(g_vec)
        stepvec = (lr / (1e-8 + g_norm)) * g_vec

        cand = jnp.array([epsJT, epsSch, dm2, dl]) + stepvec

        epsJT  = jnp.clip(cand[0], bounds[0][0], bounds[0][1])
        epsSch = jnp.clip(cand[1], bounds[1][0], bounds[1][1])
        dm2    = jnp.clip(cand[2], bounds[2][0], bounds[2][1])
        dl     = jnp.clip(cand[3], bounds[3][0], bounds[3][1])

        total_n, comps_n, rep, comps_raw = novelty_components_normalized(
            p, epsJT, epsSch, floors, dm2, dl, norm=norm
        )

        p2 = Params(**{
            **p.__dict__,
            "eps_JT":          float(epsJT),
            "eps_Sch":         float(epsSch),
            "delta_m2_UV":     float(dm2),
            "delta_lambda_UV": float(dl),
        })
        cmp2      = compare_deformed_vs_RS(p2)
        audits_ok = bool(cmp2["report"]["audit_pass_corr"])

        traj.append({
            "step":          int(t),
            "params":        (float(epsJT), float(epsSch), float(dm2), float(dl)),
            "novelty_norm":  float(total_n),
            "comps_norm":    tuple([float(x) for x in comps_n]),
            "comps_raw":     tuple([float(x) for x in comps_raw]),
            "ΔA_IR":         float(rep["ΔA_IR"]),
            "redshift_dev":  float(rep["redshift_pct_dev"]),
            "audits_ok":     audits_ok,
        })

    return traj


# ============================================================
# 12. Background-space cartography
# ============================================================

def classify_point_jax(p_base: Params, epsJT, epsSch, dm2=0.0, dl=0.0):
    """
    JAX-friendly classifier for a single point in deformation space.
    Returns:
      audits_ok, ΔA_IR, redshift_dev, ΔG_IR_eff_pct
    all as JAX arrays.
    """
    p2 = Params(**{
        **p_base.__dict__,
        "eps_JT":          epsJT,
        "eps_Sch":         epsSch,
        "delta_m2_UV":     dm2,
        "delta_lambda_UV": dl,
    })
    cmp = compare_deformed_vs_RS(p2)
    rep = cmp["report"]
    return (
        rep["audit_pass_corr"],
        rep["ΔA_IR"],
        rep["redshift_pct_dev"],
        rep["ΔG_IR_eff_pct"],
    )


def scan_background_space(
    p_base: Params,
    jt_grid=jnp.linspace(0.0, 0.35, 21),
    sch_grid=jnp.linspace(0.0, 0.6, 21),
    dm2=0.0,
    dl=0.0,
):
    """
    Background-space cartography:
      scan (epsJT, epsSch) and classify each point as allowed/forbidden
      based on audits (including too-violent warp), and record key deviations.
    """
    def one(ejt, esch):
        return classify_point_jax(p_base, ejt, esch, dm2, dl)

    audits_ok, dA, dred, dG = jax.vmap(
        lambda ejt: jax.vmap(lambda esch: one(ejt, esch))(sch_grid)
    )(jt_grid)

    return {
        "JT":            jt_grid,
        "SCH":           sch_grid,
        "audits_ok":     audits_ok,
        "ΔA_IR":         dA,
        "redshift_dev":  dred,
        "ΔG_IR_eff_pct": dG,
    }


# ============================================================
# 12b. Lightweight sweep over (epsJT, epsSch)
# ============================================================

def sweep_epsJT_epsSch(
    p: Params,
    jt_grid=jnp.linspace(0.0, 0.35, 21),
    sch_grid=jnp.linspace(0.0, 0.6, 21),
    dm2=0.0,
    dl=0.0,
):
    """
    Lightweight sweep of raw novelty metric over (epsJT, epsSch).
    Struktur dikembalikan sama persis seperti yang diharapkan run_engine().
    """
    def one(ejt, esch):
        total, comps, _ = novelty_components_raw(p, ejt, esch, dm2, dl)
        return total, comps

    totals, comps = jax.vmap(
        lambda ejt: jax.vmap(lambda esch: one(ejt, esch))(sch_grid)
    )(jt_grid)

    # comps adalah tuple 4 elemen, masing-masing array 2D
    cA   = comps[0]
    cred = comps[1]
    cmH  = comps[2]
    cG   = comps[3]

    return {
        "JT":   jt_grid,
        "SCH":  sch_grid,
        "vals": totals,
        "cA":   cA,
        "cred": cred,
        "cmH":  cmH,
        "cG":   cG,
    }


# ============================================================
# 13. Runner
# ============================================================

def run_engine():
    p = Params()

    # Baseline effective solution
    base = solve_superpotential_and_hierarchy(p)
    print("Baseline A_IR:", float(base["redshift"]["A_IR"]))
    print("Baseline redshift:", float(base["redshift"]["redshift"]))
    base_pass = bool(
        base["audits"]["monotone_A"]["nonincreasing"]
        & base["audits"]["volume_ratio_pointwise"]["pass"]
        & base["audits"]["sign_Aprime"]["Aprime_consistent_sign"]
        & base["audits"]["phi_monotone"]["phi_nonincreasing"]
        & base["audits"]["overflow"]["A_abs_ok"]
        & base["audits"]["overflow"]["Aprime_abs_ok"]
    )
    print("Baseline audits pass:", base_pass)

    # Numerical floors from Ny-convergence
    floors_info = numerical_floors_from_convergence(p, Ny_list=(1501, 3001, 6001))
    floors = floors_info["floors"]
    print("Numerical floors [cA, cred, cmH, cG]:", [float(x) for x in floors])

    # Lightweight sweep for context
    sweep = sweep_epsJT_epsSch(
        p,
        jt_grid=jnp.linspace(0.0, 0.35, 21),
        sch_grid=jnp.linspace(0.0, 0.6, 21),
    )
    total = sweep["vals"]
    idx = jnp.unravel_index(jnp.argmax(total), total.shape)
    jt_star  = float(sweep["JT"][idx[0]])
    sch_star = float(sweep["SCH"][idx[1]])
    print(
        "Sweep peak (raw novelty): epsJT≈%.4f, epsSch≈%.4f, novelty≈%.6f"
        % (jt_star, sch_star, float(total[idx]))
    )

    # Single normalized trajectory
    traj = gradient_ascent_normalized(
        p,
        floors=floors,
        init=(0.05, 0.10, 0.0, 0.0),
        lr=0.05,
        steps=20,
        norm="l1",
    )
    print("Ascent normalized last step:", traj[-1])

    # Background-space cartography
    carto = scan_background_space(
        p,
        jt_grid=jnp.linspace(0.0, 0.35, 21),
        sch_grid=jnp.linspace(0.0, 0.6, 21),
    )
    allowed_frac = float(jnp.mean(carto["audits_ok"]))
    print("Background-space cartography: allowed fraction =", allowed_frac)

    # Academic summary: component hierarchy at end vs floors
    end     = traj[-1]
    comps_n = end["comps_norm"]
    comps_r = end["comps_raw"]
    print("End normalized components [cA_n, cred_n, cmH_n, cG_n]:", comps_n)
    print("End raw components [cA, cred, cmH, cG]:", comps_r)
    print("Audits at end:", end["audits_ok"])

    return {
        "baseline":    base,
        "floors":      floors_info,
        "sweep":       sweep,
        "traj":        traj,
        "cartography": carto,
    }


if __name__ == "__main__":
    out = run_engine()
