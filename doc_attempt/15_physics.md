# Physics cookbook: implementing PDEs with Samurai

This page shows how to assemble common physics (CFD and reaction–diffusion) using Samurai’s user-facing FV operators and boundary conditions. It focuses on the high-level builders available in `samurai/schemes/fv.hpp` and the patterns used in the demos under `demos/FiniteVolume/`.

Include the FV module:

```cpp
#include <samurai/schemes/fv.hpp>
```

Typical setup shared by examples below:

- Create a mesh (uniform or multiresolution), then fields with `make_scalar_field` / `make_vector_field`.
- Attach boundary conditions with `make_bc<Dirichlet<stencil>>()` or `make_bc<Neumann<stencil>>()` (constant or function value).
- Build operators via `make_diffusion_order2`, `make_convection_upwind/weno5`, `make_gradient_order2`, `make_divergence_order2`, etc.
- Advance in time explicitly (Euler, TVD-RK3) or implicitly with PETSc using `samurai::petsc::solve`.

Notes:

- The diffusion builder implements the operator −∇·(K∇u). To get +Δu with coefficient D, use `-diff(u)` or equivalently subtract `diff(u)` in the RHS, as shown in the demos.
- FV schemes update ghosts automatically on application. If you use low-level stencil helpers (e.g. `upwind(a, u)` from `stencil_field.hpp`), call `update_ghost_mr(u)` yourself.

## Linear advection (scalar, prescribed velocity)

Use upwind or WENO5 discrete convection. Velocity can be constant or a velocity field.

```cpp
using Mesh = samurai::MRMesh<samurai::MRConfig<2>>;
Mesh mesh(/* box, min_level, max_level */);

auto u  = samurai::make_scalar_field<double>("u", mesh);
auto un = samurai::make_scalar_field<double>("un", mesh);

// Boundary conditions (example: homogeneous Dirichlet)
samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0);

// 1) Constant velocity
samurai::VelocityVector<2> a = {1.0, 1.0};
auto conv_cst = samurai::make_convection_upwind<decltype(u)>(a);   // or make_convection_weno5

// 2) Velocity field (size must be the space dimension)
auto v = samurai::make_vector_field<double, 2>("v", mesh);
// ... set v on reference cells (update ghosts if you modify v during the run)
auto conv_var = samurai::make_convection_weno5<decltype(u)>(v);

double dx  = mesh.cell_length(mesh.max_level());
double cfl = 0.5;
double dt  = cfl * dx; // see demos for variants

un = u - dt * conv_cst(u); // or conv_var(u)
```

Time step hints (as used in demos): `dt = cfl * mesh.cell_length(max_level)` for pure advection; adjust with velocity magnitude if needed.

## Nonlinear convection (Burgers)

For u_t + ∇·f(u) = 0 with f(u) = ½ u² in 1D (and its vector generalizations), use the nonlinear convection builders:

```cpp
auto u    = samurai::make_scalar_field<double>("u", mesh);
auto u1   = samurai::make_scalar_field<double>("u1", mesh);
auto u2   = samurai::make_scalar_field<double>("u2", mesh);
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

// 1D needs the 1/2 factor in front of the convective operator
double cst = (decltype(u)::mesh_t::dim == 1) ? 0.5 : 1.0;
auto conv  = cst * samurai::make_convection_weno5<decltype(u)>(); // or upwind

double dx  = mesh.cell_length(mesh.max_level());
double cfl = 0.95;
double dt  = cfl * dx / std::pow(2.0, (int)decltype(u)::mesh_t::dim);

// TVD-RK3 (SSPRK3)
u1   = u - dt * conv(u);
u2   = 0.75 * u + 0.25 * (u1 - dt * conv(u1));
unp1 = (1.0/3.0) * u + (2.0/3.0) * (u2 - dt * conv(u2));
```

See `demos/FiniteVolume/burgers.cpp` for a complete example and BC options (Dirichlet with a function, or constants).

## Diffusion (heat equation)

Build the second-order diffusion operator. Remember: it returns −∇·(K∇u).

```cpp
auto u    = samurai::make_scalar_field<double>("u", mesh);
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

// Homogeneous Neumann (zero-flux)
samurai::make_bc<samurai::Neumann<1>>(u, 0.0);
samurai::make_bc<samurai::Neumann<1>>(unp1, 0.0);

double K = 1.0; // scalar coefficient; see below for tensors
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);

// Explicit Euler (demo formula)
double dx  = mesh.cell_length(mesh.max_level());
double cfl = 0.95;
double dt  = cfl * (dx*dx) / (std::pow(2.0, (int)decltype(u)::mesh_t::dim) * K);
unp1 = u - dt * diff(u); // minus sign → +KΔu

// Implicit Backward Euler with PETSc
auto id = samurai::make_identity<decltype(u)>();
samurai::petsc::solve(id + dt * diff, unp1, u); // solves [I + dt*Diff](unp1) = u
```

Variants supported by the builder:

- Diagonal anisotropy per direction: `samurai::DiffCoeff<dim> Kdir; Kdir(0)=..., Kdir(1)=...; make_diffusion_order2<Field>(Kdir);`
- Heterogeneous: a field with value-type `DiffCoeff<dim>` filled on cells, then `make_diffusion_order2<Field>(K_field)`.
- Vector fields: `make_multi_diffusion_order2<Field>(K_comp)` accepts one coefficient per component.

See `demos/FiniteVolume/heat.cpp` for explicit/implicit usage and error checks against an exact solution.

## Convection–diffusion (passive scalar)

Combine builders directly. For u_t + a·∇u = ∇·(K∇u):

```cpp
auto conv = samurai::make_convection_upwind<decltype(u)>(a); // or velocity field + WENO5
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);

// Explicit Euler
unp1 = u - dt * conv(u) - dt * diff(u); // "- diff(u)" gives +∇·(K∇u)

// Semi-implicit example (explicit convection, implicit diffusion):
auto id  = samurai::make_identity<decltype(u)>();
auto rhs = u - dt * conv(u);
samurai::petsc::solve(id + dt * diff, unp1, rhs);
```

Time integrators: for advection-dominated problems, use TVD-RK schemes (see Burgers and Gray–Scott demos). For diffusion-dominated problems, implicit diffusion is often preferred.

## Reaction–diffusion systems (Gray–Scott, Nagumo)

Reactions are local (cell-based). Use a local cell-based scheme and set the reaction function; combine with diffusion.

```cpp
// Two-component field (U, V)
static constexpr std::size_t n_comp = 2;
auto uv   = samurai::make_vector_field<double, n_comp>("uv", mesh);
auto uv1  = samurai::make_vector_field<double, n_comp>("uv1", mesh);
auto uv2  = samurai::make_vector_field<double, n_comp>("uv2", mesh);
auto unp1 = samurai::make_vector_field<double, n_comp>("unp1", mesh);
uv1.copy_bc_from(uv); uv2.copy_bc_from(uv); unp1.copy_bc_from(uv);

// Zero-flux Neumann on both components
samurai::make_bc<samurai::Neumann<1>>(uv);

// Component-wise diffusion (Du, Dv)
samurai::DiffCoeff<n_comp> Kc; Kc(0)=2e-5; Kc(1)=1e-5;
auto diff = samurai::make_multi_diffusion_order2<decltype(uv)>(Kc);

// Local reaction: Gray–Scott
using Rcfg  = samurai::LocalCellSchemeConfig<samurai::SchemeType::NonLinear, decltype(uv), decltype(uv)>;
auto react  = samurai::make_cell_based_scheme<Rcfg>();
double F=0.04, k=0.06;
react.set_name("Reaction");
react.set_scheme_function([F,k](const auto& cell, const auto& field) -> samurai::SchemeValue<Rcfg>
{
    auto w = field[cell];
    double U = w[0], V = w[1];
    samurai::SchemeValue<Rcfg> rhs;
    rhs[0] = -U*V*V + F*(1.0 - U);
    rhs[1] =  U*V*V - (F + k)*V;
    return rhs;
});

auto rhs = [&](auto& f) { return react(f) - diff(f); }; // "- diff" → +Δ on the physics

// TVD-RK3
uv1  = uv + dt * rhs(uv);
uv2  = 0.75 * uv + 0.25 * (uv1  + dt * rhs(uv1));
unp1 = (1.0/3.0) * uv + (2.0/3.0) * (uv2 + dt * rhs(uv2));
```

For a single-component Nagumo/Fisher–KPP model, define `react.set_scheme_function` with the scalar formula and optionally `set_jacobian_function` for implicit solves (see `demos/FiniteVolume/nagumo.cpp`).

## Incompressible Stokes (block operator + PETSc)

Build the saddle-point operator with second-order Diff/Grad/Div blocks, set BC, and solve with PETSc. The stationary operator reads

```cpp
auto velocity = samurai::make_vector_field<double, 2>("velocity", mesh);
auto pressure = samurai::make_scalar_field<double>("pressure", mesh);

// BC examples: Dirichlet on velocity, Neumann on pressure
samurai::make_bc<samurai::Dirichlet<1>>(velocity, [](const auto&, const auto&, const auto& X){
    return samurai::Array<double, 2, false>{ /* vx(X), vy(X) */ };
});
samurai::make_bc<samurai::Neumann<1>>(pressure, [](const auto&, const auto&, const auto& X){
    return 0.0; // normal gradient
});

auto diff    = samurai::make_diffusion_order2<decltype(velocity)>();
auto grad    = samurai::make_gradient_order2<decltype(pressure)>();
auto div     = samurai::make_divergence_order2<decltype(velocity)>();
auto zero_op = samurai::make_zero_operator<decltype(pressure)>();

//            | Diff  Grad |
//    A(u,p)= | -Div   0   |
auto stokes  = samurai::make_block_operator<2,2>(diff, grad,
                                                 -div, zero_op);

// Solve A [u, p]^T = [f, 0]^T with PETSc
auto f = samurai::make_vector_field<double, 2>("f", mesh);
auto z = samurai::make_scalar_field<double>("z", mesh, 0.0);
auto solver = samurai::petsc::make_solver<true /* monolithic */>(stokes);
solver.set_unknowns(velocity, pressure);
solver.solve(f, z);
```

For transient Stokes, use the backward Euler block `(I + dt*Diff, dt*Grad; -Div, 0)` as in `demos/FiniteVolume/stokes_2d.cpp` and resolve each step.

## Boundary conditions (quick reference)

- Dirichlet with constant value(s): `make_bc<Dirichlet<stencil>>(field, c0[, c1, c2])` where the number of constants matches the field size.
- Dirichlet with function: the callback receives `(direction, cell, face_center_coords)` and must return a value with the field’s shape.
- Neumann (normal flux) similarly: `make_bc<Neumann<stencil>>(...)`. Omitting the value yields zero-flux.

## Multiresolution adaptation (optional)

All demos on adaptive meshes follow the same skeleton:

```cpp
auto MRadaptation = samurai::make_MRAdapt(field);
auto mra_conf     = samurai::mra_config() /* .epsilon(...).regularity(...) */;
MRadaptation(mra_conf);
// resize intermediate fields after adaptation
```

## Operator builders (at a glance)

- Convection (linear, constant velocity): `make_convection_upwind<Field>(a)`, `make_convection_weno5<Field>(a)`
- Convection (linear, velocity field): `make_convection_upwind<Field>(v)`, `make_convection_weno5<Field>(v)`
- Convection (nonlinear, e.g. Burgers or u·∇u): `make_convection_upwind<Field>()`, `make_convection_weno5<Field>()`
- Diffusion (2nd order): `make_diffusion_order2<Field>(k|K|K_field)`, `make_multi_diffusion_order2<Field>(K_per_component)`, `make_laplacian_order2<Field>()`
- Gradient/Divergence (2nd order): `make_gradient_order2<ScalarField>()`, `make_divergence_order2<VectorField>()`
- Identity and zero: `make_identity<Field>()`, `make_zero_operator<Field>()`
- Local cell-based scheme (reactions): `LocalCellSchemeConfig` + `make_cell_based_scheme<cfg>()`

For more details on FV stencils and flux definitions, see `doc_attempt/07_fv_operators_and_schemes.md` and the examples in `demos/FiniteVolume/`.

## Additional explicit-only physics (not in demos)

The following explicit-only examples illustrate how to assemble systems not covered by the bundled demos, using only user-facing FV builders and non-linear flux definitions.

### Shallow Water (Saint-Venant) in 2D (explicit Rusanov flux)

Physics

- Conservative form for q = [h, hu, hv]^T with gravity g:
  - ∂_t h   + ∂_x(hu)           + ∂_y(hv)           = 0
  - ∂_t hu  + ∂_x(hu^2 + ½ g h^2) + ∂_y(hu v)        = 0
  - ∂_t hv  + ∂_x(hu v)           + ∂_y(hv^2 + ½ g h^2) = 0
- Explicit FV with Rusanov (local Lax–Friedrichs) flux; CFL uses max(|u_n| + √(g h)).

Implementation sketch

```cpp
static constexpr std::size_t dim = 2;
static constexpr std::size_t n_comp = 3; // [h, hu, hv]
auto q    = samurai::make_vector_field<double, n_comp>("q", mesh);
auto q_np = samurai::make_vector_field<double, n_comp>("q_np", mesh);
q_np.copy_bc_from(q); // e.g., transmissive or wall BCs implemented via functions

double g = 9.81;

using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear,
                                2,            // 2-point stencil (L/R)
                                decltype(q),  // output_field_t
                                decltype(q)>; // input_field_t

samurai::FluxDefinition<cfg> sw;

auto flux_x = [g](auto U) {
    double h = U(0); double hu = U(1); double hv = U(2);
    double u = (h > 0) ? hu / h : 0.0;
    samurai::FluxValue<cfg> F; // size 3
    F(0) = hu;
    F(1) = hu * u + 0.5 * g * h * h;
    F(2) = hv * u;
    return F;
};
auto flux_y = [g](auto U) {
    double h = U(0); double hu = U(1); double hv = U(2);
    double v = (h > 0) ? hv / h : 0.0;
    samurai::FluxValue<cfg> G;
    G(0) = hv;
    G(1) = hu * v;
    G(2) = hv * v + 0.5 * g * h * h;
    return G;
};

// X-direction
sw[0].cons_flux_function = [=](samurai::FluxValue<cfg>& flux,
                                const samurai::StencilData<cfg>&,
                                const samurai::StencilValues<cfg>& U) {
    constexpr std::size_t L = 0, R = 1;
    auto UL = U[L]; auto UR = U[R];
    double hL = UL(0), hR = UR(0);
    double uL = (hL > 0) ? UL(1)/hL : 0.0;
    double uR = (hR > 0) ? UR(1)/hR : 0.0;
    double cL = std::sqrt(g * std::max(hL, 0.0));
    double cR = std::sqrt(g * std::max(hR, 0.0));
    double a  = std::max(std::abs(uL) + cL, std::abs(uR) + cR);
    flux = 0.5 * (flux_x(UL) + flux_x(UR)) - 0.5 * a * (UR - UL);
};

// Y-direction
sw[1].cons_flux_function = [=](samurai::FluxValue<cfg>& flux,
                                const samurai::StencilData<cfg>&,
                                const samurai::StencilValues<cfg>& U) {
    constexpr std::size_t L = 0, R = 1;
    auto UL = U[L]; auto UR = U[R];
    double hL = UL(0), hR = UR(0);
    double vL = (hL > 0) ? UL(2)/hL : 0.0;
    double vR = (hR > 0) ? UR(2)/hR : 0.0;
    double cL = std::sqrt(g * std::max(hL, 0.0));
    double cR = std::sqrt(g * std::max(hR, 0.0));
    double a  = std::max(std::abs(vL) + cL, std::abs(vR) + cR);
    flux = 0.5 * (flux_y(UL) + flux_y(UR)) - 0.5 * a * (UR - UL);
};

auto sw_op = samurai::make_flux_based_scheme(sw);

// Explicit update (e.g., TVD-RK3 or Euler)
q_np = q - dt * sw_op(q);
```

Pick a CFL `dt` based on the maximum eigenvalue over the mesh: `dt = cfl * dx / max(|u|+√(g h))` with dimensional extension. Handle dry states by clamping h ≥ 0 and choosing suitable boundary conditions.

### Compressible Euler in 2D (explicit Rusanov flux) — complex

Physics

- State Q = [ρ, ρu, ρv, E]^T; pressure p = (γ − 1)(E − ½ ρ(u²+v²)).
- Fluxes F_x(Q), F_y(Q) are the standard Euler fluxes; Rusanov flux with α = |u_n| + a, a = √(γ p / ρ).

Implementation sketch

```cpp
static constexpr std::size_t dim = 2;
static constexpr std::size_t n_comp = 4; // [rho, rho_u, rho_v, E]
auto Q    = samurai::make_vector_field<double, n_comp>("Q", mesh);
auto Qnp1 = samurai::make_vector_field<double, n_comp>("Qnp1", mesh);
Qnp1.copy_bc_from(Q);

double gamma_gas = 1.4;

using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, decltype(Q), decltype(Q)>;
samurai::FluxDefinition<cfg> euler;

auto px = [gamma_gas](auto U) {
    double r = U(0), ru = U(1), rv = U(2), E = U(3);
    double u = ru / r, v = rv / r;
    double p = (gamma_gas - 1.0) * (E - 0.5 * r * (u*u + v*v));
    samurai::FluxValue<cfg> Fx;
    Fx(0) = ru;
    Fx(1) = ru * u + p;
    Fx(2) = rv * u;
    Fx(3) = (E + p) * u;
    return Fx;
};
auto py = [gamma_gas](auto U) {
    double r = U(0), ru = U(1), rv = U(2), E = U(3);
    double u = ru / r, v = rv / r;
    double p = (gamma_gas - 1.0) * (E - 0.5 * r * (u*u + v*v));
    samurai::FluxValue<cfg> Fy;
    Fy(0) = rv;
    Fy(1) = ru * v;
    Fy(2) = rv * v + p;
    Fy(3) = (E + p) * v;
    return Fy;
};
auto wavespeed_x = [gamma_gas](auto U) {
    double r = U(0), ru = U(1), rv = U(2), E = U(3);
    double u = ru / r, v = rv / r;
    double p = std::max((gamma_gas - 1.0) * (E - 0.5 * r * (u*u + v*v)), 0.0);
    double a = std::sqrt(gamma_gas * p / r);
    return std::abs(u) + a;
};
auto wavespeed_y = [gamma_gas](auto U) {
    double r = U(0), ru = U(1), rv = U(2), E = U(3);
    double u = ru / r, v = rv / r;
    double p = std::max((gamma_gas - 1.0) * (E - 0.5 * r * (u*u + v*v)), 0.0);
    double a = std::sqrt(gamma_gas * p / r);
    return std::abs(v) + a;
};

// X-direction Rusanov
euler[0].cons_flux_function = [=](samurai::FluxValue<cfg>& flux,
                                   const samurai::StencilData<cfg>&,
                                   const samurai::StencilValues<cfg>& U) {
    constexpr std::size_t L = 0, R = 1;
    auto UL = U[L]; auto UR = U[R];
    double a = std::max(wavespeed_x(UL), wavespeed_x(UR));
    flux = 0.5 * (px(UL) + px(UR)) - 0.5 * a * (UR - UL);
};

// Y-direction Rusanov
euler[1].cons_flux_function = [=](samurai::FluxValue<cfg>& flux,
                                   const samurai::StencilData<cfg>&,
                                   const samurai::StencilValues<cfg>& U) {
    constexpr std::size_t L = 0, R = 1;
    auto UL = U[L]; auto UR = U[R];
    double a = std::max(wavespeed_y(UL), wavespeed_y(UR));
    flux = 0.5 * (py(UL) + py(UR)) - 0.5 * a * (UR - UL);
};

auto euler_op = samurai::make_flux_based_scheme(euler);

// Explicit step (e.g., TVD-RK3)
Qnp1 = Q - dt * euler_op(Q);
```

Choose `dt = cfl * dx / max(|u|+a, |v|+a)` aggregated over the mesh. For robustness, add positivity-preserving limiters and carefully selected boundary conditions (e.g., characteristic outflow), which you can implement as function-valued Dirichlet/Neumann.

### Porous Medium equation (nonlinear diffusion; explicit)

Physics

- u_t = ∆(u^m) with m > 1. In divergence form: u_t = ∇·(m u^{m−1} ∇u).
- Explicit FV using a heterogeneous diffusion coefficient K(u) = m u^{m−1} applied componentwise.

Implementation sketch

```cpp
auto u    = samurai::make_scalar_field<double>("u", mesh);
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
unp1.copy_bc_from(u);

// Coefficient field (diagonal tensor), filled each step from current u
auto K = samurai::make_scalar_field<samurai::DiffCoeff<decltype(u)::mesh_t::dim>>("K", mesh);

double m = 2.0;         // e.g., PME with exponent m
double k_eps = 1e-12;   // small clamp to avoid degeneracy at u≈0

auto update_K = [&]() {
    samurai::for_each_cell(u.mesh(), [&](const auto& cell) {
        double uc = std::max(u[cell], 0.0);
        double k  = m * std::pow(uc + k_eps, m - 1.0);
        samurai::DiffCoeff<decltype(u)::mesh_t::dim> kd; kd.fill(k);
        K[cell] = kd;
    });
};

update_K();
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);

// Explicit Euler (or embed in TVD-RK)
double dx  = u.mesh().cell_length(u.mesh().max_level());
double cfl = 0.4;
// Conservative dt from a bound on K
double Kmax = 0.0; samurai::for_each_cell(u.mesh(), [&](const auto& cell){
    for (std::size_t d = 0; d < decltype(u)::mesh_t::dim; ++d) Kmax = std::max(Kmax, K[cell](d));
});
double dt = cfl * (dx * dx) / (std::pow(2.0, (int)decltype(u)::mesh_t::dim) * std::max(Kmax, 1e-12));

unp1 = u - dt * diff(u); // -diff(u) ⇒ +∇·(K∇u) = ∆(u^m)

// Next step: swap(u, unp1); update_K(); rebuild diff (if needed);
```

Use zero-flux Neumann BCs to conserve mass, or problem-specific Dirichlet values. Since K depends on u, recompute K after each update and adjust `dt` if the maximum coefficient increases.

### Ideal MHD 2D (explicit Rusanov flux) — very complex

Physics

- State U = [ρ, ρu, ρv, ρw, Bx, By, Bz, E]^T in 2D; z-components may be nonzero (out-of-plane).
- Pressure p = (γ − 1)(E − ½ ρ(u²+v²+w²) − ½(Bx²+By²+Bz²)).
- Fluxes (x,y) include magnetic pressure and Maxwell stresses. Rusanov flux with wave speed a = |u_n| + c_f, where the fast magnetosonic speed satisfies
  c_f² = ½ [ a_s² + v_A² + √((a_s² + v_A²)² − 4 a_s² v_{A,n}²) ], with a_s² = γ p/ρ, v_A² = (Bx²+By²+Bz²)/ρ, v_{A,n}² = B_n²/ρ.
- Managing ∇·B ≈ 0 requires special care (e.g., GLM cleaning or Powell 8-wave). Below is a baseline explicit Rusanov scheme; add divergence control as needed.

Implementation sketch

```cpp
static constexpr std::size_t dim = 2;
static constexpr std::size_t n_comp = 8; // [rho, rho_u, rho_v, rho_w, Bx, By, Bz, E]
auto U    = samurai::make_vector_field<double, n_comp>("U", mesh);
auto Unp1 = samurai::make_vector_field<double, n_comp>("Unp1", mesh);
Unp1.copy_bc_from(U);

double gamma_gas = 1.4;

using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, decltype(U), decltype(U)>;
samurai::FluxDefinition<cfg> mhd;

auto primitives = [gamma_gas](auto W) {
    double r = W(0), ru = W(1), rv = W(2), rw = W(3);
    double Bx = W(4), By = W(5), Bz = W(6), E = W(7);
    double invr = 1.0 / r;
    double ux = ru * invr, uy = rv * invr, uz = rw * invr;
    double v2 = ux*ux + uy*uy + uz*uz;
    double B2 = Bx*Bx + By*By + Bz*Bz;
    double p  = (gamma_gas - 1.0) * (E - 0.5 * r * v2 - 0.5 * B2);
    return std::tuple{r, ux, uy, uz, Bx, By, Bz, E, p, v2, B2};
};

auto cf_x = [gamma_gas, primitives](auto W) {
    auto [r, ux, uy, uz, Bx, By, Bz, E, p, v2, B2] = primitives(W);
    double as2 = gamma_gas * std::max(p, 0.0) / r;
    double b2  = B2 / r;
    double bn2 = Bx*Bx / r;
    double term = (as2 + b2);
    double disc = std::max(term*term - 4.0 * as2 * bn2, 0.0);
    double cf2  = 0.5 * (term + std::sqrt(disc));
    return std::abs(ux) + std::sqrt(std::max(cf2, 0.0));
};
auto cf_y = [gamma_gas, primitives](auto W) {
    auto [r, ux, uy, uz, Bx, By, Bz, E, p, v2, B2] = primitives(W);
    double as2 = gamma_gas * std::max(p, 0.0) / r;
    double b2  = B2 / r;
    double bn2 = By*By / r;
    double term = (as2 + b2);
    double disc = std::max(term*term - 4.0 * as2 * bn2, 0.0);
    double cf2  = 0.5 * (term + std::sqrt(disc));
    return std::abs(uy) + std::sqrt(std::max(cf2, 0.0));
};

auto Fx = [gamma_gas, primitives](auto W) {
    auto [r, ux, uy, uz, Bx, By, Bz, E, p, v2, B2] = primitives(W);
    double pt = p + 0.5 * B2;
    double uB = ux*Bx + uy*By + uz*Bz;
    samurai::FluxValue<cfg> F;
    F(0) = r * ux;
    F(1) = r * ux * ux + pt - Bx * Bx;
    F(2) = r * ux * uy - Bx * By;
    F(3) = r * ux * uz - Bx * Bz;
    F(4) = 0.0;
    F(5) = ux * By - uy * Bx;
    F(6) = ux * Bz - uz * Bx;
    F(7) = (E + pt) * ux - Bx * uB;
    return F;
};
auto Fy = [gamma_gas, primitives](auto W) {
    auto [r, ux, uy, uz, Bx, By, Bz, E, p, v2, B2] = primitives(W);
    double pt = p + 0.5 * B2;
    double uB = ux*Bx + uy*By + uz*Bz;
    samurai::FluxValue<cfg> G;
    G(0) = r * uy;
    G(1) = r * ux * uy - Bx * By;
    G(2) = r * uy * uy + pt - By * By;
    G(3) = r * uy * uz - By * Bz;
    G(4) = uy * Bx - ux * By;
    G(5) = 0.0;
    G(6) = uy * Bz - uz * By;
    G(7) = (E + pt) * uy - By * uB;
    return G;
};

// X-direction Rusanov flux
mhd[0].cons_flux_function = [=](samurai::FluxValue<cfg>& flux,
                                const samurai::StencilData<cfg>&,
                                const samurai::StencilValues<cfg>& Uc) {
    constexpr std::size_t L = 0, R = 1;
    auto UL = Uc[L]; auto UR = Uc[R];
    double a = std::max(cf_x(UL), cf_x(UR));
    flux = 0.5 * (Fx(UL) + Fx(UR)) - 0.5 * a * (UR - UL);
};

// Y-direction Rusanov flux
mhd[1].cons_flux_function = [=](samurai::FluxValue<cfg>& flux,
                                const samurai::StencilData<cfg>&,
                                const samurai::StencilValues<cfg>& Uc) {
    constexpr std::size_t L = 0, R = 1;
    auto UL = Uc[L]; auto UR = Uc[R];
    double a = std::max(cf_y(UL), cf_y(UR));
    flux = 0.5 * (Fy(UL) + Fy(UR)) - 0.5 * a * (UR - UL);
};

auto mhd_op = samurai::make_flux_based_scheme(mhd);

// Explicit update (e.g., SSPRK3)
Unp1 = U - dt * mhd_op(U);
```

Time step: `dt = cfl * dx / max(|u| + c_f, |v| + c_f)` over the mesh. Choose boundary conditions consistent with the physical problem (e.g., conducting walls, periodic). For robustness: positivity floors on ρ and p; add divergence control (GLM or Powell) via extra variables or local source operators, respectively.
