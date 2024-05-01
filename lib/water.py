import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

__all__ = [
    # Combined:
    "calculate_water_thickness",
    # Sediments:
    "sedimentary_pore_water",
    "sedimentary_bound_water",
    # Crust:
    "crustal_pore_water",
    "crustal_bound_water",
    # Mantle lithosphere:
    "mantle_lithosphere_water",
    # Base of the lithosphere:
    "base_lithosphere_water",
]

NEW_COLUMNS = {
    "sedimentary_pore_water_thickness (m)",
    "sedimentary_bound_water_thickness (m)",
    "crustal_pore_water_thickness (m)",
    "crustal_bound_water_thickness (m)",
    "mantle_lithosphere_water_thickness (m)",
    "base_lithosphere_water_thickness (m)",
    "total_water_thickness (m)",
}

RHO_WATER = 1020.0  # kg/m^3

# Sedimentary water parameters
DEFAULT_SURFACE_POROSITY = 0.66
DEFAULT_POROSITY_DECAY = 1333

# Mantle lithosphere water parameters
MANTLE_U_REF = 3.0
MANTLE_D_REF = 1.504
MANTLE_T_REF = 1350.0
MANTLE_C_REF = 19.0
MANTLE_V_REF = 100.0
MANTLE_DELTA_V_REF = 20.0
MANTLE_SPREADRATE_REF = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
MANTLE_H2O_REF_LOW = np.array([3, 2.5, 1.8, 1.4, 1, 0.75, 0.5, 0.5, 0.5, 0.5]) * 1.e5
MANTLE_H2O_REF_HIGH = np.array([3, 2.5, 1.7, 1.1, 0.6, 0.45, 0.5, 0.4, 0.3, 0.25]) * 1.e5


def calculate_water_thickness(
    sediment_thickness="sediment_thickness (m)",
    seafloor_age="seafloor_age (Ma)",
    seafloor_spreadrate="seafloor_spreading_rate (km/Myr)",
    data=None,
    porosity_decay=DEFAULT_POROSITY_DECAY,
    surface_porosity=DEFAULT_SURFACE_POROSITY,
    sedimentary_bound_water_fraction=0.07,
    sediment_density=1700.0,
):
    if isinstance(data, str):
        data = pd.read_csv(data)
    if isinstance(data, pd.DataFrame):
        sediment_thickness = data[sediment_thickness]
        seafloor_age = data[seafloor_age]
        seafloor_spreadrate = data[seafloor_spreadrate]
    else:
        sediment_thickness = np.array(sediment_thickness)
        seafloor_age = np.array(seafloor_age)
        seafloor_spreadrate = np.array(seafloor_spreadrate)
    seafloor_spreadrate = (
        seafloor_spreadrate  # km/Myr
        * 1.0e5  # cm/Myr
        * 1.0e-6  # cm/yr
        * 0.5  # half-spreading rate
    )

    out = {}
    out["sedimentary_pore_water_thickness (m)"] = sedimentary_pore_water(
        sediment_thickness=sediment_thickness,
        porosity_decay=porosity_decay,
        surface_porosity=surface_porosity,
    )
    out["sedimentary_bound_water_thickness (m)"] = sedimentary_bound_water(
        sediment_thickness=sediment_thickness,
        pore_water_thickness=out["sedimentary_pore_water_thickness (m)"],
        p=sedimentary_bound_water_fraction,
        rho=sediment_density,
    )
    out["crustal_pore_water_thickness (m)"] = crustal_pore_water(
        age=seafloor_age,
    )
    out["crustal_bound_water_thickness (m)"] = crustal_bound_water(
        age=seafloor_age,
    )
    out["mantle_lithosphere_water_thickness (m)"] = mantle_lithosphere_water(
        spreadrate=seafloor_spreadrate,
    )
    out["base_lithosphere_water_thickness (m)"] = base_lithosphere_water(
        u=seafloor_spreadrate,
    )
    for key in list(out.keys()):
        if not isinstance(out[key], pd.Series):
            out[key] = pd.Series(out[key])
    for key, value in out.items():
        value.name = key
    out = pd.DataFrame(out)
    out = out.replace({np.inf: np.nan, -np.inf: np.nan})
    out["total_water_thickness (m)"] = out.sum(
        axis=1,
        numeric_only=True,
        skipna=True,
    )
    if isinstance(data, pd.DataFrame):
        data = data.drop(
            columns=list(NEW_COLUMNS),
            errors="ignore",
        )
        return data.join(out)
    return out


### Sedimentary water
def sedimentary_pore_water(
    sediment_thickness,
    porosity_decay=DEFAULT_POROSITY_DECAY,
    surface_porosity=DEFAULT_SURFACE_POROSITY,
):
    """Calculate sediment pore water volume density.

    Parameters
    ----------
    sediment_thickness : float
        Thickness of sediment column (m).
    porosity_decay : float, default: 1333
        Porosity decay constant for pore space calculation.
    surface_porosity : float, default: 0.66
        Surface porosity value for pore space calculation.

    Returns
    -------
    float
        Sedimentary pore water volume-area density (m^3/m^2).
    """
    water_density = (
        porosity_decay
        * surface_porosity
        * (1 - np.exp(-1.0 * sediment_thickness / porosity_decay))
    )
    return np.clip(water_density, 0.0, None)


def sedimentary_bound_water(
    sediment_thickness,
    pore_water_thickness,
    p=0.07,
    rho=1700.0,
):
    """Calculate structurally-bound water in sediments.

    Parameters
    ----------
    sediment_thickness : float
        Thickness of sediment column (m).
    water_thickness : float
        Thickness of pore water in sediment column (m).
    p : float, default: 0.07
        Water (weight) fraction.
    rho : float, default: 1700.0
        Average sediment density (kg/m^3).

    Returns
    -------
    float
        Structurally-bound water volume-area density (m^3/m^2).
    """
    water_density = p * (sediment_thickness - pore_water_thickness) * rho
    return np.clip(water_density / RHO_WATER, 0.0, None)


### Crustal water
def crustal_pore_water(age):
    """Crustal pore water as a function of crustal age.

    Parameters
    ----------
    age : array_like
        Oceanic crust age (Ma).

    Returns
    -------
    array_like
        Total crustal pore water volume-area
        density (m^3/m^2).
    """
    phi_u = 7.8 + _macro_porosity(age)
    phi_l = 5.1 + _macro_porosity(age)/2
    phi_d = 2.2 + 0.84
    phi_g = 0.7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m_rho_u = 3.01 - 0.0631*np.log10(age)
        m_rho_l = 3.01 - 0.5*0.0631*np.log10(age)
    m_rho_d = 2.98
    m_rho_g = 2.99
    rho_u = _bulk_density(phi_u, m_rho_u)
    rho_l = _bulk_density(phi_l, m_rho_l)
    rho_d = _bulk_density(phi_d, m_rho_d)
    rho_g = _bulk_density(phi_g, m_rho_g)
    H2O_wt_u, H2O_wt_l, H2O_wt_d, H2O_wt_g = _total_pore_water_percent(age)
    flux_u = H2O_wt_u/100*rho_u*1e3*300
    flux_l = H2O_wt_l/100*rho_l*1e3*300
    flux_d = H2O_wt_d/100*rho_d*1e3*1400
    flux_g = H2O_wt_g/100*rho_g*1e3*5000
    return np.clip((flux_u + flux_l + flux_d + flux_g) / RHO_WATER, 0.0, None)


def crustal_bound_water(age):
    """Structurally-bound water as a function of crustal age.

    Parameters
    ----------
    age : array_like
        Oceanic crust age (Ma).

    Returns
    -------
    array_like
        Total structurally-bound crustal water
        volume-area density (m^3/m^2).
    """
    phi_u = 7.8 + _macro_porosity(age)
    phi_l = 5.1 + _macro_porosity(age)/2
    phi_d = 2.2 + 0.84
    phi_g = 0.7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m_rho_u = 3.01 - 0.0631*np.log10(age)
        m_rho_l = 3.01 - 0.5*0.0631*np.log10(age)
    m_rho_d = 2.98
    m_rho_g = 2.99
    rho_u = _bulk_density(phi_u, m_rho_u)
    rho_l = _bulk_density(phi_l, m_rho_l)
    rho_d = _bulk_density(phi_d, m_rho_d)
    rho_g = _bulk_density(phi_g, m_rho_g)
    H2O_wt_us, H2O_wt_ls, H2O_wt_ds, H2O_wt_gs = _total_structural_water_percent(age)
    H2O_wt_up, H2O_wt_lp, H2O_wt_dp, H2O_wt_gp = _total_pore_water_percent(age)
    flux_u = H2O_wt_us*(1.0 - 0.01*H2O_wt_up)/100*rho_u*1e3*300
    flux_l = H2O_wt_ls*(1.0 - 0.01*H2O_wt_lp)/100*rho_l*1e3*300
    flux_d = H2O_wt_ds*(1.0 - 0.01*H2O_wt_dp)/100*rho_d*1e3*1400
    flux_g = H2O_wt_gs*(1.0 - 0.01*H2O_wt_gp)/100*rho_g*1e3*5000
    return np.clip((flux_u + flux_l + flux_d + flux_g) / RHO_WATER, 0.0, None)


def _macro_porosity(age):
    """Crustal porosity as a function of seafloor age.

    Parameters
    ----------
    age : float
        Oceanic crust age (Ma).

    Returns
    -------
    float
        Oceanic crust porosity (%).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = 13.01 - 5.625 * np.log10(age)
    return out


def _bulk_density(rho, phi):
    """Calculate bulk density from density and porosity.

    Parameters
    ----------
    rho : float
        Rock density (g/cm^3).
    phi : float
        Porosity (%).

    Returns
    -------
    float
        Bulk density (g/cm^3).
    """
    return (
        0.01 * phi * 1.02
        + (1.0 - 0.01 * phi) * rho
    )


def _total_structural_water_percent(age):
    """Structurally-bound water as a function of crustal age.

    Parameters
    ----------
    age : array_like
        Oceanic crust age (Ma).

    Returns
    -------
    upper, lower, dykes, gabbro : array_like
        Structurally-bound crustal water (wt %)
        for upper and lower extrusive, sheeted dykes,
        and gabbro components of oceanic crust.
    """
    ones = np.ones_like(age)
    phi_u = _macro_porosity(age)
    phi_l = phi_u/2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rho_u = 3.01 - 0.0631*np.log10(age)
        rho_l = 3.01 - 0.5*0.0631*np.log10(age)
    rho_d = 2.98
    rho_g = 2.99

    structural_water_wt_pc = lambda rho, phi: (
        103.1
        - 34.27 * rho
        + 0.17 * (1.031 - phi)
    )

    H2O_wt_u = structural_water_wt_pc(rho_u, phi_u)
    H2O_wt_l = structural_water_wt_pc(rho_l, phi_l)
    H2O_wt_d = 1.76
    H2O_wt_g = 0.79
    return H2O_wt_u*ones, H2O_wt_l*ones, H2O_wt_d*ones, H2O_wt_g*ones


def _total_pore_water_percent(age):
    """Pore water as a function of crustal age.

    Parameters
    ----------
    age : array_like
        Oceanic crust age (Ma).

    Returns
    -------
    upper, lower, dykes, gabbro : array_like
        Crustal pore water (wt %) for upper and
        lower extrusive, sheeted dykes, and
        gabbro components of oceanic crust.
    """
    ones = np.ones_like(age)
    phi_u = 7.8 + _macro_porosity(age)
    phi_l = 5.1 + _macro_porosity(age)/2
    phi_d = 2.2 + 0.84
    phi_g = 0.7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m_rho_u = 3.01 - 0.0631*np.log10(age)
        m_rho_l = 3.01 - 0.5*0.0631*np.log10(age)
    m_rho_d = 2.98
    m_rho_g = 2.99
    rho_u = _bulk_density(phi_u, m_rho_u)
    rho_l = _bulk_density(phi_l, m_rho_l)
    rho_d = _bulk_density(phi_d, m_rho_d)
    rho_g = _bulk_density(phi_g, m_rho_g)
    H2O_wt_u = phi_u*1.02/rho_u
    H2O_wt_l = phi_l*1.02/rho_l
    H2O_wt_d = phi_d*1.02/rho_d
    H2O_wt_g = phi_g*1.02/rho_g
    return H2O_wt_u*ones, H2O_wt_l*ones, H2O_wt_d*ones, H2O_wt_g*ones


### Mantle lithosphere water
def mantle_lithosphere_water(spreadrate):
    """Calculate water content in the mantle lithosphere.

    Parameters
    ----------
    spreadrate : float
        Half-spreading rate of oceanic lithosphere (cm/yr).

    Returns
    -------
    float
        Water volume-area density in mantle lithosphere (kg/m^2).
    """
    mantle_water_low = _mantle_fit_curve(MANTLE_SPREADRATE_REF, MANTLE_H2O_REF_LOW)
    mantle_water_high = _mantle_fit_curve(MANTLE_SPREADRATE_REF, MANTLE_H2O_REF_HIGH)
    return np.clip(
        0.5 * (mantle_water_low(spreadrate) + mantle_water_high(spreadrate)) / RHO_WATER,
        0.0, None,
    )


def _mantle_ridge_outflux(
    u=MANTLE_U_REF,
    d=MANTLE_D_REF,
    T=MANTLE_T_REF,
    c=MANTLE_C_REF,
    v=MANTLE_V_REF,
):
    u = np.abs(u)
    A = 0.9919
    B_u = 0.3162
    B_d = -0.3739
    B_T = 0.0089
    B_c = 0.0294
    B_v = 0.0095
    return (
        A
        + B_u * (u - MANTLE_U_REF)
        + B_d * (d - MANTLE_D_REF)
        + B_T * (T - MANTLE_T_REF)
        + B_c * (c - MANTLE_C_REF)
        + B_v * (v - MANTLE_V_REF)
    )


def _mantle_func_to_fit(s, a, b):
    """Curve to fit for water content in the mantle lithosphere.

    Parameters
    ----------
    s : float
        Half-spreading rate of oceanic lithosphere (cm/yr).
    a, b : float
        Constants to fit.

    Returns
    -------
    float
    """
    return a / (s + b)


def _mantle_fit_curve(s, h, **kwargs):
    """Fit the curve for water content in the mantle lithosphere.

    Parameters
    ----------
    s : array_like
        Reference values for half-spreading rate of oceanic
        lithosphere (cm/yr).
    h : array_like
        Reference values for mantle lithosphere H2O content.
    kwargs : dict
        Keyword arguments to pass to `scipy.optimize.curve_fit`.

    Returns
    -------
    callable
        Function which takes one parameter (half-spreading rate)
        and returns the mantle lithosphere water content.
    """
    # return curve_fit(func_to_fit, s, h, **kwargs)[0]
    a, b = curve_fit(_mantle_func_to_fit, s, h, **kwargs)[0]
    def f(s):
        """Calculate water content in the mantle lithosphere.

        Parameters
        ----------
        s : float
            Half-spreading rate of oceanic lithosphere (cm/yr).

        Returns
        -------
        float
            Water area density in mantle lithosphere (kg/m^2).
        """
        return _mantle_func_to_fit(s, a, b)
    return f


### Base lithosphere water
def base_lithosphere_water(
    u=MANTLE_U_REF,
    d=MANTLE_D_REF,
    T=MANTLE_T_REF,
    c=MANTLE_C_REF,
    v=MANTLE_V_REF,
):
    """Water taken up by the base of the lithosphere.

    Parameters
    ----------
    u : float
        Half-spreading rate of oceanic lithosphere (cm/yr).
    d : float
        Compaction length.
    T : float
        Temperature of the mantle (K).
    c : float
        Mantle fertility.
    v : float
        Mantle H2O content.

    Returns
    -------
    float
        Water uptake into the lithosphere during oceanic crust
        formation along mid-ocean ridges (m^3/m^2).
    """
    u = np.abs(u)
    A = 0.8807
    B_u = 0.262
    B_d = 0.3589
    B_T = -0.00029097
    B_c = 0.0075
    B_v = 0.0054
    C_v = 0.000035532
    out = (
        A
        + B_u * (u - MANTLE_U_REF)
        + B_d * (d - MANTLE_D_REF)
        + B_T * (T - MANTLE_T_REF)
        + B_c * (c - MANTLE_C_REF)
        + B_v * (v - MANTLE_V_REF)
        + C_v * (v ** 2 - MANTLE_V_REF ** 2)
    )  # t/m/yr
    out = (
        out # t/m/yr
        / (u * 1.0e-2) # convert from cm/yr to m/yr
    ) # t/m^2
    out = (
        (out * 1.0e3)  # kg/m^2
        / RHO_WATER  # kg/m^3
    )  # m^3/m^2
    return np.clip(out, 0.0, None)
