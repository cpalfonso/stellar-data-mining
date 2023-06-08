def format_feature_name(s):
    """Make feature names a bit easier to read in plots."""
    s = s.replace("_", " ")
    s = s[0].capitalize() + s[1:]

    s = s.replace("Co2", r"$\mathrm{CO_2}$")
    s = s.replace("m^3/m^2", r"$\mathrm{m^3\;m^{-2}}$")
    s = s.replace("(m^3", r"($\mathrm{m^3}$")
    s = s.replace("(m^2", r"($\mathrm{m^2}$")
    s = s.replace("m^3)", r"$\mathrm{m^3}$)")
    s = s.replace("m^2)", r"$\mathrm{m^2}$)")
    s = s.replace("/yr", r"$\mathrm{\;{yr}^{-1}}$")
    s = s.replace("/Myr", r"$\mathrm{\;{Myr}^{-1}}$")
    s = s.replace("degrees", r"$\degree$")

    return s
