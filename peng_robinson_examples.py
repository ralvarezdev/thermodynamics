from peng_robinson import (
    Component,
    MixtureComponent,
    PengRobinson,
    Pressure,
    PressureConverter,
    PressureUnit,
    Temperature,
    TemperatureConverter,
    TemperatureUnit,
)

# Example 1
component1 = Component(
    None,
    Temperature(417.2, TemperatureUnit.K),
    Pressure(77.1, PressureUnit.bar),
    0.069,
)

t = Temperature(343.15, TemperatureUnit.K)
p = Pressure(22.49, PressureUnit.bar)

Z_liquid, Z_vapor = PengRobinson.calculate_z_pure(component1, t, p)
print(f"Compressibility factor Z (saturated liquid): {Z_liquid}")
print(f"Compressibility factor Z (saturated vapor): {Z_vapor}")

# Example 2
methane = Component(
    "methane",
    Temperature(190.6, TemperatureUnit.K),
    Pressure(4590000, PressureUnit.Pa),
    0.012,
)
methane_mixture = MixtureComponent(
    methane,
    0.2,
)

ethane = Component(
    "ethane",
    Temperature(305.3, TemperatureUnit.K),
    Pressure(4872000, PressureUnit.Pa),
    0.100,
)
ethane_mixture = MixtureComponent(
    ethane,
    0.3,
)

propane = Component(
    "propane",
    Temperature(369.8, TemperatureUnit.K),
    Pressure(4248000, PressureUnit.Pa),
    0.152,
)
propane_mixture = MixtureComponent(
    propane,
    0.7,
)

mixture_components = [methane_mixture, ethane_mixture, propane_mixture]
t = Temperature(303.15, TemperatureUnit.K)
p = Pressure(400000, PressureUnit.Pa)

Z_liquid, Z_vapor = PengRobinson.calculate_z_mixture(mixture_components, t, p)
print(f"Compressibility factor Z (saturated liquid): {Z_liquid}")
print(f"Compressibility factor Z (saturated vapor): {Z_vapor}")
