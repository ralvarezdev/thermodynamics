from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy import ndarray, dtype


# Thermodynamic constants
@dataclass(frozen=True)
class Thermodynamic:
    R: float = 8.314  # Gas constant in J/(mol·K)


# Temperature units
class TemperatureUnit(Enum):
    K = 1
    C = 2
    F = 3


# Temperature class
@dataclass(frozen=True)
class Temperature:
    value: int | float
    unit: TemperatureUnit


# Temperature class to convert temperatures
class TemperatureConverter:
    # Methods to convert temperatures
    @staticmethod
    def convert_celsius_to_kelvin(t: int | float) -> int | float:
        return t + 273.15

    @staticmethod
    def convert_kelvin_to_celsius(t: int | float) -> int | float:
        return t - 273.15

    @staticmethod
    def convert_fahrenheit_to_kelvin(t: int | float) -> int | float:
        return (t - 32) * 5 / 9 + 273.15

    @staticmethod
    def convert_kelvin_to_fahrenheit(t: int | float) -> int | float:
        return (t - 273.15) * 9 / 5 + 32

    @staticmethod
    def convert_celsius_to_fahrenheit(t: int | float) -> int | float:
        return t * 9 / 5 + 32

    @staticmethod
    def convert_fahrenheit_to_celsius(t: int | float) -> int | float:
        return (t - 32) * 5 / 9

    @staticmethod
    def convert_temperature(t: Temperature, to_unit: TemperatureUnit) -> int | float:
        if t.unit == to_unit:
            return t.value

        if t.unit == TemperatureUnit.K:
            return (
                TemperatureConverter.convert_kelvin_to_celsius(t.value)
                if to_unit == TemperatureUnit.C
                else TemperatureConverter.convert_kelvin_to_fahrenheit(t.value)
            )

        elif t.unit == TemperatureUnit.C:
            return (
                TemperatureConverter.convert_celsius_to_kelvin(t.value)
                if to_unit == TemperatureUnit.K
                else TemperatureConverter.convert_celsius_to_fahrenheit(t.value)
            )

        else:
            return (
                TemperatureConverter.convert_fahrenheit_to_kelvin(t.value)
                if to_unit == TemperatureUnit.K
                else TemperatureConverter.convert_fahrenheit_to_celsius(t.value)
            )


# Pressure units
class PressureUnit(Enum):
    Pa = 1
    bar = 2
    atm = 3


# Pressure class
@dataclass(frozen=True)
class Pressure:
    value: int | float
    unit: PressureUnit


# Pressure class to convert pressures
class PressureConverter:
    # Methods to convert pressures
    @staticmethod
    def convert_pa_to_bar(p: int | float) -> float:
        return p / 100000

    @staticmethod
    def convert_bar_to_pa(p: int | float) -> float:
        return p * 100000

    @staticmethod
    def convert_pa_to_atm(p: int | float) -> float:
        return p / 101325

    @staticmethod
    def convert_atm_to_pa(p: int | float) -> float:
        return p * 101325

    @staticmethod
    def convert_bar_to_atm(p: int | float) -> float:
        return p / 1.01325

    @staticmethod
    def convert_atm_to_bar(p: int | float) -> float:
        return p * 1.01325

    @staticmethod
    def convert_pressure(p: Pressure, to_unit: PressureUnit) -> int | float:
        if p.unit == to_unit:
            return p.value

        if p.unit == PressureUnit.Pa:
            return (
                PressureConverter.convert_pa_to_bar(p.value)
                if to_unit == PressureUnit.bar
                else PressureConverter.convert_pa_to_atm(p.value)
            )

        elif p.unit == PressureUnit.bar:
            return (
                PressureConverter.convert_bar_to_pa(p.value)
                if to_unit == PressureUnit.Pa
                else PressureConverter.convert_bar_to_atm(p.value)
            )

        else:
            return (
                PressureConverter.convert_atm_to_pa(p.value)
                if to_unit == PressureUnit.Pa
                else PressureConverter.convert_atm_to_bar(p.value)
            )


# Component class
@dataclass(frozen=True)
class Component:
    name: str | None
    tc: Temperature
    pc: Pressure
    omega: int | float

    @staticmethod
    def get_numeric_error(var: str) -> str:
        return f"The component '{var}' property must be a number."

    @staticmethod
    def is_number(var: int | float) -> bool:
        return isinstance(var, int) or isinstance(var, float)

    # Component properties validation
    def check_component_properties(self) -> None:
        for [arg_name, arg_value] in [
            ["Tc", self.tc.value],
            ["Pc", self.pc.value],
            ["omega", self.omega],
        ]:
            if not Component.is_number(arg_value):
                raise ValueError(Component.get_numeric_error(arg_name))


@dataclass(frozen=True)
class MixtureComponent:
    component: Component
    fraction: float

    def check_component_properties(self) -> None:
        self.component.check_component_properties()

        for [arg_name, arg_value] in [["Fraction", self.fraction]]:
            if not Component.is_number(arg_value):
                raise ValueError(Component.get_numeric_error(arg_name))


class PengRobinson:
    @staticmethod
    def __calculate_a(tc: Temperature, pc: Pressure) -> float:
        # Convert critical pressure to Pa
        pc_pa = PressureConverter.convert_pressure(pc, PressureUnit.Pa)

        # Convert critical temperature to Kelvin
        tc_kelvin = TemperatureConverter.convert_temperature(tc, TemperatureUnit.K)

        return 0.45724 * (Thermodynamic.R**2) * (tc_kelvin**2) / pc_pa

    @staticmethod
    def __calculate_b(tc: Temperature, pc: Pressure) -> float:
        # Convert critical pressure to Pa
        pc_pa = PressureConverter.convert_pressure(pc, PressureUnit.Pa)

        # Convert critical temperature to Kelvin
        tc_kelvin = TemperatureConverter.convert_temperature(tc, TemperatureUnit.K)

        return 0.07780 * Thermodynamic.R * tc_kelvin / pc_pa

    @staticmethod
    def __calculate_alpha(t: Temperature, tc: Temperature, omega: int | float) -> float:
        # Convert temperatures to Kelvin
        t_kelvin = TemperatureConverter.convert_temperature(t, TemperatureUnit.K)
        tc_kelvin = TemperatureConverter.convert_temperature(tc, TemperatureUnit.K)

        tr_kelvin = t_kelvin / tc_kelvin
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

        return (1 + m * (1 - np.sqrt(tr_kelvin))) ** 2

    @staticmethod
    def __calculate_A(
        A: int | float,
        alpha: int | float,
        p: Pressure,
        t: Temperature,
    ) -> float:
        # Convert system pressure to Pa
        p_pa = PressureConverter.convert_pressure(p, PressureUnit.Pa)

        # Convert temperature to Kelvin
        t_kelvin = TemperatureConverter.convert_temperature(t, TemperatureUnit.K)

        return A * alpha * p_pa / (Thermodynamic.R * t_kelvin) ** 2

    @staticmethod
    def __calculate_B(B: int | float, p: Pressure, t: Temperature) -> float:
        # Convert system pressure to Pa
        p_pa = PressureConverter.convert_pressure(p, PressureUnit.Pa)

        # Convert temperature to Kelvin
        t_kelvin = TemperatureConverter.convert_temperature(t, TemperatureUnit.K)

        return B * p_pa / (Thermodynamic.R * t_kelvin)

    @staticmethod
    def __calculate_z_roots(A: int | float, B: int | float) -> ndarray[Any, dtype[Any]]:
        # Coefficients of the cubic Peng-Robinson equation
        coefficients = [1, -(1 - B), A - 2 * B - 3 * B**2, -(A * B - B**2 - B**3)]

        # Solving the cubic equation for Z
        z_roots = np.roots(coefficients)

        # Considering the real roots only
        z_roots = z_roots[np.isreal(z_roots)]

        # Sorting the real roots
        z_roots = np.sort(z_roots.real)

        return z_roots

    @staticmethod
    def __get_min_max(z_roots: ndarray[Any, dtype]) -> tuple[ndarray[Any, dtype], ndarray[Any, dtype]]:
        # There can be more than one real root
        z_liquid = z_roots[0]  # Minor root
        z_vapor = z_roots[-1]  # Major root

        return z_liquid, z_vapor

    @staticmethod
    def calculate_z_pure(c: Component, t: Temperature, p: Pressure) -> tuple[ndarray[Any, dtype], ndarray[Any, dtype]]:
        # Component properties
        c.check_component_properties()

        a = PengRobinson.__calculate_a(c.tc, c.pc)
        b = PengRobinson.__calculate_b(c.tc, c.pc)
        alpha = PengRobinson.__calculate_alpha(t, c.tc, c.omega)

        A = PengRobinson.__calculate_A(a, alpha, p, t)
        B = PengRobinson.__calculate_B(b, p, t)

        z_roots = PengRobinson.__calculate_z_roots(A, B)
        return PengRobinson.__get_min_max(z_roots)

    @staticmethod
    def calculate_z_mixture(
        mixture_components: list[MixtureComponent], t: Temperature, p: Pressure
    ) -> tuple[ndarray[Any, dtype], ndarray[Any, dtype]]:
        a = np.array(
            [
                PengRobinson.__calculate_a(m.component.tc, m.component.pc)
                for m in mixture_components
            ]
        )
        b = np.array(
            [
                PengRobinson.__calculate_b(m.component.tc, m.component.pc)
                for m in mixture_components
            ]
        )
        alpha = np.array(
            [
                PengRobinson.__calculate_alpha(t, m.component.tc, m.component.omega)
                for m in mixture_components
            ]
        )

        # Calculate the mixed parameters a_mix and b_mix
        a_mix = 0
        for i in range(0, len(mixture_components)):
            for j in range(0, len(mixture_components)):
                a_mix += (
                    mixture_components[i].fraction
                    * mixture_components[j].fraction
                    * np.sqrt(a[i] * a[j])
                )

        fractions = [m.fraction for m in mixture_components]
        b_mix = np.dot(fractions, b)
        alpha_mix = np.dot(fractions, alpha)

        # Calculate A_mix and B_mix
        A_mix = PengRobinson.__calculate_A(a_mix, alpha_mix, p, t)
        B_mix = PengRobinson.__calculate_B(b_mix, p, t)

        # Calculate Z
        z_roots = PengRobinson.__calculate_z_roots(A_mix, B_mix)
        return PengRobinson.__get_min_max(z_roots)
