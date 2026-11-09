"""AgriTireSim : traction sur sol meuble, usure sur route et empreinte FEM d'un pneu agricole."""
from .soil import Soil, SOILS, get_soil
from .terramechanics import Tire, traction, traction_curve

__all__ = ["Soil", "SOILS", "get_soil", "Tire", "traction", "traction_curve"]
__version__ = "0.1.0"
