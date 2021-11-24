# encoding: utf-8
"""
anonymous
anonymous
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .cuhk03 import CUHK03
from .cuhk03np_detected import CUHK03NPDetected
from .cuhk03np_labeled import CUHK03NPLabeled
from .dukemtmcreid import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .AirportALERT import AirportALERT
from .iLIDS import iLIDS
from .pku import PKU
from .prai import PRAI
from .prid import PRID
from .grid import GRID
from .saivt import SAIVT
from .sensereid import SenseReID
from .sysu_mm import SYSU_mm
from .thermalworld import Thermalworld
from .pes3d import PeS3D
from .caviara import CAVIARa
from .viper import VIPeR
from .lpw import LPW
from .shinpuhkan import Shinpuhkan
from .wildtracker import WildTrackCrop
from .cuhk_sysu import cuhkSYSU
from .personX import PersonX
from .personX_V2 import PersonX_V2
# from .personX3D import PersonX3D
from .personX3DGenerated import PersonX3DGenerated
from .make_human import MakeHuman
from .rand_person import RandPerson
from .unreal_person import UnrealPerson
from .market1501_custom import Market1501Custom
from .msmt17_custom import MSMT17Custom


# Vehicle re-id datasets
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
