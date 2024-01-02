from enum import Enum


class VeinImageType(Enum):
    GENUINE = "genuine"
    SPOOFED = "spoofed"
    SYNTHETIC_CYCLE = "spoofed_synthethic_cyclegan"
    SYNTHETIC_DIST = "spoofed_synthethic_distancegan"
    SYNTHETIC_DRIT = "spoofed_synthethic_drit"
    SYNTHETIC_STAR = "spoofed_synthethic_stargan-v2"
    SYNTHETIC_ALL = "spoofed_synthethic_*"
